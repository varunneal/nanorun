"""nanorun daemon — experiment execution, queue management, and Hub publishing.

Runs on an SSH-managed GPU machine or as a local-session daemon. Accepts commands
over WebSocket RPC, executes experiments in tmux, monitors completion, and
publishes durable artifacts.
"""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: uv pip install websockets", file=sys.stderr)
    sys.exit(1)

from nanorun.rpc_types import (
    RPC_PORT, ErrorCode, Event, EventMessage, Method, Request, Response, parse_message,
)
from nanorun.script_manifest import (
    ManifestError,
    ScriptManifest,
    compute_script_hash,
    parse_script_manifest,
)

try:
    from nanorun import hub
    HUB_AVAILABLE = True
except Exception:
    HUB_AVAILABLE = False

REPO_DIR = Path.cwd()
DAEMON_DIR = REPO_DIR / ".daemon"
MAPPINGS_DIR = DAEMON_DIR / "mappings"
OUTPUT_DIR = DAEMON_DIR / "output"
IMPORTS_DIR = DAEMON_DIR / "imports"
QUEUE_FILE = DAEMON_DIR / "queue.txt"
STATE_FILE = DAEMON_DIR / "state.json"
PID_FILE = DAEMON_DIR / "daemon.pid"
PENDING_WEIGHTS_FILE = DAEMON_DIR / "pending_weights.json"
LOGS_DIR = REPO_DIR / "logs"
ARTIFACTS_DIR = LOGS_DIR
MAPPINGS_LOG_DIR = ARTIFACTS_DIR / "mappings"
QUEUE_LOG_DIR = ARTIFACTS_DIR / "queue"
RPC_LISTEN_PORT = RPC_PORT
RPC_LISTEN_HOST = "localhost"
ENDPOINT_FILE: Optional[Path] = None
DEVICE_LOCK_FILE: Optional[Path] = None

TMUX_SESSION = "nanorun"
HUB_SYNC_INTERVAL_S = 15
QUEUE_PUSH_DEBOUNCE_S = 0.3  # coalesce bursts of queue changes into one event-driven push
EXPERIMENT_POLL_INTERVAL_S = 2
START_BLOCK_ALERT_RESOURCE_S = 600  # GPU memory release normally clears within seconds
START_BLOCK_ALERT_INFRASTRUCTURE_S = 90  # nvidia-smi/tmux failures will not self-heal
WEIGHT_STALENESS_S = 3
CODE_HASH_LENGTH = 12
GIT_HASH_LENGTH = 12
TMUX_WINDOW_NAME_MAX = 40
LOG_TAIL_BYTES = 50_000
MAPPINGS_SEGMENT_LINES = 500
QUEUE_SEGMENT_LINES = 500

RUN_ID_PATTERN = re.compile(
    r"logs/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\.txt"
)
STEP_PATTERN = re.compile(r"^step:(\d+)/(\d+)", re.MULTILINE)
METRIC_STEP_PATTERN = re.compile(r"step:(\d+)/(\d+)")
LOSS_PATTERN = re.compile(
    r"(?:val_loss|train_loss):\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
)
SEGMENT_FILE_PATTERN = re.compile(r"^mappings-(\d{6})\.jsonl$")
QUEUE_SEGMENT_FILE_PATTERN = re.compile(r"^queue-(\d{6})\.jsonl$")


class StartDisposition(str, Enum):
    """Classifies a start_experiment outcome by its correct queue response.

    RETRY_* failures are environmental — they would block any item, so the
    queue head stays in place and the monitor retries next tick. REJECT_ITEM
    is a deterministic fault of the item itself: it is popped after a durable
    failed mapping is written. The two retry categories alert on different
    deadlines: an occupied GPU is expected for seconds after a run ends,
    while a broken nvidia-smi or tmux needs attention much sooner.
    """

    STARTED = "started"
    RETRY_RESOURCE = "retry_resource"
    RETRY_INFRASTRUCTURE = "retry_infrastructure"
    REJECT_ITEM = "reject_item"


def _signal_process_group(
    proc: asyncio.subprocess.Process,
    sig: signal.Signals,
) -> None:
    """Signal a supervised subprocess group, falling back to the direct child."""
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        action = proc.terminate if sig == signal.SIGTERM else proc.kill
        action()


async def _terminate_and_reap_subprocess(
    proc: asyncio.subprocess.Process,
    terminate_timeout: float = 5.0,
) -> None:
    """Terminate a process group, escalate to SIGKILL, and consume its pipes."""
    if proc.returncode is not None:
        await proc.communicate()
        return

    _signal_process_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.communicate(), timeout=terminate_timeout)
        return
    except asyncio.TimeoutError:
        _signal_process_group(proc, signal.SIGKILL)

    await proc.communicate()


async def _run_supervised_subprocess(
    cmd: List[str],
    timeout: float,
) -> tuple[bytes, bytes, int]:
    """Run an isolated child and guarantee cleanup if it times out or is cancelled."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        await _terminate_and_reap_subprocess(proc)
        raise
    return stdout, stderr, proc.returncode


class _MappingsSegmentWriter:
    """Appends mapping lines to segmented JSONL files.

    Each segment holds up to MAPPINGS_SEGMENT_LINES lines; once full we roll
    forward to the next index. Sealed segments are never touched again, which
    keeps xet happy (hub bulk sync doesn't have to re-upload a growing tail).
    """

    def __init__(self):
        self._idx = 0
        self._lines = 0
        self._initialized = False

    def _initialize(self):
        MAPPINGS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        max_idx = -1
        for entry in MAPPINGS_LOG_DIR.iterdir():
            m = SEGMENT_FILE_PATTERN.match(entry.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        if max_idx < 0:
            self._idx = 0
            self._lines = 0
        else:
            self._idx = max_idx
            top = MAPPINGS_LOG_DIR / f"mappings-{self._idx:06d}.jsonl"
            with open(top, "rb") as f:
                self._lines = sum(1 for _ in f)
            if self._lines >= MAPPINGS_SEGMENT_LINES:
                self._idx += 1
                self._lines = 0
        self._initialized = True

    def append(self, line: str):
        if not self._initialized:
            self._initialize()
        path = MAPPINGS_LOG_DIR / f"mappings-{self._idx:06d}.jsonl"
        with open(path, "a") as f:
            f.write(line if line.endswith("\n") else line + "\n")
        self._lines += 1
        if self._lines >= MAPPINGS_SEGMENT_LINES:
            self._idx += 1
            self._lines = 0


_mappings_writer = _MappingsSegmentWriter()


class _QueueSegmentWriter:
    """Appends queue snapshot lines to segmented JSONL files.

    Each segment holds up to QUEUE_SEGMENT_LINES lines; once full we roll
    forward to the next index. Sealed segments are never touched again, which
    keeps xet happy (hub bulk sync doesn't have to re-upload a growing tail).
    """

    def __init__(self):
        self._idx = 0
        self._lines = 0
        self._initialized = False

    def _initialize(self):
        QUEUE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        max_idx = -1
        for entry in QUEUE_LOG_DIR.iterdir():
            m = QUEUE_SEGMENT_FILE_PATTERN.match(entry.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        if max_idx < 0:
            self._idx = 0
            self._lines = 0
        else:
            self._idx = max_idx
            top = QUEUE_LOG_DIR / f"queue-{self._idx:06d}.jsonl"
            with open(top, "rb") as f:
                self._lines = sum(1 for _ in f)
            if self._lines >= QUEUE_SEGMENT_LINES:
                self._idx += 1
                self._lines = 0
        self._initialized = True

    def append(self, line: str):
        if not self._initialized:
            self._initialize()
        path = QUEUE_LOG_DIR / f"queue-{self._idx:06d}.jsonl"
        with open(path, "a") as f:
            f.write(line if line.endswith("\n") else line + "\n")
        self._lines += 1
        if self._lines >= QUEUE_SEGMENT_LINES:
            self._idx += 1
            self._lines = 0


_queue_writer = _QueueSegmentWriter()


def configure_runtime(
    *,
    repo_dir: Optional[Path] = None,
    state_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    rpc_host: Optional[str] = None,
    rpc_port: Optional[int] = None,
    endpoint_file: Optional[Path] = None,
    device_lock_file: Optional[Path] = None,
    tmux_session: Optional[str] = None,
) -> None:
    """Configure one execution-daemon process before constructing the daemon."""
    global REPO_DIR, DAEMON_DIR, MAPPINGS_DIR, OUTPUT_DIR, IMPORTS_DIR
    global QUEUE_FILE, STATE_FILE, PID_FILE, PENDING_WEIGHTS_FILE
    global LOGS_DIR, ARTIFACTS_DIR, MAPPINGS_LOG_DIR, QUEUE_LOG_DIR
    global RPC_LISTEN_HOST, RPC_LISTEN_PORT, ENDPOINT_FILE, DEVICE_LOCK_FILE
    global TMUX_SESSION
    global _mappings_writer, _queue_writer

    if repo_dir is not None:
        REPO_DIR = repo_dir.resolve()
    DAEMON_DIR = (state_dir.resolve() if state_dir is not None else REPO_DIR / ".daemon")
    MAPPINGS_DIR = DAEMON_DIR / "mappings"
    OUTPUT_DIR = DAEMON_DIR / "output"
    IMPORTS_DIR = DAEMON_DIR / "imports"
    QUEUE_FILE = DAEMON_DIR / "queue.txt"
    STATE_FILE = DAEMON_DIR / "state.json"
    PID_FILE = DAEMON_DIR / "daemon.pid"
    PENDING_WEIGHTS_FILE = DAEMON_DIR / "pending_weights.json"
    LOGS_DIR = REPO_DIR / "logs"
    ARTIFACTS_DIR = (
        artifacts_dir.resolve() if artifacts_dir is not None else LOGS_DIR
    )
    MAPPINGS_LOG_DIR = ARTIFACTS_DIR / "mappings"
    QUEUE_LOG_DIR = ARTIFACTS_DIR / "queue"
    if rpc_host is not None:
        RPC_LISTEN_HOST = rpc_host
    if rpc_port is not None:
        RPC_LISTEN_PORT = rpc_port
    ENDPOINT_FILE = endpoint_file.resolve() if endpoint_file is not None else None
    DEVICE_LOCK_FILE = (
        device_lock_file.resolve() if device_lock_file is not None else None
    )
    if tmux_session:
        TMUX_SESSION = tmux_session
    _mappings_writer = _MappingsSegmentWriter()
    _queue_writer = _QueueSegmentWriter()


@dataclass
class DaemonState:
    status: str  # "idle", "running", "paused"
    current_experiment_id: Optional[int] = None
    current_window: Optional[str] = None
    current_run_id: Optional[str] = None
    session_name: Optional[str] = None
    last_updated: Optional[str] = None

    def save(self):
        self.last_updated = datetime.now(timezone.utc).isoformat()
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "DaemonState":
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except (json.JSONDecodeError, TypeError):
                pass
        return cls(status="idle")


@dataclass
class ExperimentMapping:
    experiment_id: int
    run_id: Optional[str]
    script: str
    code_hash: str
    env_vars: Dict[str, str]
    gpus: int
    gpu_type: str
    tmux_window: str
    log_file: Optional[str]
    started_at: str
    finished_at: Optional[str] = None
    status: str = "running"
    track: Optional[str] = None
    name: Optional[str] = None
    git_commit: Optional[str] = None
    parent_hash: Optional[str] = None
    kernels_path: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)
    failure_phase: Optional[str] = None

    def save(self):
        (MAPPINGS_DIR / f"{self.experiment_id}.json").write_text(json.dumps(asdict(self), indent=2))
        # Append to segmented JSONL for hub sync (sealed segments stay xet-friendly)
        _mappings_writer.append(json.dumps(asdict(self)))

    @classmethod
    def load(cls, experiment_id: int) -> Optional["ExperimentMapping"]:
        path = MAPPINGS_DIR / f"{experiment_id}.json"
        if path.exists():
            try:
                return cls(**json.loads(path.read_text()))
            except (json.JSONDecodeError, TypeError):
                pass
        return None


@dataclass
class QueuedItem:
    experiment_id: int
    script: str
    env_vars: Dict[str, str]
    gpus: int
    gpu_type: str = "H100"
    name: Optional[str] = None
    track: Optional[str] = None
    cmd_prefix: Optional[str] = None


@dataclass
class PreparedLaunch:
    """Deterministically validated launch inputs, ready for tmux dispatch."""

    experiment_id: int
    script: str
    env_vars: Dict[str, str]
    gpus: int
    gpu_type: str
    name: Optional[str]
    track: Optional[str]
    cmd_prefix: Optional[str]
    code_hash: str
    parent_path: Optional[str]
    kernels_path: Optional[str]
    dependencies: Dict[str, str]
    symlink_cmd: Optional[str]
    dependency_overlay: Optional[str]


class NanorunDaemon:
    def __init__(
        self,
        session_name: str = "default",
        hub_session: Optional[str] = None,
    ):
        self.session_name = session_name
        self.hub_session = hub_session or session_name
        self.state = DaemonState.load()
        self.state.session_name = session_name
        self.running = True
        self._pid_file_handle = None
        self._device_lock_handle = None
        self._pending_events: List[EventMessage] = []
        self._ws_clients: Set = set()
        self._uploaded_weights: Set[str] = set()
        self._pending_weight_uploads: Dict[str, Dict[str, Any]] = {}

        self._startup_time = time.monotonic()
        # Set while the queue head cannot start for an environmental reason
        # (RETRY_* dispositions); cleared on a successful start or item pop.
        self._start_block: Optional[Dict[str, Any]] = None
        # Event-driven queue push: _emit_queue_changed sets this, _queue_push_task
        # drains it (debounced) and uploads just the queue segment to the hub.
        self._queue_dirty = asyncio.Event()
        self._queue_push_failures = 0
        for d in [
            DAEMON_DIR,
            MAPPINGS_DIR,
            OUTPUT_DIR,
            IMPORTS_DIR,
            LOGS_DIR,
            ARTIFACTS_DIR,
            MAPPINGS_LOG_DIR,
            QUEUE_LOG_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        self._load_pending_weight_uploads()
        # Recover from stale "running" state (daemon was killed mid-experiment)
        if self.state.status == "running":
            window = self.state.current_window
            if not window or not self._tmux_window_exists(window):
                exp_id = self.state.current_experiment_id
                if exp_id:
                    mapping = ExperimentMapping.load(exp_id)
                    if mapping and mapping.status == "running":
                        mapping.status = "failed"
                        mapping.finished_at = datetime.now(timezone.utc).isoformat()
                        mapping.save()
                    print(f"[daemon] Recovered from stale state: experiment {exp_id} marked failed")
                self.state.status = "idle"
                self.state.current_experiment_id = None
                self.state.current_window = None
                self.state.current_run_id = None
                self.state.save()
        # Publish current queue snapshot once at startup (so a freshly started
        # daemon's queue lands on the hub even before the first change).
        self._emit_queue_changed()

    @property
    def _hub_namespace(self) -> str:
        """Hub key namespace, with compatibility for test/legacy instances."""
        return getattr(self, "hub_session", self.session_name)

    # --- events ---

    def _emit(self, event: Event, **data):
        self._pending_events.append(EventMessage(event=event, data=data))

    def _emit_queue_changed(self):
        ts = datetime.now(timezone.utc).isoformat()
        queue = [asdict(item) for item in self.read_queue()]
        _queue_writer.append(json.dumps({"ts": ts, "queue": queue}))
        self._emit(Event.QUEUE_CHANGED, queue=queue, ts=ts)
        # Wake the event-driven hub push so the snapshot reaches the hub in ~1s
        # instead of waiting for the 15s bulk sync. All callers run on the event
        # loop (or pre-loop in __init__), so a direct .set() is safe.
        self._queue_dirty.set()

    async def _flush_events(self):
        if not self._pending_events or not self._ws_clients:
            self._pending_events.clear()
            return
        events = self._pending_events[:]
        self._pending_events.clear()
        for event_msg in events:
            raw = event_msg.to_json()
            dead = set()
            for ws in self._ws_clients:
                try:
                    await ws.send(raw)
                except websockets.ConnectionClosed:
                    dead.add(ws)
            self._ws_clients -= dead

    # --- PID lock ---

    def acquire_pid_lock(self) -> bool:
        if DEVICE_LOCK_FILE is not None:
            try:
                DEVICE_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
                self._device_lock_handle = open(DEVICE_LOCK_FILE, "a+")
                fcntl.flock(
                    self._device_lock_handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                self._device_lock_handle.seek(0)
                self._device_lock_handle.truncate()
                self._device_lock_handle.write(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "session": self.session_name,
                            "state_dir": str(DAEMON_DIR),
                        }
                    )
                )
                self._device_lock_handle.flush()
            except (IOError, OSError):
                if self._device_lock_handle:
                    self._device_lock_handle.close()
                    self._device_lock_handle = None
                print(
                    "Another local nanorun execution session already owns this device",
                    file=sys.stderr,
                )
                return False

        try:
            self._pid_file_handle = open(PID_FILE, "w")
            fcntl.flock(self._pid_file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._pid_file_handle.write(str(os.getpid()))
            self._pid_file_handle.flush()
            return True
        except (IOError, OSError):
            if self._pid_file_handle:
                self._pid_file_handle.close()
                self._pid_file_handle = None
            self._release_device_lock()
            try:
                print(f"Another daemon is already running (PID {PID_FILE.read_text().strip()})", file=sys.stderr)
            except Exception:
                print("Another daemon is already running", file=sys.stderr)
            return False

    def _release_device_lock(self):
        if not self._device_lock_handle:
            return
        try:
            fcntl.flock(self._device_lock_handle.fileno(), fcntl.LOCK_UN)
            self._device_lock_handle.close()
        except Exception:
            pass
        self._device_lock_handle = None

    def release_pid_lock(self):
        if self._pid_file_handle:
            try:
                fcntl.flock(self._pid_file_handle.fileno(), fcntl.LOCK_UN)
                self._pid_file_handle.close()
            except Exception:
                pass
            self._pid_file_handle = None
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass
        self._release_device_lock()

    # --- GPU ---

    def _query_gpu_processes(self) -> tuple[bool, List[Dict[str, Any]], str]:
        """Query compute processes, distinguishing an empty GPU from a failed probe."""
        cmd = "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits"
        success, stdout, stderr = self._run_cmd(cmd)
        if not success:
            return False, [], stderr.strip() or "nvidia-smi failed"
        processes = []
        if stdout.strip():
            for line in stdout.strip().split("\n"):
                parts = line.strip().split(", ")
                if len(parts) >= 3:
                    try:
                        processes.append({"pid": int(parts[0]), "name": parts[1], "memory_mb": int(parts[2])})
                    except ValueError:
                        pass
        return True, processes, ""

    def get_gpu_processes(self) -> List[Dict[str, Any]]:
        _, processes, _ = self._query_gpu_processes()
        return processes

    # --- queue ---

    def read_queue(self) -> List[QueuedItem]:
        if not QUEUE_FILE.exists():
            return []
        items = []
        for line in QUEUE_FILE.read_text().strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(QueuedItem(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Invalid queue line: {line}", file=sys.stderr)
        return items

    def write_queue(self, items: List[QueuedItem]):
        QUEUE_FILE.write_text("\n".join(json.dumps(asdict(item)) for item in items) + "\n" if items else "")

    def add_to_queue(self, item: QueuedItem, first: bool = False) -> int:
        items = self.read_queue()
        if first:
            items.insert(0, item)
        else:
            items.append(item)
        self.write_queue(items)
        return len(items)

    def remove_from_queue(self, index: int) -> bool:
        items = self.read_queue()
        if 0 <= index < len(items):
            items.pop(index)
            self.write_queue(items)
            return True
        return False

    def clear_queue(self):
        self.write_queue([])

    # --- code hash ---

    def get_git_commit(self) -> Optional[str]:
        try:
            r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True, timeout=5)
            return r.stdout.strip()[:GIT_HASH_LENGTH] if r.returncode == 0 else None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _parse_frontmatter(self, script_path: str) -> ScriptManifest:
        full_path = REPO_DIR / script_path
        if not full_path.exists():
            return ScriptManifest()
        return parse_script_manifest(full_path.read_text())

    def _compute_file_hash(self, rel_path: str) -> Optional[str]:
        full_path = REPO_DIR / rel_path
        if not full_path.exists():
            return None
        return hashlib.sha256(full_path.read_bytes()).hexdigest()[:CODE_HASH_LENGTH]

    def _compute_code_hash(
        self,
        script: str,
        kernels_path: Optional[str] = None,
        dependencies: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        return compute_script_hash(
            REPO_DIR,
            script,
            kernels_path=kernels_path,
            dependencies=dependencies,
        )

    def _kernels_symlink_cmd(self, script_path: str, kernels_path: str) -> str:
        script_dir = Path(script_path).parent
        kernels_dir = Path(kernels_path).parent
        target = (
            Path(kernels_path).name
            if kernels_dir == script_dir
            else REPO_DIR / kernels_path
        )
        destination = script_dir / "triton_kernels.py"
        return f"ln -sf {shlex.quote(str(target))} {shlex.quote(str(destination))}"

    def _prepare_dependency_overlay(
        self,
        script_path: str,
        experiment_id: int,
        dependencies: Dict[str, str],
    ) -> Optional[str]:
        """Stage dependency aliases outside the repository worktree."""

        if not dependencies:
            return None

        script_dir = (REPO_DIR / script_path).parent
        overlay = IMPORTS_DIR / str(experiment_id)
        staged = False
        for module, dependency_path in dependencies.items():
            source = (REPO_DIR / dependency_path).resolve()
            sibling = script_dir / f"{module}.py"
            if sibling.exists() or sibling.is_symlink():
                if sibling.resolve() == source:
                    continue
                # Older kernels runs leave this daemon-managed compatibility
                # symlink in the script directory. Remove it when the script
                # migrates to a generic triton_kernels dependency so the
                # dependency overlay can take precedence.
                if module == "triton_kernels" and sibling.is_symlink():
                    sibling.unlink()
                else:
                    raise ManifestError(
                        f"Dependency module {module} conflicts with existing file: "
                        f"{sibling.relative_to(REPO_DIR)}"
                    )

            overlay.mkdir(parents=True, exist_ok=True)
            destination = overlay / f"{module}.py"
            if destination.exists() and not destination.is_symlink():
                raise ManifestError(
                    f"Dependency overlay path is not a symlink: {destination}"
                )
            if destination.is_symlink():
                if destination.resolve() == source:
                    staged = True
                    continue
                destination.unlink()
            destination.symlink_to(source)
            staged = True
        return str(overlay) if staged else None

    # --- tmux ---

    def _run_cmd(self, cmd: str, timeout: int = 30) -> tuple[bool, str, str]:
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def _tmux_window_exists(self, window: str) -> bool:
        ok, stdout, _ = self._run_cmd(f"tmux list-windows -t {TMUX_SESSION} -F '#W'")
        return ok and window in stdout.strip().split("\n")

    def _tmux_create(self, window: str, command: str) -> bool:
        self._run_cmd(f"tmux has-session -t {TMUX_SESSION} 2>/dev/null || tmux new-session -d -s {TMUX_SESSION}")
        ok, _, err = self._run_cmd(
            f"tmux new-window -t {TMUX_SESSION} -n {shlex.quote(window)} "
            f"-c {shlex.quote(str(REPO_DIR))}"
        )
        if not ok:
            print(f"Failed to create tmux window: {err}", file=sys.stderr)
            return False
        self._run_cmd(f"tmux set-option -t {TMUX_SESSION}:{window} remain-on-exit on")
        escaped = command.replace("'", "'\\''")
        ok, _, err = self._run_cmd(f"tmux respawn-pane -t {TMUX_SESSION}:{window} -k '{escaped}'")
        if not ok:
            print(f"Failed to respawn pane: {err}", file=sys.stderr)
        return ok

    def _tmux_kill(self, window: str) -> bool:
        ok, _, _ = self._run_cmd(f"tmux kill-window -t {TMUX_SESSION}:{window}")
        return ok

    def _tmux_pane_dead(self, window: str) -> bool:
        ok, stdout, _ = self._run_cmd(f"tmux list-panes -t {TMUX_SESSION}:{window} -F '#{{pane_dead}}'")
        return ok and stdout.strip() == "1"

    # --- experiment lifecycle ---

    def _resolve_script_path(self, script: str) -> Optional[str]:
        """Resolve a repository-relative Python entrypoint.

        The local CLI performs the same boundary check, but the daemon repeats
        it because queue RPC state is an external input.
        """
        path = Path(script)
        if path.is_absolute() or ".." in path.parts:
            return None

        full = REPO_DIR / path
        if full.exists():
            resolved = full.resolve()
            try:
                relative = resolved.relative_to(REPO_DIR.resolve())
            except ValueError:
                return None
            return relative.as_posix() if resolved.is_file() and resolved.suffix == ".py" else None

        # Case-insensitive search: walk each path component
        current = REPO_DIR
        for part in path.parts:
            match = None
            try:
                for entry in current.iterdir():
                    if entry.name.lower() == part.lower():
                        match = entry
                        break
            except OSError:
                return script
            if not match:
                return None
            current = match
        resolved = current.resolve()
        try:
            relative = resolved.relative_to(REPO_DIR.resolve())
        except ValueError:
            return None
        return relative.as_posix() if resolved.is_file() and resolved.suffix == ".py" else None

    def _build_run_command(self, script: str, env_vars: Dict[str, str], gpus: int,
                           exp_id: int, cmd_prefix: Optional[str] = None,
                           symlink_cmd: Optional[str] = None,
                           dependency_overlay: Optional[str] = None,
                           gpu_type: str = "H100") -> str:
        env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
        if env_str:
            env_str += " "
        if gpu_type == "MPS":
            run_cmd = f"python {script}"
        else:
            run_cmd = f"torchrun --standalone --nproc_per_node={gpus} {script}"
        if cmd_prefix:
            run_cmd = f"{cmd_prefix} {run_cmd}"
        output_file = OUTPUT_DIR / f"{exp_id}.txt"
        parts = ["source .venv/bin/activate"]
        if symlink_cmd:
            parts.append(symlink_cmd)
        if dependency_overlay:
            parts.append(
                f"export PYTHONPATH={shlex.quote(dependency_overlay)}"
                "${PYTHONPATH:+:$PYTHONPATH}"
            )
        parts.append(f"{env_str}{run_cmd} 2>&1 | tee {output_file}")
        return " && ".join(parts)

    def _start_failure(self, disposition: StartDisposition, error: str, **extra) -> Dict[str, Any]:
        return {"success": False, "disposition": disposition.value, "error": error, **extra}

    def _prepare_experiment(self, experiment_id: int, script: str, env_vars: Dict[str, str],
                            gpus: int, gpu_type: str, name: Optional[str],
                            track: Optional[str], cmd_prefix: Optional[str],
                            ) -> tuple[Optional[PreparedLaunch], Optional[Dict[str, Any]]]:
        """Deterministic validation and staging; touches no GPU or tmux state.

        Runs before the GPU probe so a broken item is rejected even while the
        GPU is occupied — otherwise it would look retryable indefinitely.
        Returns (prepared, None) on success or (None, failure result).
        """
        resolved = self._resolve_script_path(script)
        if not resolved:
            return None, self._start_failure(
                StartDisposition.REJECT_ITEM,
                "Script must be a repository-relative .py file",
            )
        script = resolved

        try:
            manifest = self._parse_frontmatter(script)
        except (ManifestError, UnicodeError) as error:
            return None, self._start_failure(
                StartDisposition.REJECT_ITEM,
                f"Invalid script frontmatter: {error}",
            )
        except OSError as error:
            # The script resolved a moment ago, so an unreadable file points
            # at the filesystem, not the item.
            return None, self._start_failure(
                StartDisposition.RETRY_INFRASTRUCTURE,
                f"Cannot read script: {error}",
            )

        parent_path = manifest.parent
        kernels_path = manifest.kernels
        dependencies: Dict[str, str] = {}
        if parent_path:
            resolved_parent = self._resolve_script_path(parent_path)
            if not resolved_parent:
                return None, self._start_failure(
                    StartDisposition.REJECT_ITEM,
                    "Frontmatter parent must be an existing "
                    f"repository-relative .py file: {parent_path}",
                )
            parent_path = resolved_parent
        if kernels_path:
            resolved_kernels = self._resolve_script_path(kernels_path)
            if not resolved_kernels:
                return None, self._start_failure(
                    StartDisposition.REJECT_ITEM,
                    "Frontmatter kernels must be an existing "
                    f"repository-relative .py file: {kernels_path}",
                )
            kernels_path = resolved_kernels
        for module, dependency_path in manifest.dependencies:
            resolved_dependency = self._resolve_script_path(dependency_path)
            if not resolved_dependency:
                return None, self._start_failure(
                    StartDisposition.REJECT_ITEM,
                    f"Frontmatter dependency {module} must be an existing "
                    f"repository-relative .py file: {dependency_path}",
                )
            dependencies[module] = resolved_dependency

        symlink_cmd = self._kernels_symlink_cmd(script, kernels_path) if kernels_path else None
        try:
            dependency_overlay = self._prepare_dependency_overlay(
                script,
                experiment_id,
                dependencies,
            )
        except ManifestError as error:
            return None, self._start_failure(StartDisposition.REJECT_ITEM, str(error))
        except OSError as error:
            return None, self._start_failure(
                StartDisposition.RETRY_INFRASTRUCTURE,
                f"Dependency overlay failed: {error}",
            )

        code_hash = self._compute_code_hash(script, kernels_path, dependencies)
        if not code_hash:
            return None, self._start_failure(
                StartDisposition.REJECT_ITEM,
                f"Script or declared dependency not found: {script}",
            )

        return PreparedLaunch(
            experiment_id=experiment_id, script=script, env_vars=env_vars,
            gpus=gpus, gpu_type=gpu_type, name=name, track=track,
            cmd_prefix=cmd_prefix, code_hash=code_hash, parent_path=parent_path,
            kernels_path=kernels_path, dependencies=dependencies,
            symlink_cmd=symlink_cmd, dependency_overlay=dependency_overlay,
        ), None

    def _probe_gpu(self, gpu_type: str) -> Optional[Dict[str, Any]]:
        """Return a failure result if the GPU cannot accept a launch, else None."""
        if gpu_type == "MPS":
            return None  # no nvidia-smi on MPS devices
        ok, gpu_procs, probe_error = self._query_gpu_processes()
        if not ok:
            return self._start_failure(
                StartDisposition.RETRY_INFRASTRUCTURE,
                f"GPU probe failed: {probe_error}",
            )
        if gpu_procs:
            pids = [p["pid"] for p in gpu_procs]
            total_mem = sum(p["memory_mb"] for p in gpu_procs)
            return self._start_failure(
                StartDisposition.RETRY_RESOURCE,
                f"GPU busy: PIDs {pids} ({total_mem}MB)",
                gpu_processes=gpu_procs,
            )
        return None

    def _launch_experiment(self, prepared: PreparedLaunch) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).strftime("%m%d_%H%M%S")
        base_name = prepared.name or Path(prepared.script).stem
        window_name = f"{timestamp}_{base_name}"[:TMUX_WINDOW_NAME_MAX]

        cmd = self._build_run_command(
            prepared.script,
            prepared.env_vars,
            prepared.gpus,
            prepared.experiment_id,
            prepared.cmd_prefix,
            prepared.symlink_cmd,
            prepared.dependency_overlay,
            prepared.gpu_type,
        )
        if not self._tmux_create(window_name, cmd):
            return self._start_failure(
                StartDisposition.RETRY_INFRASTRUCTURE,
                "Failed to create tmux window",
            )

        mapping = ExperimentMapping(
            experiment_id=prepared.experiment_id, run_id=None, script=prepared.script,
            code_hash=prepared.code_hash, env_vars=prepared.env_vars,
            gpus=prepared.gpus, gpu_type=prepared.gpu_type,
            tmux_window=window_name, log_file=None,
            started_at=datetime.now(timezone.utc).isoformat(),
            track=prepared.track, name=prepared.name, git_commit=self.get_git_commit(),
            parent_hash=self._compute_file_hash(prepared.parent_path) if prepared.parent_path else None,
            kernels_path=prepared.kernels_path,
            dependencies=prepared.dependencies,
        )
        mapping.save()

        self.state.status = "running"
        self.state.current_experiment_id = prepared.experiment_id
        self.state.current_window = window_name
        self.state.current_run_id = None
        self.state.save()
        self._uploaded_weights.clear()

        self._emit(Event.EXPERIMENT_STARTED, **asdict(mapping))
        return {"success": True, "disposition": StartDisposition.STARTED.value,
                "code_hash": prepared.code_hash, "window_name": window_name}

    def start_experiment(self, experiment_id: int, script: str, env_vars: Dict[str, str],
                         gpus: int = 1, gpu_type: str = "H100", name: Optional[str] = None,
                         track: Optional[str] = None, cmd_prefix: Optional[str] = None) -> Dict[str, Any]:
        if self.state.status == "running":
            return self._start_failure(
                StartDisposition.RETRY_RESOURCE,
                f"Already running experiment {self.state.current_experiment_id}",
            )
        prepared, failure = self._prepare_experiment(
            experiment_id, script, env_vars, gpus, gpu_type, name, track, cmd_prefix,
        )
        if failure:
            return failure
        gpu_failure = self._probe_gpu(gpu_type)
        if gpu_failure:
            return gpu_failure
        return self._launch_experiment(prepared)

    def check_current_experiment(self):
        """Poll experiment status: detect run_id, handle completion."""
        if self.state.status != "running" or not self.state.current_window:
            return
        window = self.state.current_window

        if not self._tmux_window_exists(window) or self._tmux_pane_dead(window):
            # A short-lived script may exit before the normal polling path sees
            # its (possibly buffered) run-id print. The tee'd output is complete
            # once the pane is dead, so make one final detection attempt before
            # classifying the experiment.
            if not self.state.current_run_id:
                run_id = self._detect_run_id()
                if run_id:
                    self._record_run_id(run_id)
            if self._tmux_window_exists(window):
                self._tmux_kill(window)
            self._handle_experiment_finished()
            return

        # Still running — try to detect run_id from log file existence
        if not self.state.current_run_id:
            run_id = self._detect_run_id()
            if run_id:
                self._record_run_id(run_id)

        # A final metric is both the success signal and the workload boundary.
        # Terminate the tmux window before finalizing so no GPU work can continue
        # after the run is recorded as complete.
        if self.state.current_run_id:
            log_path = LOGS_DIR / f"{self.state.current_run_id}.txt"
            last_metric = self._find_last_metric_step(log_path)
            if last_metric and last_metric[0] >= last_metric[1]:
                print(
                    f"[daemon] Final metric detected for experiment "
                    f"{self.state.current_experiment_id}: "
                    f"step:{last_metric[0]}/{last_metric[1]}"
                )
                if self._tmux_kill(window):
                    self._handle_experiment_finished()
                else:
                    print(
                        f"[daemon] Failed to terminate completed experiment "
                        f"{self.state.current_experiment_id}; will retry",
                        file=sys.stderr,
                    )

    def _record_run_id(self, run_id: str):
        """Persist and emit a run ID detected from experiment stdout."""
        self._materialize_log(run_id)
        self.state.current_run_id = run_id
        self.state.save()
        mapping = ExperimentMapping.load(self.state.current_experiment_id)
        if mapping:
            mapping.run_id = run_id
            mapping.log_file = f"logs/{run_id}.txt"
            mapping.save()
        self._emit(
            Event.EXPERIMENT_RUN_ID,
            experiment_id=self.state.current_experiment_id,
            run_id=run_id,
        )

    def _materialize_log(self, run_id: str) -> Optional[Path]:
        """Expose a locally-produced run log in the session artifact directory."""
        source = LOGS_DIR / f"{run_id}.txt"
        destination = ARTIFACTS_DIR / f"{run_id}.txt"
        if source == destination or not source.is_file():
            return source if source.is_file() else None
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                try:
                    destination.write_bytes(source.read_bytes())
                except OSError:
                    return None
        elif destination.stat().st_ino != source.stat().st_ino:
            # Cross-filesystem fallback: append only bytes not yet materialized.
            destination_size = destination.stat().st_size
            source_size = source.stat().st_size
            if source_size > destination_size:
                with open(source, "rb") as src, open(destination, "ab") as dst:
                    src.seek(destination_size)
                    dst.write(src.read())
        return destination

    def _detect_run_id(self) -> Optional[str]:
        """Check tee'd output file for run_id pattern.

        Searches both head and tail of the file — the run_id is typically
        printed once at script startup and can be pushed out of the tail
        window by verbose per-step output.
        """
        output_file = OUTPUT_DIR / f"{self.state.current_experiment_id}.txt"
        if not output_file.exists():
            return None
        try:
            with open(output_file, "rb") as f:
                head = f.read(8192).decode("utf-8", errors="ignore")
                match = RUN_ID_PATTERN.search(head)
                if match:
                    return match.group(1)
                f.seek(max(0, f.seek(0, 2) - 8192))
                tail = f.read().decode("utf-8", errors="ignore")
            match = RUN_ID_PATTERN.search(tail)
            return match.group(1) if match else None
        except Exception:
            return None

    def _find_last_step(self, path: Path) -> tuple[int, int] | None:
        """Find the last step:N/M line in a file. Returns (step, total) or None."""
        if not path.exists():
            return None
        try:
            ok, stdout, _ = self._run_cmd(f"grep -oP 'step:\\K\\d+/\\d+' '{path}' | tail -1", timeout=10)
            if ok and stdout.strip():
                parts = stdout.strip().split("/")
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return None

    def _find_last_metric_step(self, path: Path) -> tuple[int, int] | None:
        """Find the last loss-bearing metric step near the end of a live log.

        Live termination deliberately requires a real metric line rather than
        any ``step:N/M`` text. Training scripts commonly print their own source
        at startup, and an unfilled f-string must never terminate a workload.
        """
        output = self._read_file_tail(path)
        last = None
        for line in output.splitlines():
            step = METRIC_STEP_PATTERN.search(line)
            if step and LOSS_PATTERN.search(line):
                last = (int(step.group(1)), int(step.group(2)))
        return last

    def _handle_experiment_finished(self):
        exp_id = self.state.current_experiment_id
        run_id = self.state.current_run_id
        if exp_id:
            mapping = ExperimentMapping.load(exp_id)
            if mapping:
                mapping.finished_at = datetime.now(timezone.utc).isoformat()
                # Classify: grep entire log for last step line (tail-only missed
                # completions in verbose logs with large JSON blocks between steps)
                status = "failed"
                if mapping.run_id:
                    log_path = LOGS_DIR / f"{mapping.run_id}.txt"
                    last = self._find_last_step(log_path) or self._find_last_step(OUTPUT_DIR / f"{exp_id}.txt")
                    if last and last[0] >= last[1]:
                        status = "completed"
                mapping.status = status
                mapping.save()
                print(f"[daemon] Experiment {exp_id} {status}")
                if status == "completed":
                    self._queue_final_weight_upload(exp_id, mapping.run_id)
                    self._emit(Event.EXPERIMENT_FINISHED, experiment_id=exp_id,
                               status="completed", run_id=run_id, code_hash=mapping.code_hash)
                else:
                    crash_log = self._read_file_tail(OUTPUT_DIR / f"{exp_id}.txt")
                    self._emit(Event.EXPERIMENT_FAILED, experiment_id=exp_id,
                               status=status, run_id=run_id, crash_log=crash_log)
        self.state.status = "idle"
        self.state.current_experiment_id = None
        self.state.current_window = None
        self.state.current_run_id = None
        self._uploaded_weights.clear()
        self.state.save()
        # Deliberately no _process_queue() here: starting the next item
        # microseconds after the previous run's window died races GPU memory
        # release. The monitor loop starts it on a later tick.

    def _read_file_tail(self, path: Path, max_bytes: int = LOG_TAIL_BYTES) -> str:
        if not path.exists():
            return ""
        try:
            with open(path, "rb") as f:
                f.seek(max(0, f.seek(0, 2) - max_bytes))
                return f.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _process_queue(self):
        """Attempt to start the queue head once.

        The monitor loop calls this on every idle tick, so retryable failures
        need no loop or sleep here: the head stays queued and the next tick
        retries. Only a REJECT_ITEM pops without starting — after a durable
        failed mapping is written, because a disconnected observer that misses
        the event must not see the item silently vanish from the queue.
        """
        if self.state.status in ("paused", "running"):
            return
        items = self.read_queue()
        if not items:
            return
        item = items[0]
        result = self.start_experiment(
            experiment_id=item.experiment_id, script=item.script,
            env_vars=item.env_vars, gpus=item.gpus, gpu_type=item.gpu_type,
            name=item.name, track=item.track, cmd_prefix=item.cmd_prefix,
        )
        if result["success"]:
            print(f"Started queued experiment {item.experiment_id}: {item.script}")
            self._clear_start_block()
            items.pop(0)
            self.write_queue(items)
            self._emit_queue_changed()
            return
        disposition = result.get("disposition", StartDisposition.RETRY_INFRASTRUCTURE.value)
        if disposition == StartDisposition.REJECT_ITEM.value:
            self._clear_start_block()
            self._record_launch_failure(item, result.get("error", "Launch failed"))
            items.pop(0)
            self.write_queue(items)
            self._emit_queue_changed()
        else:
            self._note_start_block(disposition, result.get("error", ""))

    def _record_launch_failure(self, item: QueuedItem, error: str):
        """Persist a terminal mapping for an item that never launched.

        Mapping first, then the event, then the caller pops the item and
        publishes the queue snapshot — the hub-synced mapping stream is the
        durable authority, the WebSocket event is best-effort.
        """
        print(f"Rejecting queued experiment {item.experiment_id}: {error}", file=sys.stderr)
        now = datetime.now(timezone.utc).isoformat()
        try:
            # Lands in the normal crash-log path (get_crash_log RPC + backfill)
            (OUTPUT_DIR / f"{item.experiment_id}.txt").write_text(f"Launch failed: {error}\n")
        except OSError:
            pass
        mapping = ExperimentMapping(
            experiment_id=item.experiment_id, run_id=None, script=item.script,
            code_hash="", env_vars=item.env_vars, gpus=item.gpus,
            gpu_type=item.gpu_type, tmux_window="", log_file=None,
            started_at=now, finished_at=now, status="failed",
            track=item.track, name=item.name, git_commit=self.get_git_commit(),
            failure_phase="launch",
        )
        mapping.save()
        self._emit(Event.EXPERIMENT_FAILED, experiment_id=item.experiment_id,
                   status="failed", run_id=None, failure_phase="launch",
                   crash_log=f"Launch failed: {error}")

    def _note_start_block(self, disposition: str, error: str):
        now = time.monotonic()
        block = self._start_block
        if block is None or block["disposition"] != disposition:
            self._start_block = {"disposition": disposition, "error": error,
                                 "since": now, "alerted": False}
            print(f"[daemon] Queue head blocked ({disposition}): {error}", file=sys.stderr)
            return
        block["error"] = error
        threshold = (START_BLOCK_ALERT_RESOURCE_S
                     if disposition == StartDisposition.RETRY_RESOURCE.value
                     else START_BLOCK_ALERT_INFRASTRUCTURE_S)
        if not block["alerted"] and now - block["since"] >= threshold:
            block["alerted"] = True
            print(
                f"[daemon] ALERT: queue blocked for {(now - block['since']) / 60:.0f}m "
                f"({disposition}): {error}",
                file=sys.stderr,
            )

    def _clear_start_block(self):
        self._start_block = None

    def cancel_experiment(self) -> Dict[str, Any]:
        if self.state.status != "running":
            return {"success": False, "error": "No experiment running"}
        exp_id = self.state.current_experiment_id
        if self.state.current_window:
            self._tmux_kill(self.state.current_window)
        if exp_id:
            mapping = ExperimentMapping.load(exp_id)
            if mapping:
                mapping.finished_at = datetime.now(timezone.utc).isoformat()
                mapping.status = "cancelled"
                mapping.save()
        self.state.status = "idle"
        self.state.current_experiment_id = None
        self.state.current_window = None
        self.state.current_run_id = None
        self._uploaded_weights.clear()
        self.state.save()
        self._emit(Event.EXPERIMENT_FINISHED, experiment_id=exp_id, status="cancelled")
        return {"success": True, "cancelled_experiment_id": exp_id}

    def pause(self) -> Dict[str, Any]:
        result = {"was_running": self.state.status == "running"}
        if self.state.status == "running":
            self.cancel_experiment()
        self.state.status = "paused"
        self.state.save()
        return {"success": True, **result}

    def resume(self) -> Dict[str, Any]:
        if self.state.status == "running":
            return {"success": True, "message": "Already running"}
        self.state.status = "idle"
        self.state.save()
        self._process_queue()
        return {"success": True, "status": self.state.status}

    # --- RPC dispatch ---

    def handle_rpc_request(self, request: Request) -> Response:
        method, params, rid = request.method, request.params, request.id
        try:
            handler = self._rpc_handlers.get(method)
            if handler:
                return handler(self, params, rid)
            return Response.err(rid, ErrorCode.INVALID_METHOD, f"Unknown method: {method}")
        except KeyError as e:
            return Response.err(rid, ErrorCode.INVALID_PARAMS, f"Missing parameter: {e}")
        except Exception as e:
            print(f"[daemon] RPC error handling {method}: {e}", file=sys.stderr)
            return Response.err(rid, ErrorCode.INTERNAL, str(e))

    def _rpc_ping(self, params, rid):
        return Response.ok(rid, pong=True, status=self.state.status)

    def _rpc_run(self, params, rid):
        result = self.start_experiment(
            experiment_id=params["experiment_id"], script=params["script"],
            env_vars=params.get("env_vars", {}), gpus=params.get("gpus") or 1,
            gpu_type=params.get("gpu_type") or "H100", name=params.get("name"), track=params.get("track"),
        )
        return Response.ok(rid, **result) if result["success"] else Response.err(rid, ErrorCode.CONFLICT, result["error"])

    def _rpc_queue_add(self, params, rid):
        item = QueuedItem(
            experiment_id=params["experiment_id"], script=params["script"],
            env_vars=params.get("env_vars", {}), gpus=params.get("gpus") or 1,
            gpu_type=params.get("gpu_type") or "H100", name=params.get("name"),
            track=params.get("track"), cmd_prefix=params.get("cmd_prefix"),
        )
        new_len = self.add_to_queue(item, first=params.get("first", False))
        position = 1 if params.get("first", False) else new_len
        started = False
        if params.get("auto_start", False):
            if self.state.status == "paused":
                self.state.status = "idle"
                self.state.save()
            if self.state.status == "idle":
                self._process_queue()
                started = self.state.status == "running"
        self._emit_queue_changed()
        return Response.ok(rid, success=True, position=position, daemon_status=self.state.status, started=started)

    def _rpc_cancel(self, params, rid):
        result = self.cancel_experiment()
        if result["success"] and not params.get("pause", False):
            self._process_queue()
        return Response.ok(rid, **result) if result["success"] else Response.err(rid, ErrorCode.CONFLICT, result["error"])

    def _rpc_pause(self, params, rid):
        return Response.ok(rid, **self.pause())

    def _rpc_resume(self, params, rid):
        return Response.ok(rid, **self.resume())

    def _rpc_status(self, params, rid):
        self.check_current_experiment()
        queue = self.read_queue()
        start_block = None
        if self._start_block:
            start_block = {
                "disposition": self._start_block["disposition"],
                "error": self._start_block["error"],
                "blocked_s": round(time.monotonic() - self._start_block["since"], 1),
            }
        return Response.ok(rid, status=self.state.status,
            current_experiment_id=self.state.current_experiment_id,
            current_window=self.state.current_window,
            current_run_id=self.state.current_run_id,
            queue_length=len(queue), queue=[asdict(item) for item in queue],
            hub_session=self._hub_namespace,
            gpu_processes=self.get_gpu_processes(),
            start_block=start_block,
            ts=datetime.now(timezone.utc).isoformat())

    def _rpc_gpu_processes(self, params, rid):
        return Response.ok(rid, gpu_processes=self.get_gpu_processes())

    def _rpc_queue_list(self, params, rid):
        return Response.ok(rid, queue=[asdict(item) for item in self.read_queue()])

    def _rpc_queue_clear(self, params, rid):
        self.clear_queue()
        self._emit_queue_changed()
        return Response.ok(rid, success=True)

    def _rpc_queue_remove(self, params, rid):
        success = self.remove_from_queue(params.get("index", -1))
        if success:
            self._emit_queue_changed()
        return Response.ok(rid, success=success)

    def _rpc_queue_set(self, params, rid):
        items = [
            QueuedItem(experiment_id=d.get("experiment_id", 0), script=d["script"],
                       env_vars=d.get("env_vars", {}), gpus=d.get("gpus") or 1,
                       gpu_type=d.get("gpu_type") or "H100", name=d.get("name"),
                       track=d.get("track"), cmd_prefix=d.get("cmd_prefix"))
            for d in params.get("items", [])
        ]
        self.write_queue(items)
        self._emit_queue_changed()
        return Response.ok(rid, success=True, count=len(items))

    def _rpc_get_mapping(self, params, rid):
        mapping = ExperimentMapping.load(params.get("experiment_id"))
        if mapping:
            return Response.ok(rid, success=True, mapping=asdict(mapping))
        return Response.err(rid, ErrorCode.NOT_FOUND, f"Mapping not found for experiment {params.get('experiment_id')}")

    def _rpc_list_mappings(self, params, rid):
        mappings = []
        for f in MAPPINGS_DIR.glob("*.json"):
            try:
                mappings.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, TypeError):
                pass
        return Response.ok(rid, mappings=mappings)

    def _rpc_get_crash_log(self, params, rid):
        exp_id = params.get("experiment_id")
        output = self._read_file_tail(OUTPUT_DIR / f"{exp_id}.txt")
        if output:
            mapping = ExperimentMapping.load(exp_id)
            return Response.ok(rid, success=True, experiment_id=exp_id,
                               content=output, mapping=asdict(mapping) if mapping else None)
        return Response.err(rid, ErrorCode.NOT_FOUND, f"No output for experiment {exp_id}")

    def _rpc_list_crash_logs(self, params, rid):
        crash_logs = []
        if OUTPUT_DIR.exists():
            for f in sorted(OUTPUT_DIR.glob("*.txt"), reverse=True):
                try:
                    eid = int(f.stem)
                    mapping = ExperimentMapping.load(eid)
                    crash_logs.append({"experiment_id": eid,
                                       "status": mapping.status if mapping else "unknown",
                                       "script": mapping.script if mapping else None})
                except ValueError:
                    pass
        return Response.ok(rid, crash_logs=crash_logs[:20])

    _rpc_handlers = {
        Method.PING: _rpc_ping, Method.RUN: _rpc_run, Method.QUEUE_ADD: _rpc_queue_add,
        Method.CANCEL: _rpc_cancel, Method.PAUSE: _rpc_pause, Method.RESUME: _rpc_resume,
        Method.STATUS: _rpc_status, Method.GPU_PROCESSES: _rpc_gpu_processes,
        Method.QUEUE_LIST: _rpc_queue_list, Method.QUEUE_CLEAR: _rpc_queue_clear,
        Method.QUEUE_REMOVE: _rpc_queue_remove, Method.QUEUE_SET: _rpc_queue_set,
        Method.GET_MAPPING: _rpc_get_mapping, Method.LIST_MAPPINGS: _rpc_list_mappings,
        Method.GET_CRASH_LOG: _rpc_get_crash_log, Method.LIST_CRASH_LOGS: _rpc_list_crash_logs,
    }

    # --- WebSocket server ---

    async def ws_handler(self, websocket):
        self._ws_clients.add(websocket)
        addr = getattr(websocket, "remote_address", "unknown")
        print(f"[rpc] Client connected: {addr}")
        try:
            async for raw in websocket:
                try:
                    msg = parse_message(raw)
                    if not isinstance(msg, Request):
                        await websocket.send(Response.err("unknown", ErrorCode.INVALID_METHOD, "Expected request").to_json())
                        continue
                    print(f"[rpc] {msg.method.value} (id={msg.id})")
                    await websocket.send(self.handle_rpc_request(msg).to_json())
                    await self._flush_events()
                except (ValueError, json.JSONDecodeError) as e:
                    await websocket.send(Response.err("unknown", ErrorCode.INVALID_PARAMS, f"Invalid message: {e}").to_json())
        except websockets.ConnectionClosed:
            pass
        finally:
            self._ws_clients.discard(websocket)
            print(f"[rpc] Client disconnected: {addr}")

    # --- hub sync ---

    def _load_pending_weight_uploads(self):
        """Load durable post-exit checkpoint uploads."""
        if not PENDING_WEIGHTS_FILE.exists():
            return
        try:
            data = json.loads(PENDING_WEIGHTS_FILE.read_text())
            if isinstance(data, dict):
                self._pending_weight_uploads = data
        except (OSError, json.JSONDecodeError):
            print("[hub] Ignoring invalid pending weight upload state", file=sys.stderr)

    def _save_pending_weight_uploads(self):
        """Atomically persist post-exit checkpoint uploads."""
        tmp = PENDING_WEIGHTS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._pending_weight_uploads, indent=2))
        tmp.replace(PENDING_WEIGHTS_FILE)

    def _queue_final_weight_upload(self, experiment_id: int, run_id: Optional[str]):
        """Queue closed checkpoints for upload after the GPU process exits."""
        if not run_id:
            return
        weights_dir = LOGS_DIR / run_id
        if not weights_dir.is_dir():
            return
        files = [
            path.name
            for path in sorted(weights_dir.glob("*.pt"))
            if str(path) not in self._uploaded_weights
        ]
        if not files:
            return
        key = f"{experiment_id}:{run_id}"
        existing = self._pending_weight_uploads.get(key, {})
        pending_files = sorted(set(existing.get("files", [])) | set(files))
        self._pending_weight_uploads[key] = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "files": pending_files,
        }
        try:
            self._save_pending_weight_uploads()
        except OSError as e:
            # The in-memory queue can still upload on the next hub pass. Never
            # let persistence trouble strand daemon lifecycle finalization.
            print(
                f"[hub] Could not persist pending weight uploads: {e}",
                file=sys.stderr,
            )
        print(
            f"[hub] Queued {len(pending_files)} post-exit weight upload(s) "
            f"for experiment {experiment_id}"
        )

    async def _hub_sync_logs(self):
        # Runs sync_logs_up in a fresh subprocess on each cycle.
        #
        # Why: hf_xet (the Rust upload client) accumulates in-process state
        # that occasionally wedges after many consecutive calls — after it
        # wedges, every retry in the same process fails identically with
        # "Data processing error: Format error: I/O error: failed to fill whole
        # buffer". A daemon restart fixes it; a fresh subprocess achieves the
        # same reset at ~200ms cost per cycle without taking down the daemon.
        if not HUB_AVAILABLE:
            return
        if self.state.current_run_id:
            self._materialize_log(self.state.current_run_id)
        cmd = [
            sys.executable, "-u", "-c",
            "from pathlib import Path; from nanorun import hub; "
            f"hub.sync_logs_up(Path({str(ARTIFACTS_DIR)!r}), {self._hub_namespace!r})",
        ]
        try:
            stdout, stderr, returncode = await _run_supervised_subprocess(
                cmd,
                timeout=120,
            )
            if returncode != 0:
                err_text = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"subprocess exit {returncode}: {err_text}")
            self._hub_sync_failures = 0
        except Exception as e:
            self._hub_sync_failures = getattr(self, '_hub_sync_failures', 0) + 1
            print(f"[hub] Log sync failed ({type(e).__name__}): {e}", file=sys.stderr)
            if self._hub_sync_failures >= 3:
                self._emit(Event.HUB_SYNC_FAILED, error=f"{type(e).__name__}: {e}")

    async def _hub_upload_weights(self):
        if not HUB_AVAILABLE or self.state.status != "running":
            return
        run_id, exp_id = self.state.current_run_id, self.state.current_experiment_id
        if not run_id or not exp_id:
            return
        weights_dir = LOGS_DIR / run_id
        if not weights_dir.is_dir():
            return
        now = time.time()
        try:
            for pt_file in weights_dir.glob("*.pt"):
                key = str(pt_file)
                if key in self._uploaded_weights or now - pt_file.stat().st_mtime < WEIGHT_STALENESS_S:
                    continue
                print(f"[hub] Uploading weight: {pt_file.name} for experiment {exp_id}")
                try:
                    await asyncio.to_thread(hub.upload_weight, pt_file, exp_id, pt_file.name, self._hub_namespace)
                    self._uploaded_weights.add(key)
                    print(f"[hub] Uploaded: {pt_file.name}")
                except Exception as e:
                    print(f"[hub] Weight upload failed ({pt_file.name}): {e}", file=sys.stderr)
        except Exception as e:
            print(f"[hub] Weight check failed: {e}", file=sys.stderr)

    async def _hub_upload_pending_weights(self):
        """Upload checkpoints queued after a workload has stopped.

        Files need no age check here: the experiment process has already been
        terminated, so no writer can still be mutating them.
        """
        if not HUB_AVAILABLE or not self._pending_weight_uploads:
            return
        for key, item in list(self._pending_weight_uploads.items()):
            exp_id = item.get("experiment_id")
            run_id = item.get("run_id")
            remaining = []
            for filename in item.get("files", []):
                pt_file = LOGS_DIR / str(run_id) / filename
                if not pt_file.is_file():
                    print(
                        f"[hub] Pending weight disappeared: {pt_file}",
                        file=sys.stderr,
                    )
                    continue
                try:
                    print(
                        f"[hub] Uploading post-exit weight: {filename} "
                        f"for experiment {exp_id}"
                    )
                    await asyncio.to_thread(
                        hub.upload_weight,
                        pt_file,
                        exp_id,
                        filename,
                        self._hub_namespace,
                    )
                    print(f"[hub] Uploaded post-exit weight: {filename}")
                except Exception as e:
                    remaining.append(filename)
                    print(
                        f"[hub] Post-exit weight upload failed ({filename}): {e}",
                        file=sys.stderr,
                    )
            if remaining:
                current = self._pending_weight_uploads.get(key)
                if current is not None:
                    current["files"] = remaining
            else:
                self._pending_weight_uploads.pop(key, None)
            self._save_pending_weight_uploads()

    # --- background tasks ---

    async def _experiment_monitor_task(self):
        # Mutually exclusive branches: when check_current_experiment flips
        # running → idle, the queue is NOT pumped until the next iteration, so
        # at least one poll interval separates a completion (tmux kill) from
        # the next start attempt — the GPU gets time to release memory.
        while self.running:
            try:
                if self.state.status == "running":
                    self.check_current_experiment()
                elif self.state.status == "idle":
                    self._process_queue()
                await self._flush_events()
            except Exception as e:
                print(f"[daemon] Monitor error: {e}", file=sys.stderr)
            await asyncio.sleep(EXPERIMENT_POLL_INTERVAL_S)

    async def _hub_sync_task(self):
        if not HUB_AVAILABLE:
            print("[hub] huggingface_hub not available, hub sync disabled")
            return
        await asyncio.sleep(2)
        while self.running:
            try:
                await self._hub_sync_logs()
                await self._hub_upload_weights()
                await self._hub_upload_pending_weights()
            except Exception as e:
                print(f"[hub] Sync task error: {e}", file=sys.stderr)
            await asyncio.sleep(HUB_SYNC_INTERVAL_S)

    async def _hub_push_queue(self):
        # Event-driven queue-only upload. Mirrors _hub_sync_logs (fresh subprocess to
        # dodge hf_xet in-process wedging) but scoped to queue/*.jsonl and with a short
        # timeout since the payload is tiny. Passes LOGS_DIR (not QUEUE_LOG_DIR) so the
        # queue/ path component is preserved in the remote key. Does NOT emit
        # HUB_SYNC_FAILED — the 15s bulk sync remains the canonical health signal, so a
        # failure here (while the bulk path still works) is only logged.
        if not HUB_AVAILABLE:
            return
        cmd = [
            sys.executable, "-u", "-c",
            "from pathlib import Path; from nanorun import hub; "
            f"hub.sync_queue_up(Path({str(ARTIFACTS_DIR)!r}), {self._hub_namespace!r})",
        ]
        try:
            _, stderr, returncode = await _run_supervised_subprocess(
                cmd,
                timeout=30,
            )
            if returncode != 0:
                err_text = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"subprocess exit {returncode}: {err_text}")
            self._queue_push_failures = 0
        except Exception as e:
            self._queue_push_failures += 1
            print(f"[hub] Queue push failed ({type(e).__name__}): {e}", file=sys.stderr)

    async def _queue_push_task(self):
        # Drains _queue_dirty (set by _emit_queue_changed) and pushes the queue snapshot
        # to the hub. Debounce coalesces bursts (e.g. a sweep add): we always upload the
        # whole current segment file (latest state) and clear the flag before uploading,
        # so no change is lost — a set arriving during the upload just triggers one more
        # idempotent push next iteration.
        if not HUB_AVAILABLE:
            return
        while self.running:
            await self._queue_dirty.wait()
            await asyncio.sleep(QUEUE_PUSH_DEBOUNCE_S)
            self._queue_dirty.clear()
            await self._hub_push_queue()

    # --- main ---

    async def run_async(self):
        print(f"nanorun daemon starting")
        print(
            f"  Session: {self.session_name}  State: {self.state.status}  "
            f"RPC: localhost:{RPC_LISTEN_PORT}  "
            f"Hub: {'yes' if HUB_AVAILABLE else 'no'} ({self._hub_namespace})"
        )

        if not self.acquire_pid_lock():
            sys.exit(1)
        print(f"  PID: {os.getpid()}")

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: setattr(self, 'running', False))

        try:
            server = await websockets.serve(
                self.ws_handler,
                RPC_LISTEN_HOST,
                RPC_LISTEN_PORT,
                max_size=2**24,
            )
            actual_port = server.sockets[0].getsockname()[1]
            if ENDPOINT_FILE is not None:
                ENDPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
                tmp_endpoint = ENDPOINT_FILE.with_suffix(".tmp")
                tmp_endpoint.write_text(
                    json.dumps({"pid": os.getpid(), "port": actual_port})
                )
                tmp_endpoint.replace(ENDPOINT_FILE)
            print(f"  WebSocket server listening on {RPC_LISTEN_HOST}:{actual_port}")

            # Auto-start queue if idle with pending items
            if self.state.status == "idle" and self.read_queue():
                print(f"[daemon] Resuming queue ({len(self.read_queue())} pending)")
                self._process_queue()

            tasks = [
                asyncio.create_task(self._experiment_monitor_task()),
                asyncio.create_task(self._hub_sync_task()),
                asyncio.create_task(self._queue_push_task()),
            ]
            while self.running:
                await asyncio.sleep(0.5)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            server.close()
            await server.wait_closed()
        finally:
            if ENDPOINT_FILE is not None:
                try:
                    data = json.loads(ENDPOINT_FILE.read_text())
                    if data.get("pid") == os.getpid():
                        ENDPOINT_FILE.unlink(missing_ok=True)
                except (OSError, json.JSONDecodeError):
                    pass
            self.release_pid_lock()
        print("nanorun daemon stopped")


def main():
    parser = argparse.ArgumentParser(description="nanorun daemon")
    parser.add_argument("--session", default=None)
    parser.add_argument("--hub-session", default=None)
    parser.add_argument("--repo-dir", type=Path, default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--listen-host", default=None)
    parser.add_argument("--endpoint-file", type=Path, default=None)
    parser.add_argument("--device-lock-file", type=Path, default=None)
    parser.add_argument("--port", type=int, default=RPC_PORT)
    parser.add_argument("--tmux-session", default=None)
    args = parser.parse_args()
    configure_runtime(
        repo_dir=args.repo_dir,
        state_dir=args.state_dir,
        artifacts_dir=args.artifacts_dir,
        rpc_host=args.listen_host,
        rpc_port=args.port,
        endpoint_file=args.endpoint_file,
        device_lock_file=args.device_lock_file,
        tmux_session=args.tmux_session,
    )
    session_name = args.session
    if not session_name:
        state_file = DAEMON_DIR / "state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                session_name = state.get("session_name")
            except (json.JSONDecodeError, OSError):
                pass
    if not session_name:
        print("ERROR: --session is required (no previous session found in .daemon/state.json)")
        sys.exit(1)
    asyncio.run(
        NanorunDaemon(
            session_name=session_name,
            hub_session=args.hub_session,
        ).run_async()
    )


if __name__ == "__main__":
    main()
