"""Remote daemon — experiment execution, queue management, hub sync.

Runs on the remote GPU machine. Accepts commands over WebSocket RPC
(localhost:9321, tunneled through SSH). Executes experiments in tmux,
monitors completion, pushes logs/weights to HuggingFace Bucket.
"""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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

try:
    from nanorun import hub
    HUB_AVAILABLE = True
except Exception:
    HUB_AVAILABLE = False

REPO_DIR = Path.home() / "nanorun"
DAEMON_DIR = REPO_DIR / ".daemon"
MAPPINGS_DIR = DAEMON_DIR / "mappings"
OUTPUT_DIR = DAEMON_DIR / "output"
QUEUE_FILE = DAEMON_DIR / "queue.txt"
STATE_FILE = DAEMON_DIR / "state.json"
PID_FILE = DAEMON_DIR / "daemon.pid"
LOGS_DIR = REPO_DIR / "logs"
MAPPINGS_LOG_DIR = LOGS_DIR / "mappings"

TMUX_SESSION = "nanorun"
HUB_SYNC_INTERVAL_S = 5
EXPERIMENT_POLL_INTERVAL_S = 2
WEIGHT_STALENESS_S = 3
CODE_HASH_LENGTH = 12
GIT_HASH_LENGTH = 12
TMUX_WINDOW_NAME_MAX = 40
LOG_TAIL_BYTES = 50_000
MAPPINGS_SEGMENT_LINES = 500

RUN_ID_PATTERN = re.compile(
    r"logs/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\.txt"
)
FRONTMATTER_PATTERN = re.compile(r"^[\s]*(?:\"\"\"|'{3})(.*?)(?:\"\"\"|'{3})", re.DOTALL)
PARENT_FIELD_PATTERN = re.compile(r"parent:\s*(.+?)(?:\n|$)")
KERNELS_FIELD_PATTERN = re.compile(r"kernels:\s*(.+?)(?:\n|$)")
STEP_PATTERN = re.compile(r"^step:(\d+)/(\d+)", re.MULTILINE)
SEGMENT_FILE_PATTERN = re.compile(r"^mappings-(\d{6})\.jsonl$")


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


class NanorunDaemon:
    def __init__(self, session_name: str = "default"):
        self.session_name = session_name
        self.state = DaemonState.load()
        self.state.session_name = session_name
        self.running = True
        self._pid_file_handle = None
        self._pending_events: List[EventMessage] = []
        self._ws_clients: Set = set()
        self._uploaded_weights: Set[str] = set()
        self._startup_time = time.monotonic()
        for d in [DAEMON_DIR, MAPPINGS_DIR, OUTPUT_DIR, LOGS_DIR, MAPPINGS_LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)
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

    # --- events ---

    def _emit(self, event: Event, **data):
        self._pending_events.append(EventMessage(event=event, data=data))

    def _emit_queue_changed(self):
        self._emit(Event.QUEUE_CHANGED, queue=[asdict(item) for item in self.read_queue()])

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
            try:
                print(f"Another daemon is already running (PID {PID_FILE.read_text().strip()})", file=sys.stderr)
            except Exception:
                print("Another daemon is already running", file=sys.stderr)
            return False

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

    # --- GPU ---

    def get_gpu_processes(self) -> List[Dict[str, Any]]:
        cmd = "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits"
        success, stdout, _ = self._run_cmd(cmd)
        processes = []
        if success and stdout.strip():
            for line in stdout.strip().split("\n"):
                parts = line.strip().split(", ")
                if len(parts) >= 3:
                    try:
                        processes.append({"pid": int(parts[0]), "name": parts[1], "memory_mb": int(parts[2])})
                    except ValueError:
                        pass
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

    def _parse_frontmatter(self, script_path: str) -> tuple[Optional[str], Optional[str]]:
        full_path = REPO_DIR / script_path
        if not full_path.exists():
            return None, None
        try:
            content = full_path.read_text()
            match = FRONTMATTER_PATTERN.match(content)
            if not match:
                return None, None
            docstring = match.group(1)
            parent = PARENT_FIELD_PATTERN.search(docstring)
            kernels = KERNELS_FIELD_PATTERN.search(docstring)
            return (parent.group(1).strip() if parent else None, kernels.group(1).strip() if kernels else None)
        except Exception:
            return None, None

    def _compute_file_hash(self, rel_path: str) -> Optional[str]:
        full_path = REPO_DIR / rel_path
        if not full_path.exists():
            return None
        return hashlib.sha256(full_path.read_bytes()).hexdigest()[:CODE_HASH_LENGTH]

    def _compute_code_hash(self, script: str, kernels_path: Optional[str] = None) -> Optional[str]:
        script_full = REPO_DIR / script
        if not script_full.exists():
            return None
        script_bytes = script_full.read_bytes()
        if kernels_path:
            kernels_full = REPO_DIR / kernels_path
            if kernels_full.exists():
                return hashlib.sha256(script_bytes + b"\n---KERNELS---\n" + kernels_full.read_bytes()).hexdigest()[:CODE_HASH_LENGTH]
        return hashlib.sha256(script_bytes).hexdigest()[:CODE_HASH_LENGTH]

    def _kernels_symlink_cmd(self, script_path: str, kernels_path: str) -> str:
        script_dir = Path(script_path).parent
        kernels_dir = Path(kernels_path).parent
        target = Path(kernels_path).name if str(kernels_dir) == str(script_dir) else f"$HOME/nanorun/{kernels_path}"
        return f"ln -sf {target} {script_dir}/triton_kernels.py"

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
        ok, _, err = self._run_cmd(f"tmux new-window -t {TMUX_SESSION} -n '{window}'")
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

    def _resolve_script_path(self, script: str) -> str:
        """Resolve script path with case-insensitive lookup on Linux."""
        full = REPO_DIR / script
        if full.exists():
            return script
        # Case-insensitive search: walk each path component
        current = REPO_DIR
        for part in Path(script).parts:
            match = None
            try:
                for entry in current.iterdir():
                    if entry.name.lower() == part.lower():
                        match = entry
                        break
            except OSError:
                return script
            if not match:
                return script
            current = match
        return str(current.relative_to(REPO_DIR))

    def _build_run_command(self, script: str, env_vars: Dict[str, str], gpus: int,
                           exp_id: int, cmd_prefix: Optional[str] = None,
                           symlink_cmd: Optional[str] = None) -> str:
        env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
        if env_str:
            env_str += " "
        torchrun = f"torchrun --standalone --nproc_per_node={gpus} {script}"
        if cmd_prefix:
            torchrun = f"{cmd_prefix} {torchrun}"
        output_file = OUTPUT_DIR / f"{exp_id}.txt"
        parts = ["source $HOME/.local/bin/env", "source .venv/bin/activate"]
        if symlink_cmd:
            parts.append(symlink_cmd)
        parts.append(f"{env_str}{torchrun} 2>&1 | tee {output_file}")
        return " && ".join(parts)

    def start_experiment(self, experiment_id: int, script: str, env_vars: Dict[str, str],
                         gpus: int = 1, gpu_type: str = "H100", name: Optional[str] = None,
                         track: Optional[str] = None, cmd_prefix: Optional[str] = None) -> Dict[str, Any]:
        if self.state.status == "running":
            return {"success": False, "error": f"Already running experiment {self.state.current_experiment_id}"}

        script = self._resolve_script_path(script)

        gpu_procs = self.get_gpu_processes()
        if gpu_procs:
            pids = [p["pid"] for p in gpu_procs]
            total_mem = sum(p["memory_mb"] for p in gpu_procs)
            return {"success": False, "error": f"GPU busy: PIDs {pids} ({total_mem}MB)", "gpu_processes": gpu_procs}

        parent_path, kernels_path = self._parse_frontmatter(script)
        symlink_cmd = self._kernels_symlink_cmd(script, kernels_path) if kernels_path else None
        code_hash = self._compute_code_hash(script, kernels_path)
        if not code_hash:
            return {"success": False, "error": f"Script not found: {script}"}

        timestamp = datetime.now(timezone.utc).strftime("%m%d_%H%M%S")
        base_name = name or Path(script).stem
        window_name = f"{timestamp}_{base_name}"[:TMUX_WINDOW_NAME_MAX]

        cmd = self._build_run_command(script, env_vars, gpus, experiment_id, cmd_prefix, symlink_cmd)
        if not self._tmux_create(window_name, cmd):
            return {"success": False, "error": "Failed to create tmux window"}

        mapping = ExperimentMapping(
            experiment_id=experiment_id, run_id=None, script=script,
            code_hash=code_hash, env_vars=env_vars, gpus=gpus, gpu_type=gpu_type,
            tmux_window=window_name, log_file=None,
            started_at=datetime.now(timezone.utc).isoformat(),
            track=track, name=name, git_commit=self.get_git_commit(),
            parent_hash=self._compute_file_hash(parent_path) if parent_path else None,
            kernels_path=kernels_path,
        )
        mapping.save()

        self.state.status = "running"
        self.state.current_experiment_id = experiment_id
        self.state.current_window = window_name
        self.state.current_run_id = None
        self.state.save()
        self._uploaded_weights.clear()

        self._emit(Event.EXPERIMENT_STARTED, **asdict(mapping))
        return {"success": True, "code_hash": code_hash, "window_name": window_name}

    def check_current_experiment(self):
        """Poll experiment status: detect run_id, handle completion."""
        if self.state.status != "running" or not self.state.current_window:
            return
        window = self.state.current_window

        if not self._tmux_window_exists(window) or self._tmux_pane_dead(window):
            if self._tmux_window_exists(window):
                self._tmux_kill(window)
            self._handle_experiment_finished()
            return

        # Still running — try to detect run_id from log file existence
        if not self.state.current_run_id:
            run_id = self._detect_run_id()
            if run_id:
                self.state.current_run_id = run_id
                self.state.save()
                mapping = ExperimentMapping.load(self.state.current_experiment_id)
                if mapping:
                    mapping.run_id = run_id
                    mapping.log_file = f"logs/{run_id}.txt"
                    mapping.save()
                self._emit(Event.EXPERIMENT_RUN_ID,
                           experiment_id=self.state.current_experiment_id, run_id=run_id)

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
        self._process_queue()

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
        if self.state.status in ("paused", "running"):
            return
        items = self.read_queue()
        while items:
            item = items[0]
            print(f"Starting queued experiment {item.experiment_id}: {item.script}")
            result = self.start_experiment(
                experiment_id=item.experiment_id, script=item.script,
                env_vars=item.env_vars, gpus=item.gpus, gpu_type=item.gpu_type,
                name=item.name, track=item.track, cmd_prefix=item.cmd_prefix,
            )
            items.pop(0)
            self.write_queue(items)
            self._emit_queue_changed()
            if result["success"]:
                return
            print(f"Failed to start: {result.get('error')} — skipping", file=sys.stderr)
        print("Queue is empty")

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
            env_vars=params.get("env_vars", {}), gpus=params.get("gpus", 1),
            gpu_type=params.get("gpu_type", "H100"), name=params.get("name"), track=params.get("track"),
        )
        return Response.ok(rid, **result) if result["success"] else Response.err(rid, ErrorCode.CONFLICT, result["error"])

    def _rpc_queue_add(self, params, rid):
        item = QueuedItem(
            experiment_id=params["experiment_id"], script=params["script"],
            env_vars=params.get("env_vars", {}), gpus=params.get("gpus", 1),
            gpu_type=params.get("gpu_type", "H100"), name=params.get("name"),
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
        return Response.ok(rid, status=self.state.status,
            current_experiment_id=self.state.current_experiment_id,
            current_window=self.state.current_window,
            current_run_id=self.state.current_run_id,
            queue_length=len(queue), queue=[asdict(item) for item in queue],
            gpu_processes=self.get_gpu_processes())

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
                       env_vars=d.get("env_vars", {}), gpus=d.get("gpus", 1),
                       gpu_type=d.get("gpu_type", "H100"), name=d.get("name"),
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
        cmd = [
            sys.executable, "-u", "-c",
            "from pathlib import Path; from nanorun import hub; "
            f"hub.sync_logs_up(Path({str(LOGS_DIR)!r}), {self.session_name!r})",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode != 0:
                err_text = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"subprocess exit {proc.returncode}: {err_text}")
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
                    await asyncio.to_thread(hub.upload_weight, pt_file, exp_id, pt_file.name, self.session_name)
                    self._uploaded_weights.add(key)
                    print(f"[hub] Uploaded: {pt_file.name}")
                except Exception as e:
                    print(f"[hub] Weight upload failed ({pt_file.name}): {e}", file=sys.stderr)
        except Exception as e:
            print(f"[hub] Weight check failed: {e}", file=sys.stderr)

    # --- background tasks ---

    async def _experiment_monitor_task(self):
        while self.running:
            try:
                self.check_current_experiment()
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
            except Exception as e:
                print(f"[hub] Sync task error: {e}", file=sys.stderr)
            await asyncio.sleep(HUB_SYNC_INTERVAL_S)

    # --- main ---

    async def run_async(self):
        print(f"nanorun daemon starting")
        print(f"  Session: {self.session_name}  State: {self.state.status}  RPC: localhost:{RPC_PORT}  Hub: {'yes' if HUB_AVAILABLE else 'no'}")

        if not self.acquire_pid_lock():
            sys.exit(1)
        print(f"  PID: {os.getpid()}")

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: setattr(self, 'running', False))

        try:
            server = await websockets.serve(self.ws_handler, "localhost", RPC_PORT, max_size=2**24)
            print(f"  WebSocket server listening on localhost:{RPC_PORT}")

            # Auto-start queue if idle with pending items
            if self.state.status == "idle" and self.read_queue():
                print(f"[daemon] Resuming queue ({len(self.read_queue())} pending)")
                self._process_queue()

            tasks = [
                asyncio.create_task(self._experiment_monitor_task()),
                asyncio.create_task(self._hub_sync_task()),
            ]
            while self.running:
                await asyncio.sleep(0.5)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            server.close()
            await server.wait_closed()
        finally:
            self.release_pid_lock()
        print("nanorun daemon stopped")


def main():
    parser = argparse.ArgumentParser(description="nanorun remote daemon")
    parser.add_argument("--session", default="default")
    args = parser.parse_args()
    asyncio.run(NanorunDaemon(session_name=args.session).run_async())


if __name__ == "__main__":
    main()
