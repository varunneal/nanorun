"""Local metrics daemon - event-driven, multi-session.

Two independent connections per session:
  1. Remote daemon (SSH/WebSocket) — experiment events, queue state, RPC
  2. HuggingFace Hub (HTTPS) — log/mapping sync

These are decoupled: hub sync continues when SSH drops and vice versa.
One main thread for session discovery and signal handling.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from .config import Config, SessionConfig, infer_track_from_path
from .rpc_client import RpcClient, RpcError, kill_tunnel
from .rpc_types import Event, EventMessage, Method
from .tracker import (
    get_db, close_db, parse_metric_line, record_metric,
    update_experiment_status, update_experiment_metadata,
    get_experiment, create_experiment_from_mapping,
    set_crash_log, get_crash_log, terminate_session_experiments,
)


def safe_json_load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _setup_logging():
    config_dir = Config.get_config_dir()
    log_file = config_dir / "local_daemon" / "daemon.log"
    log_file.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("local_daemon")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger

log = _setup_logging()


@dataclass
class DaemonPaths:
    _config_dir: Path = field(default_factory=lambda: Config.get_config_dir())

    @property
    def daemon_dir(self) -> Path:
        d = self._config_dir / "local_daemon"; d.mkdir(exist_ok=True); return d

    @property
    def logs_dir(self) -> Path:
        d = self._config_dir / "logs"; d.mkdir(exist_ok=True); return d

    @property
    def pid_file(self) -> Path:
        return self.daemon_dir / "daemon.pid"

    def state_file(self, s: str) -> Path:
        return Config.get_session_state_dir(s) / "state.json"

    def queue_cache_file(self, s: str) -> Path:
        return Config.get_session_state_dir(s) / "queue_cache.json"

    def crashes_file(self, s: str) -> Path:
        return Config.get_session_state_dir(s) / "crashes.json"

    def events_file(self, s: str) -> Path:
        return Config.get_session_state_dir(s) / "events.log"

PATHS = DaemonPaths()


def get_crashes_file(session_name: Optional[str] = None) -> Path:
    return PATHS.crashes_file(session_name or Config.get_active_session_name() or "_default")

def get_queue_cache_file(session_name: Optional[str] = None) -> Path:
    return PATHS.queue_cache_file(session_name or Config.get_active_session_name() or "_default")

def get_events_file(session_name: Optional[str] = None) -> Path:
    return PATHS.events_file(session_name or Config.get_active_session_name() or "_default")


def record_crash(session_name: str, experiment_id: int, crash_log: Optional[str] = None):
    crashes_file = PATHS.crashes_file(session_name)
    crashes = safe_json_load(crashes_file, default=[])
    crashes.append({
        "experiment_id": experiment_id, "session_name": session_name,
        "crash_log_snippet": crash_log[-2000:] if crash_log else None,
        "timestamp": datetime.now(timezone.utc).isoformat(), "seen": False,
    })
    crashes_file.write_text(json.dumps(crashes[-20:], indent=2))
    try:
        subprocess.run([
            "osascript", "-e",
            f'display notification "Experiment {experiment_id} failed ({session_name})" '
            f'with title "nanorun" sound name "Basso"'
        ], capture_output=True, timeout=5)
    except Exception:
        pass


def _append_queue_history(session_name: str, ts: str, queue_items: list):
    history_file = PATHS._config_dir / "queue_history.jsonl"
    entry = json.dumps({"ts": ts, "session": session_name, "queue": queue_items})
    with open(history_file, "a") as f:
        f.write(entry + "\n")


# Guards every write to queue_cache.json so the SessionTracker (event/STATUS
# fast-path) and the HubSyncer (hub-snapshot reconcile) can't clobber each other.
_queue_cache_lock = threading.Lock()


def _queue_ts_should_overwrite(new_ts, old_ts) -> bool:
    """Precedence test: True if a write stamped `new_ts` should overwrite a cache
    stamped `old_ts`. The skip condition is `old_ts >= new_ts`, so a duplicate
    same-ts write is a harmless no-op (an event and its hub snapshot line carry
    the same remote ts). On parse failure fall back to lexicographic compare; if
    that is ambiguous, favor freshness (write)."""
    if not old_ts:
        return True
    try:
        return datetime.fromisoformat(new_ts) > datetime.fromisoformat(old_ts)
    except (ValueError, TypeError):
        pass
    try:
        return new_ts > old_ts  # lexicographic fallback (ISO8601 sorts by time)
    except TypeError:
        return True  # ambiguous — favor freshness


def _write_queue_cache_with_precedence(session_name, ts, queue_items, connected=None):
    """Write queue_cache.json under the newest-ts-wins precedence rule.

    Two threads write this file: the SessionTracker (event / STATUS fast-path,
    passes connected=True) and the HubSyncer (hub-snapshot reconcile, passes
    connected=None to preserve). Both stamp `synced_at` with a remote-clock ts so
    they're directly comparable; the newer ts wins.
    """
    with _queue_cache_lock:
        path = get_queue_cache_file(session_name)
        existing = safe_json_load(path, default=None)
        old_ts = existing.get("synced_at") if isinstance(existing, dict) else None
        overwrite = _queue_ts_should_overwrite(ts, old_ts)

        if connected is not None:
            resolved_connected = connected
        elif isinstance(existing, dict):
            resolved_connected = existing.get("connected", True)
        else:
            resolved_connected = True

        if overwrite:
            data = {"synced_at": ts, "queue": queue_items, "connected": resolved_connected}
        else:
            # Stale queue write: preserve existing queue/synced_at. Still apply an
            # explicit connected flag if it changed; otherwise nothing to do.
            if not isinstance(existing, dict):
                return
            if connected is None or existing.get("connected") == connected:
                return
            data = dict(existing)
            data["connected"] = connected

        try:
            path.write_text(json.dumps(data, indent=2))
        except OSError:
            pass


# Track file offsets for hub-syncer mappings ingestion (keyed by session_name -> {filename: offset})
_hub_mappings_offsets: Dict[str, Dict[str, int]] = {}


def _ingest_mappings_for_session(session_name: str):
    """Read mappings from hub-synced log directory and create/update experiment DB entries.

    Called by HubSyncer for sessions whose SessionTracker is not alive.
    """
    session_logs = PATHS.logs_dir / session_name
    if not session_logs.exists():
        return

    offsets = _hub_mappings_offsets.setdefault(session_name, {})

    # Legacy flat file
    legacy = session_logs / "mappings.jsonl"
    if legacy.exists():
        size = legacy.stat().st_size
        if size > offsets.get("legacy", 0):
            with open(legacy) as f:
                f.seek(offsets.get("legacy", 0))
                new_content = f.read()
                offsets["legacy"] = f.tell()
            _ingest_mapping_lines_for_session(session_name, new_content)

    # Segmented files
    segments_dir = session_logs / "mappings"
    if segments_dir.exists():
        for path in sorted(segments_dir.glob("mappings-*.jsonl")):
            size = path.stat().st_size
            offset = offsets.get(path.name, 0)
            if size <= offset:
                continue
            with open(path) as f:
                f.seek(offset)
                new_content = f.read()
                offsets[path.name] = f.tell()
            _ingest_mapping_lines_for_session(session_name, new_content)


def _ingest_mapping_lines_for_session(session_name: str, content: str):
    """Parse mapping lines and create/update experiments in DB."""
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            mapping = json.loads(line)
        except json.JSONDecodeError:
            continue
        exp_id = mapping.get("experiment_id")
        if not exp_id:
            continue
        existing = get_experiment(exp_id)
        if not existing:
            script = mapping.get("script", "unknown")
            name = mapping.get("name") or Path(script).stem
            track = mapping.get("track") or infer_track_from_path(script)
            try:
                create_experiment_from_mapping(
                    experiment_id=exp_id, name=name, script=script,
                    status=mapping.get("status", "running"), track=track,
                    code_hash=mapping.get("code_hash"), remote_run_id=mapping.get("run_id"),
                    tmux_window=mapping.get("tmux_window"), started_at=mapping.get("started_at"),
                    finished_at=mapping.get("finished_at"), env_vars=mapping.get("env_vars"),
                    gpus=mapping.get("gpus", 1), gpu_type=mapping.get("gpu_type", "H100"),
                    git_commit=mapping.get("git_commit"), parent_hash=mapping.get("parent_hash"),
                    kernels_path=mapping.get("kernels_path"), session_name=session_name,
                )
                log.info(f"[hub] Created experiment {exp_id} ({Path(script).name}) for {session_name}")
            except Exception as e:
                log.error(f"[hub] Failed to create experiment {exp_id}: {e}")
        else:
            remote_status = mapping.get("status")
            if remote_status and remote_status != existing.status:
                if existing.status == "completed" and remote_status == "failed":
                    pass
                elif remote_status in ("completed", "failed") or existing.status in ("running", "queued"):
                    update_experiment_status(exp_id, remote_status)
            if mapping.get("run_id") and not existing.remote_run_id:
                update_experiment_metadata(exp_id, remote_run_id=mapping["run_id"])


def _ingest_queue_for_session(session_name: str):
    """Reconcile queue_cache.json from the hub-synced queue snapshot stream.

    Mirrors _ingest_mappings_for_session: SSH-independent, runs every hub cycle.
    Latest queue state = the LAST non-empty line of the HIGHEST-index queue
    segment. Reading the tail each cycle is fine (segments are <=500 small lines)
    and idempotent — the precedence writer discards stale writes — so no offset
    tracking is needed.
    """
    session_logs = PATHS.logs_dir / session_name
    queue_dir = session_logs / "queue"
    if not queue_dir.exists():
        return

    # Walk segments from highest index down; take the last non-empty line of the
    # newest segment. If the newest segment is empty, fall back to the previous.
    last_line = None
    for path in sorted(queue_dir.glob("queue-*.jsonl"), reverse=True):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if line.strip():
                last_line = line
                break
        if last_line is not None:
            break
    if last_line is None:
        return

    try:
        snapshot = json.loads(last_line)
    except json.JSONDecodeError:
        return
    ts = snapshot.get("ts")
    if not ts:
        # No ordering key: skip rather than clobber a fresh event write with an
        # unorderable snapshot (contract: prefer skipping to a local-now stamp).
        return
    queue = snapshot.get("queue", [])
    _write_queue_cache_with_precedence(session_name, ts, queue, connected=None)


def _sync_all_logs(session_name: str):
    from . import hub
    session_config = Config.load_session(session_name)
    session_logs_dir = PATHS.logs_dir / session_name
    if session_config and session_config.session_type == "iris":
        backend = hub._IrisBackend(session_config)
        backend.sync_logs_down(session_logs_dir, session_name)
    else:
        hub.sync_logs_down(session_logs_dir, session_name)


class HubSyncer:
    """Single HTTPS connection to Hub, syncs logs for all sessions.

    Runs in its own thread, decoupled from SSH/RPC connections.
    One instance per daemon (not per session) since it's a single auth/endpoint.
    """

    SYNC_INTERVAL = 10.0

    def __init__(self, daemon: "LocalMetricsDaemon"):
        self.daemon = daemon
        self.running = True
        self.status: str = "disconnected"  # disconnected, connected
        self.last_error: Optional[str] = None
        self.last_sync_at: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._was_connected: bool = False
        self._last_sync: float = 0

    def start(self):
        self._thread = threading.Thread(
            target=self._loop, name="hub-syncer", daemon=True,
        )
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def reconnectable(self) -> bool:
        return not self.is_alive and self._was_connected

    def event(self, message: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [hub] {message}"
        print(line, flush=True)

    def _loop(self):
        log.info("[hub] Starting hub syncer")
        self.event("Hub syncer starting")
        self.status = "connected"
        self._was_connected = True
        while self.running:
            now = time.monotonic()
            if now - self._last_sync >= self.SYNC_INTERVAL:
                self._do_sync()
                self._last_sync = now
            time.sleep(1)
        close_db()  # release this thread's DB connection (final WAL checkpoint)
        log.info("[hub] Syncer stopped")

    SYNC_TIMEOUT = 60.0

    def _log_session_error(self, session_name: str, message: str, *, exc_info: bool = False):
        previous = self._session_errors_logged.get(session_name)
        if previous != message:
            log.warning(message, exc_info=exc_info)
            self._session_errors_logged[session_name] = message
        self.last_error = message

    def _do_sync(self):
        if not hasattr(self, '_session_errors_logged'):
            self._session_errors_logged = {}

        # Honor the per-session sync_paused flag: skip its scan entirely. For iris
        # sessions this is the whole background footprint (W&B + iris job polling);
        # on-demand CLI commands still hit the backend directly and are unaffected.
        # The flag is re-read every cycle, so pause/resume applies without a restart.
        all_sessions = Config.list_sessions()
        paused = {s.name for s in all_sessions if getattr(s, "sync_paused", False)}
        prev_paused = getattr(self, "_paused_sessions", set())
        for name in sorted(paused - prev_paused):
            self.event(f"[{name}] sync paused — skipping background metric/log scan")
        for name in sorted(prev_paused - paused):
            self.event(f"[{name}] sync resumed")
        self._paused_sessions = paused
        sessions = [s.name for s in all_sessions if s.name not in paused]

        # Sync all sessions in parallel so one slow/hanging session doesn't block others
        sync_errors: dict[str, Exception] = {}

        def _sync_one(name):
            try:
                self._sync_with_timeout(name)
            except Exception as e:
                sync_errors[name] = e

        threads = [threading.Thread(target=_sync_one, args=(s,), daemon=True) for s in sessions]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.SYNC_TIMEOUT + 5)

        # Parse metrics and handle errors sequentially
        for session_name in sessions:
            had_error = False
            if session_name in sync_errors:
                e = sync_errors[session_name]
                had_error = True
                if isinstance(e, TimeoutError):
                    self._log_session_error(
                        session_name,
                        f"[hub] Sync timed out for {session_name}",
                        exc_info=True,
                    )
                else:
                    self._log_session_error(
                        session_name,
                        f"[hub] Sync failed for {session_name}: {e}",
                        exc_info=True,
                    )

            # SSH-independent queue reconcile — must run for EVERY session, whether
            # or not its SessionTracker is alive. Isolated so a failure here can't
            # abort the loop or the metrics parse below.
            try:
                _ingest_queue_for_session(session_name)
            except Exception as e:
                had_error = True
                self._log_session_error(
                    session_name,
                    f"[hub] Queue ingest failed for {session_name}: {e}",
                    exc_info=True,
                )

            try:
                self._parse_metrics_for_session(session_name)
            except Exception as e:
                had_error = True
                self._log_session_error(
                    session_name,
                    f"[hub] Metric parse failed for {session_name}: {e}",
                    exc_info=True,
                )
                continue

            if not had_error:
                self._session_errors_logged.pop(session_name, None)
                self.last_error = None
        self.last_sync_at = datetime.now().strftime("%H:%M:%S")

    def _parse_metrics_for_session(self, session_name: str):
        """Parse metrics from local log files. Dispatches to correct parser by session type."""
        session_config = Config.load_session(session_name)
        if session_config and session_config.session_type == "iris":
            self._parse_iris_metrics(session_name)
        else:
            # For SSH sessions: parse metrics here if the session tracker is dead
            # OR alive but disconnected (it only parses while in the event loop).
            tracker = self.daemon.trackers.get(session_name)
            tracker_connected = (
                isinstance(tracker, SessionTracker)
                and tracker._thread is not None
                and tracker._thread.is_alive()
                and tracker.state.status == "connected"
            )
            if not tracker_connected:
                self._parse_ssh_metrics(session_name)

    def _parse_ssh_metrics(self, session_name: str):
        """Parse metrics from local log files for SSH sessions whose tracker is dead."""
        session_logs = PATHS.logs_dir / session_name
        if not session_logs.exists():
            return

        # Ingest mappings first so new experiments get created before we parse their logs
        _ingest_mappings_for_session(session_name)

        conn = get_db()
        rows = conn.execute(
            "SELECT id, remote_run_id FROM experiments "
            "WHERE remote_run_id IS NOT NULL AND session_name = ?",
            (session_name,),
        ).fetchall()
        conn.close()

        for row in rows:
            exp_id, run_id = row["id"], row["remote_run_id"]
            local_log = session_logs / f"{run_id}.txt"
            if not local_log.exists():
                flat = PATHS.logs_dir / f"{run_id}.txt"
                if flat.exists():
                    local_log = flat
                else:
                    continue
            if local_log.stat().st_size <= _log_offsets.get(run_id, 0):
                continue
            recorded, _ = _parse_local_metrics(exp_id, run_id, local_log)
            if recorded:
                self.event(f"[{session_name}] Synced {recorded} metric steps for experiment {exp_id}")

    def _parse_iris_metrics(self, session_name: str):
        """Parse iris log files (keyed by wandb_run_id) + status reconciliation."""
        from .tracker import parse_metric_line, record_metric, update_experiment_status
        from .hub import _IrisBackend, _IRIS_STATE_MAP

        session_logs = PATHS.logs_dir / session_name
        if not session_logs.exists():
            return

        session_config = Config.load_session(session_name)
        if not session_config:
            return

        # Reconcile experiment status from iris job list (keyed by iris job ID)
        backend = _IrisBackend(session_config)
        jobs = backend.list_jobs()
        job_states = {}
        for j in jobs:
            name = j.get("name", "")
            if not name or ":" in name.split("/")[-1]:
                continue
            job_states[name] = _IRIS_STATE_MAP.get(j.get("state", ""), "")

        import json as _json
        conn = get_db()
        rows = conn.execute(
            "SELECT id, remote_run_id, status, env_vars FROM experiments "
            "WHERE remote_run_id IS NOT NULL AND session_name = ?",
            (session_name,),
        ).fetchall()
        conn.close()

        for row in rows:
            exp_id, local_status = row["id"], row["status"]
            try:
                env = _json.loads(row["env_vars"]) if row["env_vars"] else {}
            except _json.JSONDecodeError as e:
                log.warning(
                    f"[hub] Invalid env_vars JSON for iris experiment {exp_id}: {e}",
                    exc_info=True,
                )
                continue
            iris_job_id = env.get("_iris_job_id")
            if not iris_job_id:
                continue
            iris_status = job_states.get(iris_job_id)
            if iris_status and iris_status != local_status:
                if local_status == "completed" and iris_status == "failed":
                    continue
                update_experiment_status(exp_id, iris_status)
                if iris_status != "running":
                    self.event(f"[{session_name}] Experiment {exp_id} → {iris_status}")

        for row in rows:
            exp_id, wandb_run_id = row["id"], row["remote_run_id"]
            log_path = session_logs / f"{wandb_run_id}.txt"
            if not log_path.exists():
                continue

            with open(log_path) as f:
                new_content = f.read()

            new_count = 0
            found_final = False
            for line in new_content.split("\n"):
                m = parse_metric_line(line)
                if not m:
                    continue
                is_final = m["total_steps"] is not None and m["step"] >= m["total_steps"]
                found_final = found_final or is_final
                inserted = record_metric(
                    experiment_id=exp_id, step=m["step"],
                    total_steps=m["total_steps"], val_loss=m["val_loss"],
                    train_loss=m.get("train_loss"),
                    train_time_ms=m["train_time_ms"],
                    step_avg_ms=m.get("step_avg_ms"), is_final_step=is_final,
                )
                if inserted:
                    new_count += 1

            if found_final and row["status"] != "completed":
                update_experiment_status(exp_id, "completed")
                self.event(f"[{session_name}] Experiment {exp_id} completed")

            if new_count:
                self.event(f"[{session_name}] Synced {new_count} metric steps for experiment {exp_id}")

    def _sync_with_timeout(self, session_name: str):
        """Run sync in a worker thread with a timeout to prevent hanging."""
        exc_box = [None]

        def _worker():
            try:
                _sync_all_logs(session_name)
            except Exception as e:
                exc_box[0] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=self.SYNC_TIMEOUT)
        if t.is_alive():
            raise TimeoutError(f"Sync for {session_name} hung for >{self.SYNC_TIMEOUT}s")
        if exc_box[0]:
            raise exc_box[0]


_log_offsets: dict[str, int] = {}

def _parse_local_metrics(experiment_id: int, run_id: str, log_path: Path) -> tuple[int, bool]:
    """Parse metrics from local log file. Marks experiment completed if final step found.
    Returns (count, found_final)."""
    if not log_path.exists():
        return 0, False
    file_size = log_path.stat().st_size
    offset = _log_offsets.get(run_id, 0)
    if file_size <= offset:
        return 0, False

    with open(log_path) as f:
        f.seek(offset)
        new_content = f.read()
        new_offset = f.tell()

    recorded, found_final = 0, False
    for line in new_content.split("\n"):
        metric = parse_metric_line(line)
        if metric:
            is_final = metric["step"] == metric["total_steps"]
            found_final = found_final or is_final
            inserted = record_metric(
                experiment_id=experiment_id, step=metric["step"],
                total_steps=metric["total_steps"], val_loss=metric["val_loss"],
                train_loss=metric.get("train_loss"),
                train_time_ms=metric["train_time_ms"],
                step_avg_ms=metric.get("step_avg_ms"), is_final_step=is_final,
            )
            if inserted:
                recorded += 1

    _log_offsets[run_id] = new_offset
    if recorded:
        log.info(f"Recorded {recorded} metrics for exp {experiment_id}")
    if found_final:
        update_experiment_status(experiment_id, "completed")
    return recorded, found_final


@dataclass
class SessionState:
    status: str = "disconnected"
    tracking_experiment_id: Optional[int] = None
    tracking_run_id: Optional[str] = None
    metrics_synced: int = 0
    last_error: Optional[str] = None

    def save(self, session_name: str, _cache={}):
        data = json.dumps(asdict(self), indent=2)
        if _cache.get(session_name) == data:
            return
        PATHS.state_file(session_name).write_text(data)
        _cache[session_name] = data

    @classmethod
    def load(cls, session_name: str) -> "SessionState":
        data = safe_json_load(PATHS.state_file(session_name), default={})
        if data:
            known = {f.name for f in cls.__dataclass_fields__.values()}
            try:
                return cls(**{k: v for k, v in data.items() if k in known})
            except TypeError:
                pass
        return cls()


class SessionTracker:
    """Manages connection and tracking for one remote session.

    The connection is treated as disposable: a dropped SSH tunnel or WebSocket
    is reconnected automatically with capped exponential backoff. The local
    daemon passively mirrors remote state, so losing the connection should never
    require manual intervention.
    """

    RECONNECT_BASE_DELAY = 2.0    # first retry after a drop
    RECONNECT_MAX_DELAY = 30.0    # backoff ceiling
    SLEEP_GAP_THRESHOLD = 15.0    # wall-clock jump that implies host sleep/suspend
    DEAD_SESSION_GRACE = 120.0    # unreachable longer than this => machine is gone

    def __init__(self, config: SessionConfig, daemon: "LocalMetricsDaemon"):
        self.config = config
        self.session_name = config.name
        self.daemon = daemon
        self.running = True
        self.rpc: Optional[RpcClient] = None
        self.state = SessionState.load(self.session_name)
        self.current_experiment_id: Optional[int] = None
        self.current_run_id: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._last_queue_key: Optional[str] = None
        self._was_connected: bool = False
        self._outage_started_mono: Optional[float] = None  # monotonic anchor: counts awake retry time only
        self._dead_finalized: bool = False                 # finalized in-flight work for this outage
        self._mappings_offset: int = 0  # legacy mappings.jsonl offset
        self._segment_offsets: dict[str, int] = {}  # per-segment byte offset

    def start(self):
        self._thread = threading.Thread(
            target=self._session_loop, name=f"session-{self.session_name}", daemon=True,
        )
        self._thread.start()

    def stop(self):
        self.running = False
        if self.rpc:
            try: self.rpc.close()
            except Exception: pass
        if self._thread:
            self._thread.join(timeout=10)

    def event(self, message: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [{self.session_name}] {message}"
        print(line, flush=True)
        try:
            with PATHS.events_file(self.session_name).open("a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _adopt_running_experiment(self):
        """If state.json lost tracking info but the DB still has a running experiment
        for this session, adopt it so _finalize_dead_session can mark it failed."""
        if self.current_experiment_id:
            return
        try:
            conn = get_db()
            row = conn.execute(
                "SELECT id, remote_run_id FROM experiments "
                "WHERE status = 'running' AND session_name = ? "
                "ORDER BY id DESC LIMIT 1",
                (self.session_name,),
            ).fetchone()
            conn.close()
            if row:
                self.current_experiment_id = row["id"]
                self.current_run_id = row["remote_run_id"]
                self.event(f"Adopted running experiment {row['id']} from DB")
        except Exception as e:
            log.warning(f"[{self.session_name}] Failed to adopt running experiment: {e}")

    def _is_paused(self) -> bool:
        """Re-read the session's sync_paused flag from disk (cheap small JSON).

        Read live rather than cached so a `nanorun local pause` / dashboard toggle
        takes effect without restarting the daemon."""
        sc = Config.load_session(self.session_name)
        return bool(sc and getattr(sc, "sync_paused", False))

    def _session_loop(self):
        log.info(f"[{self.session_name}] Tracker starting")
        self.event("Session tracker starting")
        self._adopt_running_experiment()
        backoff = self.RECONNECT_BASE_DELAY
        was_paused = False
        while self.running:
            # Paused: drop the connection and idle. The thread stays alive (so the
            # discovery loop won't restart it and it isn't flagged "reconnectable"),
            # but we hold no SSH tunnel and do no scanning. We never enter the
            # disconnect/backoff/dead-session path while paused, so a pause can't be
            # mistaken for a dead machine. Resume reconnects and reconciles.
            if self._is_paused():
                if not was_paused:
                    self.event("Sync paused — tracker idle, connection dropped")
                    was_paused = True
                    if self.rpc:
                        try: self.rpc.close()
                        except Exception: pass
                        self.rpc = None
                    kill_tunnel(self.session_name)
                    self._set_cache_connected(False)
                    self._outage_started_mono = None
                    self._dead_finalized = False
                    backoff = self.RECONNECT_BASE_DELAY
                self.state.status = "paused"
                self.state.save(self.session_name)
                self._sleep_interruptible(2.0)
                continue
            if was_paused:
                self.event("Sync resumed — reconnecting tracker")
                was_paused = False
            try:
                self._connect()
                if not self.rpc:
                    raise ConnectionError("Connection failed")
                self.state.status = "connected"
                self._was_connected = True
                self.state.last_error = None
                self.state.save(self.session_name)
                self.event("Connected to remote daemon")
                self._set_cache_connected(True)
                self._initial_sync()
                # Healthy link: clear outage tracking and reset backoff.
                self._outage_started_mono = None
                self._dead_finalized = False
                backoff = self.RECONNECT_BASE_DELAY
                self._event_loop()  # returns on disconnect, sleep, or shutdown
            except Exception as e:
                log.warning(f"[{self.session_name}] Session loop error: {e}")
                self.state.last_error = str(e)
            finally:
                if self.rpc:
                    try: self.rpc.close()
                    except Exception: pass
                    self.rpc = None
                self.state.status = "disconnected"
                self.state.save(self.session_name)

            if not self.running:
                break

            # Anchor the outage on a MONOTONIC clock. time.monotonic() doesn't tick
            # while macOS is asleep, so grace counts only *awake* time spent actually
            # retrying — the laptop sleeping never counts against the machine. (Using
            # wall time here would declare a healthy machine dead after an overnight
            # sleep, since wall_elapsed would blow past the grace window instantly.)
            if self._outage_started_mono is None:
                self._outage_started_mono = time.monotonic()

            # Mark the cached queue stale (kept, not cleared) and force a fresh
            # tunnel — a half-dead tunnel keeps a live local listener, so reusing
            # it would loop forever. Then back off and reconnect.
            self._on_disconnect()
            kill_tunnel(self.session_name)

            # Unreachable past the grace window (of awake retry time) => treat the
            # machine as gone and finalize in-flight work. Self-healing: if the
            # machine comes back, reconnect re-attaches via _reconcile_tracking and
            # metrics sync restores the true status (failed -> running/completed).
            outage = time.monotonic() - self._outage_started_mono
            if outage > self.DEAD_SESSION_GRACE and not self._dead_finalized:
                self._finalize_dead_session(outage)
                self._dead_finalized = True

            self.event(f"Disconnected — reconnecting in {backoff:.0f}s")
            suspended = self._sleep_interruptible(backoff)
            if suspended > self.SLEEP_GAP_THRESHOLD:
                # The host suspended during backoff. Prior reachability evidence is
                # now stale (the machine may have changed state over a long sleep),
                # so restart the grace window — re-verify before declaring it dead.
                self.event(f"Host suspended ~{suspended:.0f}s during backoff — restarting grace window")
                self._outage_started_mono = time.monotonic()
                self._dead_finalized = False
                backoff = self.RECONNECT_BASE_DELAY
            else:
                backoff = min(backoff * 2, self.RECONNECT_MAX_DELAY)
        close_db()  # release this thread's DB connection (final WAL checkpoint)
        log.info(f"[{self.session_name}] Tracker stopped")

    def _sleep_interruptible(self, seconds: float) -> float:
        """Sleep up to *seconds* (monotonic), waking early if the tracker is
        stopping. Returns the wall time apparently spent suspended during the
        sleep — i.e. wall_elapsed minus monotonic_elapsed, which on macOS is the
        duration the host was asleep."""
        start_mono = time.monotonic()
        start_wall = time.time()
        deadline = start_mono + seconds
        while self.running and time.monotonic() < deadline:
            time.sleep(min(0.2, deadline - time.monotonic()))
        return max(0.0, (time.time() - start_wall) - (time.monotonic() - start_mono))

    def _connect(self):
        self.state.status = "connecting"
        self.state.save(self.session_name)
        try:
            self.rpc = RpcClient(self.config)
            self.rpc.connect(timeout=15)
            if not self.rpc.call(Method.PING, timeout=5).get("pong"):
                raise ConnectionError("Ping failed")
        except Exception as e:
            log.warning(f"[{self.session_name}] Connect failed: {e}")
            if self.rpc:
                try: self.rpc.close()
                except Exception: pass
            self.rpc = None

    def _event_loop(self):
        SYNC_INTERVAL = 3.0
        last_mono = time.monotonic()
        last_wall = time.time()
        next_sync = last_mono + SYNC_INTERVAL
        while self.running and self.rpc:
            # Detect host sleep/suspend cheaply (macOS): time.monotonic() does not
            # advance while the machine is asleep, but the wall clock does. So when
            # the loop unblocks, (wall_delta - mono_delta) ~= the time spent asleep.
            # A busy-but-awake iteration advances both clocks together (divergence
            # ~0), so this fires *only* on suspend — not on a merely slow loop. On
            # wake the socket is almost certainly dead, so reconnect now instead of
            # waiting ~45s for SSH keepalive (ServerAliveInterval=15 x 3) to notice.
            now_mono = time.monotonic()
            now_wall = time.time()
            slept = (now_wall - last_wall) - (now_mono - last_mono)
            if slept > self.SLEEP_GAP_THRESHOLD:
                self.event(f"Detected ~{slept:.0f}s host sleep/suspend — reconnecting")
                return
            last_mono, last_wall = now_mono, now_wall
            try:
                ev = self.rpc.recv_event(timeout=1)
                if ev:
                    self._handle_event(ev)
                now = time.monotonic()
                if now >= next_sync:
                    next_sync = now + SYNC_INTERVAL
                    self._sync_logs_and_metrics()
            except ConnectionError:
                log.warning(f"[{self.session_name}] WebSocket disconnected")
                return
            except Exception as e:
                log.exception(f"[{self.session_name}] Event loop error: {e}")
                if not self.running:
                    return

    # --- events ---

    def _handle_event(self, ev: EventMessage):
        log.debug(f"[{self.session_name}] Event: {ev.event.value}")
        data = ev.data or {}
        if ev.event == Event.EXPERIMENT_STARTED:
            self._on_experiment_started(data)
        elif ev.event == Event.EXPERIMENT_FINISHED:
            self._reconcile_tracking(None, None)
        elif ev.event == Event.EXPERIMENT_FAILED:
            self._on_experiment_failed(data)
        elif ev.event == Event.EXPERIMENT_RUN_ID:
            self._on_run_id(data)
        elif ev.event == Event.QUEUE_CHANGED:
            self._sync_queue_cache(data.get("queue", []), ts=data.get("ts"))
        elif ev.event == Event.HUB_SYNC_FAILED:
            self.event(f"WARNING: Hub log sync failing on remote — run 'nanorun daemon restart --session {self.session_name}' to fix")
        self.state.save(self.session_name)

    def _on_experiment_started(self, data: dict):
        exp_id = data.get("experiment_id")
        if exp_id:
            self._ensure_experiment(exp_id, data)
            self._reconcile_tracking(exp_id, data.get("run_id"))

    def _on_experiment_failed(self, data: dict):
        exp_id = data.get("experiment_id")
        if self.current_experiment_id:
            self._finalize_experiment("failed")
        if exp_id:
            record_crash(self.session_name, exp_id, get_crash_log(exp_id) or data.get("crash_log"))

    def _on_run_id(self, data: dict):
        self._reconcile_tracking(data.get("experiment_id"), data.get("run_id"))
        meta = {}
        if data.get("code_hash"):
            meta["code_hash"] = data["code_hash"]
        if data.get("tmux_window"):
            meta["tmux_window"] = data["tmux_window"]
        if meta and self.current_experiment_id:
            update_experiment_metadata(self.current_experiment_id, **meta)

    # --- core reconciliation ---

    def _reconcile_tracking(self, remote_exp_id: Optional[int], remote_run_id: Optional[str]):
        """Compare remote active state with local tracking and fix drift."""
        if remote_exp_id != self.current_experiment_id:
            if self.current_experiment_id:
                self._finalize_experiment("completed")
            if remote_exp_id:
                self._start_tracking(remote_exp_id, remote_run_id)
        elif remote_exp_id and not self.current_run_id and remote_run_id:
            self.current_run_id = remote_run_id
            self.state.tracking_run_id = remote_run_id
            update_experiment_metadata(remote_exp_id, remote_run_id=remote_run_id)
            self.event(f"Attached run log: {remote_run_id}.txt")

    def _start_tracking(self, exp_id: int, run_id: Optional[str]):
        exp = get_experiment(exp_id)
        if not exp:
            # Experiment not in DB yet — fetch mapping from remote and create it
            if self.rpc:
                try:
                    r = self.rpc.call(Method.GET_MAPPING, experiment_id=exp_id, timeout=10)
                    if r.get("success") and r.get("mapping"):
                        self._create_experiment(exp_id, r["mapping"])
                        exp = get_experiment(exp_id)
                except Exception as e:
                    log.warning(f"[{self.session_name}] Failed to fetch mapping for {exp_id}: {e}")
            if not exp:
                # Still not available — process local mappings as fallback
                self._process_mappings_file()
                exp = get_experiment(exp_id)
        name = Path(exp.script).name if exp and getattr(exp, "script", None) else None
        self.event(f"Tracking experiment {exp_id}" + (f" ({name})" if name else ""))
        conn = get_db()
        conn.execute(
            "UPDATE experiments SET status = 'queued' WHERE status = 'running' AND id != ? AND session_name = ?",
            (exp_id, self.session_name),
        )
        conn.commit(); conn.close()
        update_experiment_status(exp_id, "running")
        if run_id:
            update_experiment_metadata(exp_id, remote_run_id=run_id)
        self.current_experiment_id = exp_id
        self.current_run_id = run_id
        self.state.tracking_experiment_id = exp_id
        self.state.tracking_run_id = run_id
        if not run_id:
            self.event(f"Experiment {exp_id} started (waiting for run_id)")

    def _finalize_experiment(self, final_status: str = "completed"):
        if not self.current_experiment_id:
            return
        exp_id = self.current_experiment_id
        self.event(f"Experiment {exp_id} {final_status}")
        exp = get_experiment(exp_id)
        if exp and exp.status in ("running", "queued", "cancelled"):
            update_experiment_status(exp_id, final_status)
        # Fetch output log from remote
        if exp and self.rpc and not get_crash_log(exp_id):
            try:
                r = self.rpc.call(Method.GET_CRASH_LOG, experiment_id=exp_id, timeout=10)
                if r.get("success") and r.get("content"):
                    set_crash_log(exp_id, r["content"])
            except Exception:
                pass
        self.current_experiment_id = None
        self.current_run_id = None
        self.state.tracking_experiment_id = None
        self.state.tracking_run_id = None

    # --- sync ---

    def _initial_sync(self):
        if not self.rpc:
            return
        self.event("Running initial sync...")
        try:
            status = self.rpc.call(Method.STATUS, timeout=10)
            self._reconcile_tracking(
                status.get("current_experiment_id"), status.get("current_run_id"),
            )
            if status.get("queue") is not None:
                self._sync_queue_cache(status["queue"], ts=status.get("ts"))
            # Mappings + metrics are synced by the 3s _sync_logs_and_metrics loop
            self.event("Initial sync: connected")
            self.state.save(self.session_name)
        except Exception as e:
            self.event(f"Initial sync failed: {e}")
            log.exception(f"[{self.session_name}] Initial sync error")

    def _on_disconnect(self):
        """Handle a dropped connection without destroying state.

        Because we reconnect automatically, a drop is usually transient (host
        sleep, network blip) while the remote daemon keeps running the queue.
        Mutating local experiment/queue status here would be a lie that flickers
        on every reconnect, so we don't: experiment statuses are left alone and
        reconciled on reconnect via _initial_sync -> _reconcile_tracking. We only
        flag the cached queue as stale so `nanorun job queue` can say so.
        """
        self.event("Connection lost to remote daemon — will auto-reconnect")
        self._set_cache_connected(False)
        # Force a clean re-sync of the queue cache on reconnect even if the
        # remote queue is byte-identical to what we last saw.
        self._last_queue_key = None

    def _set_cache_connected(self, connected: bool):
        """Flip the queue cache's connected flag in place, preserving the
        last-known queue and its synced_at so the data stays stale-but-honest.

        Guarded by the module-level queue-cache lock so it can't race the
        precedence writer. Inlines its own read-modify-write — it must NOT call
        the precedence helper while holding the lock (that would deadlock)."""
        with _queue_cache_lock:
            path = PATHS.queue_cache_file(self.session_name)
            data = safe_json_load(path, default=None) or {"synced_at": None, "queue": []}
            data["connected"] = connected
            try:
                path.write_text(json.dumps(data, indent=2))
            except OSError:
                pass

    def _finalize_dead_session(self, outage: float):
        """The machine has been unreachable past the grace window — presume it's
        gone and move in-flight work to a terminal state. The in-flight experiment
        was training and got killed by infra (-> failed); queued experiments never
        ran (-> cancelled). Tracking state is cleared so a later reconnect can
        re-attach and self-heal if the machine actually returns.
        """
        note = (f"Machine disconnected — session '{self.session_name}' unreachable across "
                f"{outage:.0f}s of reconnect attempts; experiment did not complete.")
        # In-flight training run got killed by infra (-> failed); queued
        # experiments never ran (-> cancelled). Shared with the session-removal
        # path (see tracker.terminate_session_experiments).
        failed_ids, cancelled_ids = terminate_session_experiments(
            self.session_name,
            running_status="failed",
            queued_status="cancelled",
            note=note,
        )
        if failed_ids:
            self.event(f"Marked {len(failed_ids)} experiment(s) failed — machine gone "
                       f"({outage:.0f}s of awake retries)")
        if cancelled_ids:
            self.event(f"Marked {len(cancelled_ids)} queued experiments cancelled — machine gone")

        # Clear tracking so reconnect re-attaches cleanly (and self-heals).
        self.current_experiment_id = None
        self.current_run_id = None
        self.state.tracking_experiment_id = None
        self.state.tracking_run_id = None
        self.state.save(self.session_name)

    def _sync_logs_and_metrics(self):
        """Process local log files (downloaded by HubSyncer) and reparse metrics."""
        # Process mappings.jsonl — discover new experiments, update statuses
        self._process_mappings_file()

        # Build set of known run_ids
        conn = get_db()
        rows = conn.execute(
            "SELECT id, remote_run_id FROM experiments WHERE remote_run_id IS NOT NULL AND session_name = ?",
            (self.session_name,),
        ).fetchall()
        conn.close()
        known = {row["remote_run_id"]: row["id"] for row in rows}

        # Reparse known experiments whose log files have new bytes
        session_logs = PATHS.logs_dir / self.session_name
        for run_id, exp_id in known.items():
            local_log = session_logs / f"{run_id}.txt"
            if not local_log.exists():
                # Fallback: pre-migration flat dir
                flat = PATHS.logs_dir / f"{run_id}.txt"
                if flat.exists():
                    local_log = flat
                else:
                    continue
            if local_log.stat().st_size <= _log_offsets.get(run_id, 0):
                continue
            recorded, _ = _parse_local_metrics(exp_id, run_id, local_log)
            if recorded:
                self.state.metrics_synced += recorded
                self.event(f"Synced {recorded} new metric steps for experiment {exp_id}")

        # Fetch output logs for experiments missing one
        if self.rpc:
            conn = get_db()
            missing_output = conn.execute(
                "SELECT id FROM experiments WHERE id NOT IN (SELECT experiment_id FROM crash_logs) "
                "AND status IN ('completed', 'failed') "
                "AND session_name = ? ORDER BY id DESC LIMIT 5",
                (self.session_name,),
            ).fetchall()
            for row in missing_output:
                try:
                    r = self.rpc.call(Method.GET_CRASH_LOG, experiment_id=row["id"], timeout=10)
                    if r.get("success") and r.get("content"):
                        set_crash_log(row["id"], r["content"])
                except Exception:
                    pass

    def _process_mappings_file(self):
        """Read mappings from session-specific log directory.

        Segments live under logs/{session}/mappings/mappings-NNNNNN.jsonl (sealed once full,
        so xet sync doesn't choke on a growing tail). Legacy file is still processed
        so older remote daemons' data isn't lost.
        """
        session_logs = PATHS.logs_dir / self.session_name

        # Legacy flat file — kept for dual-read until all remotes are on segments
        legacy = session_logs / "mappings.jsonl"
        if legacy.exists():
            size = legacy.stat().st_size
            if size > self._mappings_offset:
                with open(legacy) as f:
                    f.seek(self._mappings_offset)
                    new_content = f.read()
                    self._mappings_offset = f.tell()
                self._ingest_mapping_lines(new_content)

        # Segmented files — process in index order
        segments_dir = session_logs / "mappings"
        if segments_dir.exists():
            for path in sorted(segments_dir.glob("mappings-*.jsonl")):
                size = path.stat().st_size
                offset = self._segment_offsets.get(path.name, 0)
                if size <= offset:
                    continue
                with open(path) as f:
                    f.seek(offset)
                    new_content = f.read()
                    self._segment_offsets[path.name] = f.tell()
                self._ingest_mapping_lines(new_content)

    def _ingest_mapping_lines(self, content: str):
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                mapping = json.loads(line)
            except json.JSONDecodeError:
                continue
            exp_id = mapping.get("experiment_id")
            if not exp_id:
                continue
            existing = get_experiment(exp_id)
            if not existing:
                self._create_experiment(exp_id, mapping)
            else:
                # Update status if remote is more authoritative — but never
                # downgrade "completed" to "failed" (local metric parsing is
                # more reliable than remote's tail-based classification).
                remote_status = mapping.get("status")
                if remote_status and remote_status != existing.status:
                    if existing.status == "completed" and remote_status == "failed":
                        pass
                    elif remote_status in ("completed", "failed") or existing.status in ("running", "queued"):
                        update_experiment_status(exp_id, remote_status)
                # Backfill run_id if we didn't have it
                if mapping.get("run_id") and not existing.remote_run_id:
                    update_experiment_metadata(exp_id, remote_run_id=mapping["run_id"])

    def _sync_queue_cache(self, queue_items, ts=None):
        # Dedup on queue CONTENT only (not ts): a byte-identical queue is a no-op.
        queue_key = json.dumps(queue_items, sort_keys=True)
        if self._last_queue_key == queue_key:
            return
        self._last_queue_key = queue_key
        ts = ts or datetime.now(timezone.utc).isoformat()
        _write_queue_cache_with_precedence(self.session_name, ts, queue_items, connected=True)
        _append_queue_history(self.session_name, ts, queue_items)

    # --- experiment helpers ---

    def _ensure_experiment(self, exp_id: int, data: dict):
        exp = get_experiment(exp_id)
        if not exp:
            self._create_experiment(exp_id, data)
        elif exp.session_name != self.session_name:
            update_experiment_metadata(exp_id, session_name=self.session_name)
            log.info(f"[{self.session_name}] Claimed experiment {exp_id} from '{exp.session_name}'")

    def _create_experiment(self, exp_id: int, data: dict):
        if get_experiment(exp_id):
            return
        script = data.get("script", "unknown")
        name = data.get("name") or Path(script).stem
        track = data.get("track") or infer_track_from_path(script)
        try:
            create_experiment_from_mapping(
                experiment_id=exp_id, name=name, script=script,
                status=data.get("status", "running"), track=track,
                code_hash=data.get("code_hash"), remote_run_id=data.get("run_id"),
                tmux_window=data.get("tmux_window"), started_at=data.get("started_at"),
                finished_at=data.get("finished_at"), env_vars=data.get("env_vars"),
                gpus=data.get("gpus", 1), gpu_type=data.get("gpu_type", "H100"),
                git_commit=data.get("git_commit"), parent_hash=data.get("parent_hash"),
                kernels_path=data.get("kernels_path"), session_name=self.session_name,
            )
            self.event(f"Created experiment {exp_id} ({Path(script).name})")
        except Exception as e:
            log.error(f"[{self.session_name}] Failed to create experiment {exp_id}: {e}")


class LocalMetricsDaemon:
    def __init__(self, dashboard_port: int = 8080, no_dashboard: bool = False):
        self.running = True
        self.trackers: Dict[str, SessionTracker] = {}
        self.hub_syncer: Optional[HubSyncer] = None
        self.interactive = False
        self.dashboard_port = dashboard_port
        self.no_dashboard = no_dashboard

    def _discover_sessions(self) -> Dict[str, SessionConfig]:
        return {s.name: s for s in Config.list_sessions()}

    # --- session trackers (SSH/RPC) ---

    def _start_tracker(self, config: SessionConfig):
        from .session_connector import get_connector
        connector = get_connector(config.name)
        if not connector.needs_session_tracker:
            self.trackers[config.name] = None  # sentinel: served by HubSyncer
            return
        tracker = SessionTracker(config, self)
        self.trackers[config.name] = tracker
        tracker.start()
        log.info(f"Started tracker for '{config.name}'")

    def _stop_tracker(self, name: str):
        tracker = self.trackers.pop(name, None)
        if tracker is not None and isinstance(tracker, SessionTracker):
            tracker.stop()
            log.info(f"Stopped tracker for '{name}'")

    # --- hub syncer (HTTPS, single instance) ---

    def _start_hub_syncer(self):
        self.hub_syncer = HubSyncer(self)
        self.hub_syncer.start()
        log.info("Started hub syncer")

    def _stop_hub_syncer(self):
        if self.hub_syncer:
            self.hub_syncer.stop()
            self.hub_syncer = None
            log.info("Stopped hub syncer")

    # --- dashboard ---

    def _start_dashboard(self):
        """Start the dashboard server in a daemon thread."""
        import uvicorn
        from .dashboard.app import app

        app.state.daemon = self

        config = uvicorn.Config(
            app, host="0.0.0.0", port=self.dashboard_port, log_level="warning"
        )
        server = uvicorn.Server(config)

        def _run_dashboard():
            try:
                server.run()
            except Exception as e:
                log.error(f"Dashboard crashed: {e}")
                print(f"WARNING: Dashboard failed to start: {e}", file=sys.stderr)

        thread = threading.Thread(target=_run_dashboard, name="dashboard", daemon=True)
        thread.start()
        log.info(f"Dashboard started on port {self.dashboard_port}")
        print(f"\033[36mDashboard: http://localhost:{self.dashboard_port}\033[0m")
        import webbrowser
        webbrowser.open(f"http://localhost:{self.dashboard_port}")

    # --- reconnect logic ---

    def _reconnectable_trackers(self) -> list[str]:
        """Dead trackers that were previously connected (not failed on first connect)."""
        return [name for name, t in self.trackers.items()
                if isinstance(t, SessionTracker) and t._thread and not t._thread.is_alive() and t._was_connected]

    def _hub_reconnectable(self) -> bool:
        return self.hub_syncer is not None and self.hub_syncer.reconnectable

    def reconnect_session(self, name: str) -> bool:
        """Restart a tracker for the given session."""
        current = self._discover_sessions()
        if name not in current:
            return False
        tracker = self.trackers.get(name)
        if tracker and tracker._thread and tracker._thread.is_alive():
            return False
        self._stop_tracker(name)
        self._start_tracker(current[name])
        return True

    def reconnect_hub(self) -> bool:
        """Restart the hub syncer."""
        if self.hub_syncer and self.hub_syncer.is_alive:
            return False
        self._stop_hub_syncer()
        self._start_hub_syncer()
        return True

    def _check_reconnect(self):
        """In interactive mode, detect dead connections and prompt for reconnect."""
        reconnectable_sessions = self._reconnectable_trackers()
        hub_dead = self._hub_reconnectable()
        if not reconnectable_sessions and not hub_dead:
            self._reconnect_prompted = False
            return
        if not getattr(self, '_reconnect_prompted', False):
            parts = []
            if reconnectable_sessions:
                parts.append(f"session(s): {', '.join(reconnectable_sessions)}")
            if hub_dead:
                parts.append("hub")
            print(f"\nDisconnected — {'; '.join(parts)}")
            print("Press Enter to reconnect, or Ctrl+C to quit...", flush=True)
            self._reconnect_prompted = True
        import select
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return
        sys.stdin.readline()
        self._reconnect_prompted = False
        current = self._discover_sessions()
        for name in reconnectable_sessions:
            self._stop_tracker(name)
            if name in current:
                print(f"Reconnecting session {name}...")
                self._start_tracker(current[name])
        if hub_dead:
            print("Reconnecting hub...")
            self.reconnect_hub()

    # --- main loop ---

    def run(self):
        log.info("=" * 50)
        log.info("Local daemon starting")
        self.interactive = sys.stdin.isatty()

        def handle_signal(signum, frame):
            log.info(f"Received signal {signum}")
            self.running = False

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        try:
            PATHS.pid_file.write_text(str(os.getpid()))
            sessions = self._discover_sessions()
            if not sessions:
                print("No sessions configured. Run 'nanorun session start' first.", file=sys.stderr)
                return
            print(f"Starting daemon with {len(sessions)} session(s): {', '.join(sessions.keys())}")
            if not self.no_dashboard:
                self._start_dashboard()
            self._start_hub_syncer()
            for config in sessions.values():
                self._start_tracker(config)
            tick = 0
            while self.running:
                time.sleep(1)
                tick += 1
                if tick % 10 == 0:
                    current = self._discover_sessions()
                    for name in set(current) - set(self.trackers):
                        self._start_tracker(current[name])
                    for name in set(self.trackers) - set(current):
                        self._stop_tracker(name)
                if self.interactive and tick % 5 == 0:
                    self._check_reconnect()
        except KeyboardInterrupt:
            self.running = False
        finally:
            for name in list(self.trackers):
                self._stop_tracker(name)
            self._stop_hub_syncer()
            close_db()  # release the main thread's DB connection
            remove_pid_file()
            log.info("Daemon stopped")


def remove_pid_file():
    if PATHS.pid_file.exists():
        PATHS.pid_file.unlink()

def get_daemon_pid() -> Optional[int]:
    if PATHS.pid_file.exists():
        try:
            pid = int(PATHS.pid_file.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError):
            PATHS.pid_file.unlink(missing_ok=True)
    return None

def is_daemon_running() -> bool:
    return get_daemon_pid() is not None

def stop_daemon() -> bool:
    pid = get_daemon_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(10):
                time.sleep(0.2)
                if not is_daemon_running():
                    return True
            os.kill(pid, signal.SIGKILL)
            remove_pid_file()
            return True
        except ProcessLookupError:
            remove_pid_file()
    return False

def restart_daemon(skip_stop: bool = False) -> Optional[int]:
    if not skip_stop and is_daemon_running():
        stop_daemon()
        time.sleep(0.5)
    subprocess.Popen(
        [sys.executable, "-m", "nanorun.local_daemon"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(10):
        time.sleep(0.3)
        pid = get_daemon_pid()
        if pid:
            return pid
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--dashboard-port", type=int, default=8080)
    args = parser.parse_args()

    if is_daemon_running():
        print("Daemon is already running", file=sys.stderr)
        sys.exit(1)
    LocalMetricsDaemon(
        dashboard_port=args.dashboard_port, no_dashboard=args.no_dashboard
    ).run()

if __name__ == "__main__":
    main()
