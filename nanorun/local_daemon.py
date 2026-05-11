"""Local metrics daemon - event-driven, multi-session.

One listener thread per session (WebSocket events + periodic metric sync).
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
from .rpc_client import RpcClient, RpcError
from .rpc_types import Event, EventMessage, Method
from .tracker import (
    get_db, parse_metric_line, record_metric,
    update_experiment_status, update_experiment_metadata,
    get_experiment, create_experiment_from_mapping,
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


def _sync_all_logs(session_name: str):
    from . import hub
    hub.sync_logs_down(PATHS.logs_dir, session_name)


_log_offsets: dict[str, int] = {}

def _parse_local_metrics(experiment_id: int, run_id: str) -> tuple[int, bool]:
    """Parse metrics from local log file. Marks experiment completed if final step found.
    Returns (count, found_final)."""
    local_log = PATHS.logs_dir / f"{run_id}.txt"
    if not local_log.exists():
        return 0, False
    file_size = local_log.stat().st_size
    offset = _log_offsets.get(run_id, 0)
    if file_size <= offset:
        return 0, False

    with open(local_log) as f:
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
    """Manages connection and tracking for one remote session."""

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

    def _session_loop(self):
        log.info(f"[{self.session_name}] Tracker starting")
        self.event("Session tracker starting")
        try:
            self._connect()
            if not self.rpc:
                self.state.status = "disconnected"
                self.state.last_error = "Connection failed"
                self.state.save(self.session_name)
                self.event("Connection failed — session marked disconnected")
                return
            self.state.status = "connected"
            self._was_connected = True
            self.state.save(self.session_name)
            self.event("Connected to remote daemon")
            self._initial_sync()
            self._event_loop()
        except Exception as e:
            log.exception(f"[{self.session_name}] Session loop error: {e}")
            self.state.last_error = str(e)
        finally:
            if self.rpc:
                try: self.rpc.close()
                except Exception: pass
                self.rpc = None
            self.state.status = "disconnected"
            self.state.save(self.session_name)
            if self.running:
                self.event("Disconnected — run 'nanorun session start' to reconnect")
            log.info(f"[{self.session_name}] Tracker stopped")

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
        next_sync = time.monotonic() + SYNC_INTERVAL
        while self.running and self.rpc:
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
                self._handle_disconnect()
                break
            except Exception as e:
                log.exception(f"[{self.session_name}] Event loop error: {e}")
                if not self.running:
                    break

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
            self._sync_queue_cache(data.get("queue", []))
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
            exp = get_experiment(exp_id)
            record_crash(self.session_name, exp_id, exp.crash_log if exp else data.get("crash_log"))

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
        if exp and not exp.crash_log and self.rpc:
            try:
                r = self.rpc.call(Method.GET_CRASH_LOG, experiment_id=exp_id, timeout=10)
                if r.get("success") and r.get("content"):
                    update_experiment_metadata(exp_id, crash_log=r["content"])
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
                self._sync_queue_cache(status["queue"])
            # Mappings + metrics are synced by the 3s _sync_logs_and_metrics loop
            self.event("Initial sync: connected")
            self.state.save(self.session_name)
        except Exception as e:
            self.event(f"Initial sync failed: {e}")
            log.exception(f"[{self.session_name}] Initial sync error")

    def _handle_disconnect(self):
        self.event("Connection lost to remote daemon")

        # Mark running experiment as cancelled
        if self.current_experiment_id:
            exp = get_experiment(self.current_experiment_id)
            if exp and exp.status == "running":
                update_experiment_status(self.current_experiment_id, "cancelled")
                self.event(f"Marked experiment {self.current_experiment_id} as cancelled (disconnected)")

        # Mark all queued experiments for this session as cancelled
        conn = get_db()
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """UPDATE experiments SET status = 'cancelled', finished_at = COALESCE(finished_at, ?)
               WHERE status = 'queued' AND session_name = ?""",
            (now, self.session_name),
        )
        conn.commit()
        cancelled_queued = cursor.rowcount or 0
        conn.close()
        if cancelled_queued:
            self.event(f"Marked {cancelled_queued} queued experiments as cancelled")

        # Preserve queue cache as backup, then clear the active cache
        cache_file = PATHS.queue_cache_file(self.session_name)
        backup_file = cache_file.with_suffix(".backup.json")
        if cache_file.exists():
            import shutil
            shutil.copy2(cache_file, backup_file)
            # Clear active cache (queue is gone with the session)
            cache_file.write_text(json.dumps({
                "synced_at": datetime.now(timezone.utc).isoformat(), "queue": [],
            }, indent=2))

        # Clear tracking state
        self.current_experiment_id = None
        self.current_run_id = None
        self.state.tracking_experiment_id = None
        self.state.tracking_run_id = None

    def _sync_logs_and_metrics(self):
        """Bulk-download logs + mappings from hub, discover experiments, reparse metrics."""
        try:
            _sync_all_logs(self.session_name)
        except Exception as e:
            log.warning(f"[{self.session_name}] Log sync failed: {e}")
            return

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
        for run_id, exp_id in known.items():
            local_log = PATHS.logs_dir / f"{run_id}.txt"
            if not local_log.exists():
                continue
            if local_log.stat().st_size <= _log_offsets.get(run_id, 0):
                continue
            recorded, _ = _parse_local_metrics(exp_id, run_id)
            if recorded:
                self.state.metrics_synced += recorded
                self.event(f"Synced {recorded} new metric steps for experiment {exp_id}")

        # Fetch output logs for experiments missing one
        if self.rpc:
            conn = get_db()
            missing_output = conn.execute(
                "SELECT id FROM experiments WHERE crash_log IS NULL AND status IN ('completed', 'failed') "
                "AND session_name = ? ORDER BY id DESC LIMIT 5",
                (self.session_name,),
            ).fetchall()
            conn.close()
            for row in missing_output:
                try:
                    r = self.rpc.call(Method.GET_CRASH_LOG, experiment_id=row["id"], timeout=10)
                    if r.get("success") and r.get("content"):
                        update_experiment_metadata(row["id"], crash_log=r["content"])
                except Exception:
                    pass

    def _process_mappings_file(self):
        """Read mappings from hub — dual-read legacy mappings.jsonl and segmented files.

        Segments live under logs/mappings/mappings-NNNNNN.jsonl (sealed once full,
        so xet sync doesn't choke on a growing tail). Legacy file is still processed
        so older remote daemons' data isn't lost.
        """
        # Legacy flat file — kept for dual-read until all remotes are on segments
        legacy = PATHS.logs_dir / "mappings.jsonl"
        if legacy.exists():
            size = legacy.stat().st_size
            if size > self._mappings_offset:
                with open(legacy) as f:
                    f.seek(self._mappings_offset)
                    new_content = f.read()
                    self._mappings_offset = f.tell()
                self._ingest_mapping_lines(new_content)

        # Segmented files — process in index order
        segments_dir = PATHS.logs_dir / "mappings"
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

    def _sync_queue_cache(self, queue_items: list):
        queue_key = json.dumps(queue_items, sort_keys=True)
        if self._last_queue_key == queue_key:
            return
        self._last_queue_key = queue_key
        PATHS.queue_cache_file(self.session_name).write_text(json.dumps({
            "synced_at": datetime.now(timezone.utc).isoformat(), "queue": queue_items,
        }, indent=2))

    # --- experiment helpers ---

    def _ensure_experiment(self, exp_id: int, data: dict):
        if not get_experiment(exp_id):
            self._create_experiment(exp_id, data)

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
    def __init__(self):
        self.running = True
        self.trackers: Dict[str, SessionTracker] = {}
        self.interactive = False

    def _discover_sessions(self) -> Dict[str, SessionConfig]:
        return {s.name: s for s in Config.list_sessions()}

    def _start_tracker(self, config: SessionConfig):
        tracker = SessionTracker(config, self)
        self.trackers[config.name] = tracker
        tracker.start()
        log.info(f"Started tracker for '{config.name}'")

    def _stop_tracker(self, name: str):
        tracker = self.trackers.pop(name, None)
        if tracker:
            tracker.stop()
            log.info(f"Stopped tracker for '{name}'")

    def _reconnectable_trackers(self) -> list[str]:
        """Dead trackers that were previously connected (not failed on first connect)."""
        return [name for name, t in self.trackers.items()
                if t._thread and not t._thread.is_alive() and t._was_connected]

    def _check_reconnect(self):
        """In interactive mode, detect dead trackers and prompt for reconnect."""
        reconnectable = self._reconnectable_trackers()
        if not reconnectable:
            self._reconnect_prompted = False
            return
        if not getattr(self, '_reconnect_prompted', False):
            names = ", ".join(reconnectable)
            print(f"\nSession(s) disconnected: {names}")
            print("Press Enter to reconnect, or Ctrl+C to quit...", flush=True)
            self._reconnect_prompted = True
        import select
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return
        sys.stdin.readline()
        self._reconnect_prompted = False
        current = self._discover_sessions()
        for name in reconnectable:
            self._stop_tracker(name)
            if name in current:
                print(f"Reconnecting {name}...")
                self._start_tracker(current[name])

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
    if is_daemon_running():
        print("Daemon is already running", file=sys.stderr)
        sys.exit(1)
    LocalMetricsDaemon().run()

if __name__ == "__main__":
    main()
