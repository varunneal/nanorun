"""Experiment queue management.

Queue is stored as a plain text file for easy manual editing.
Queue state (active/paused) is stored separately.

In the new daemon architecture:
- The authoritative queue lives on the remote daemon
- Local queue.txt is kept as a mirror/cache for visibility
- Operations like add/remove/clear delegate to daemon when possible
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from .config import Config

if TYPE_CHECKING:
    from .remote_control import DaemonClient


@dataclass
class QueuedExperiment:
    """An experiment waiting in the queue."""
    script: str
    env_vars: Dict[str, str]
    track: Optional[str]
    gpus: int
    gpu_type: str
    name: Optional[str]
    session_name: Optional[str] = None

    def to_line(self) -> str:
        """Serialize to a single line for queue file."""
        env_json = json.dumps(self.env_vars) if self.env_vars else "{}"
        track = self.track or ""
        name = self.name or ""
        return f"{self.script}|{env_json}|{track}|{self.gpus}|{name}|{self.gpu_type}"

    @classmethod
    def from_line(cls, line: str) -> Optional["QueuedExperiment"]:
        """Parse a single line from queue file."""
        line = line.strip()
        if not line or line.startswith("#"):
            return None

        parts = line.split("|")
        if len(parts) < 4:
            return None

        try:
            return cls(
                script=parts[0],
                env_vars=json.loads(parts[1]) if parts[1] else {},
                track=parts[2] if parts[2] else None,
                gpus=int(parts[3]),
                gpu_type=parts[5] if len(parts) > 5 and parts[5] else "H100",
                name=parts[4] if len(parts) > 4 and parts[4] else None,
            )
        except (json.JSONDecodeError, ValueError):
            return None


# =============================================================================
# Queue file operations
# =============================================================================

def get_queue_path() -> Path:
    """Get path to queue file."""
    config_dir = Config.get_config_dir()
    return config_dir / "queue.txt"


def get_queue_backup_path() -> Path:
    """Get path to queue backup file."""
    return get_queue_path().with_name("queue.txt.backup")


def get_queue_state_path() -> Path:
    """Get path to queue state file."""
    config_dir = Config.get_config_dir()
    return config_dir / "queue_state.txt"


def read_queue(session_name: Optional[str] = None) -> List[QueuedExperiment]:
    """Read queued experiments from local cache (synced from remote daemon).

    If session_name is given, reads only that session's queue.
    If None, combines queues from all sessions.
    """
    from .local_daemon import safe_json_load, get_queue_cache_file

    if session_name is not None:
        return _read_session_queue(session_name)

    # Combine all sessions
    sessions = Config.list_sessions()
    experiments = []
    for sc in sessions:
        experiments.extend(_read_session_queue(sc.name))
    return experiments


def _read_session_queue(session_name: str) -> List[QueuedExperiment]:
    from .local_daemon import safe_json_load, get_queue_cache_file

    data = safe_json_load(get_queue_cache_file(session_name), default={})
    queue_items = data.get("queue", []) if data else []

    experiments = []
    for item in queue_items:
        experiments.append(QueuedExperiment(
            script=item.get("script", ""),
            env_vars=item.get("env_vars", {}),
            track=item.get("track"),
            gpus=item.get("gpus", 1),
            gpu_type=item.get("gpu_type", "H100"),
            name=item.get("name"),
            session_name=session_name,
        ))
    return experiments


def _write_queue_file(path: Path, experiments: List[QueuedExperiment]) -> None:
    """Write experiments to a queue-format file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# nanorun experiment queue", "# Format: script|env_vars_json|track|gpus|name", ""]
    for exp in experiments:
        lines.append(exp.to_line())

    path.write_text("\n".join(lines) + "\n")


def write_queue(experiments: List[QueuedExperiment]) -> None:
    """Write experiments to queue file."""
    _write_queue_file(get_queue_path(), experiments)


def write_queue_backup(experiments: List[QueuedExperiment]) -> None:
    """Write experiments to queue backup file."""
    _write_queue_file(get_queue_backup_path(), experiments)


# =============================================================================
# Queue state operations
# =============================================================================

def get_queue_state() -> str:
    """Get queue state: 'active' or 'paused'."""
    state_path = get_queue_state_path()
    if state_path.exists():
        state = state_path.read_text().strip()
        if state in ("active", "paused"):
            return state
    return "active"  # Default to active


def set_queue_state(state: str) -> None:
    """Set queue state: 'active' or 'paused'."""
    if state not in ("active", "paused"):
        raise ValueError(f"Invalid queue state: {state}")

    state_path = get_queue_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(state + "\n")


def is_queue_paused() -> bool:
    """Check if queue processing is paused."""
    return get_queue_state() == "paused"


def is_queue_active() -> bool:
    """Check if queue processing is active."""
    return get_queue_state() == "active"


# =============================================================================
# Daemon-aware queue operations
# =============================================================================

@dataclass
class AddToQueueResult:
    """Result of adding an experiment to queue."""
    success: bool
    position: Optional[int] = None
    daemon_status: Optional[str] = None
    started: bool = False
    error: Optional[str] = None


def add_to_queue_via_daemon(
    experiment_id: int,
    script: str,
    env_vars: Optional[Dict[str, str]] = None,
    track: Optional[str] = None,
    gpus: int = 1,
    gpu_type: str = "H100",
    name: Optional[str] = None,
    cmd_prefix: Optional[str] = None,
    first: bool = False,
    auto_start: bool = False,
    session_name: Optional[str] = None,
) -> AddToQueueResult:
    """Add an experiment to queue via daemon.

    This is the preferred method in the daemon architecture.

    Args:
        experiment_id: Local experiment ID (from tracker.create_experiment)
        script: Script path relative to repo root
        env_vars: Environment variables
        track: Experiment track
        gpus: Number of GPUs
        gpu_type: GPU type (H100, H200, GH200)
        name: Experiment name
        cmd_prefix: Command prefix (e.g., 'nsys profile ...')
        first: If True, add to front of queue instead of end
        auto_start: If True, daemon will auto-start if idle (avoids extra round-trip)

    Returns:
        AddToQueueResult with position, daemon_status, and whether experiment started
    """
    from .remote_control import get_daemon_client, DaemonError

    daemon = get_daemon_client(session_name)
    if not daemon:
        return AddToQueueResult(success=False, error="No active session")

    with daemon:
        try:
            if not daemon.ensure_running():
                return AddToQueueResult(success=False, error="Could not start daemon")

            result = daemon.add_to_queue(
                experiment_id=experiment_id,
                script=script,
                env_vars=env_vars or {},
                gpus=gpus,
                gpu_type=gpu_type,
                name=name,
                track=track,
                cmd_prefix=cmd_prefix,
                first=first,
                auto_start=auto_start,
            )

            if result.get("success"):
                return AddToQueueResult(
                    success=True,
                    position=result.get("position", 1),
                    daemon_status=result.get("daemon_status"),
                    started=result.get("started", False),
                )
            return AddToQueueResult(success=False, error=result.get("error", "Unknown error"))

        except DaemonError as e:
            return AddToQueueResult(success=False, error=str(e))


def clear_queue_via_daemon(session_name: Optional[str] = None) -> Optional[int]:
    """Clear queue via daemon.

    Returns count of items cleared, or None if daemon unreachable.
    """
    from .remote_control import get_daemon_client, DaemonError

    daemon = get_daemon_client(session_name)
    if not daemon:
        return None

    with daemon:
        try:
            if not daemon.ensure_running():
                return None
            status = daemon.get_status()
            queue_len = status.get("queue_length", 0)
            result = daemon.clear_queue()
            return queue_len if result.get("success") else None
        except DaemonError:
            return None


def remove_from_queue_via_daemon(index: int, session_name: Optional[str] = None) -> bool:
    """Remove item from queue via daemon."""
    from .remote_control import get_daemon_client, DaemonError

    daemon = get_daemon_client(session_name)
    if not daemon:
        return False

    with daemon:
        try:
            if not daemon.ensure_running():
                return False
            result = daemon.remove_from_queue(index)
            return result.get("success", False)
        except DaemonError:
            return False


def get_daemon_status(session_name: Optional[str] = None) -> Optional[Dict]:
    """Get daemon status including current experiment and queue info."""
    from .remote_control import get_daemon_client, DaemonError

    daemon = get_daemon_client(session_name)
    if not daemon:
        return None

    with daemon:
        try:
            if not daemon.ensure_running():
                return None
            return daemon.get_status()
        except DaemonError:
            return None


def read_local_queue_file() -> List[QueuedExperiment]:
    """Read queue from local editable file (.nanorun/queue.txt).

    This reads the pipe-delimited format that users can manually edit.
    Different from read_queue() which reads from the daemon-synced cache.
    """
    queue_path = get_queue_path()
    if not queue_path.exists():
        return []

    experiments = []
    for line in queue_path.read_text().strip().split("\n"):
        exp = QueuedExperiment.from_line(line)
        if exp:
            experiments.append(exp)
    return experiments


def push_queue_to_daemon(session_name: Optional[str] = None) -> Optional[int]:
    """Push local queue file to remote daemon, replacing remote queue.

    Reads from .nanorun/queue.txt (the editable local file) and sends
    the contents to the remote daemon. This mimics doing a queue clear
    followed by `job add` for each item - generating experiment IDs so
    the local daemon can create DB entries.

    Returns:
        Number of items pushed, or None if daemon unreachable.
    """
    import time
    import random

    # Read local queue file first
    experiments = read_local_queue_file()

    # Clear remote queue
    cleared = clear_queue_via_daemon(session_name=session_name)
    if cleared is None:
        return None

    # Add each item like job add would
    count = 0
    for exp in experiments:
        # Generate experiment ID (same as job add)
        experiment_id = int(time.time() * 1000) + random.randint(0, 999)

        pos = add_to_queue_via_daemon(
            experiment_id=experiment_id,
            script=exp.script,
            env_vars=exp.env_vars,
            track=exp.track,
            gpus=exp.gpus,
            gpu_type=exp.gpu_type,
            name=exp.name,
            session_name=session_name,
        )
        if pos is not None:
            count += 1

    return count
