"""Session connector abstraction — uniform interface for job operations.

Two implementations:
  - SshConnector: wraps DaemonClient RPC + queue.py + runner.py
  - IrisConnector: shells out to iris CLI
"""

import hashlib
import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config, SessionConfig, get_repo_root, infer_track_from_path
from .hub import _IRIS_STATE_MAP
from .tracker import (
    create_experiment, update_experiment_metadata, update_experiment_status,
    get_db,
)


# =============================================================================
# Result types
# =============================================================================


@dataclass
class SubmitResult:
    experiment_id: Optional[int] = None
    job_id: Optional[str] = None
    error: Optional[str] = None
    started: bool = False
    position: Optional[int] = None


@dataclass
class QueueItem:
    job_id: str
    script: str
    state: str
    env_vars: Dict[str, str] = field(default_factory=dict)
    track: Optional[str] = None
    gpu_type: Optional[str] = None
    gpus: int = 1
    created_at: Optional[str] = None
    session_name: Optional[str] = None


@dataclass
class ConnectorResult:
    success: bool = True
    message: Optional[str] = None
    unsupported: bool = False


@dataclass
class QueueMeta:
    """Freshness/connection metadata for a queue() result.

    `live` is True when queue() returns data fetched on the spot (e.g. Iris),
    so staleness can't apply. For cache-backed connectors, `connected` is False
    when the snapshot is stale and `synced_at` is when it was last fresh.
    """
    connected: bool = True
    synced_at: Optional[str] = None
    live: bool = False

    @property
    def stale(self) -> bool:
        return not self.live and not self.connected


# =============================================================================
# ABC
# =============================================================================


class SessionConnector(ABC):
    """Uniform interface for job operations across session types."""

    needs_session_tracker: bool = True

    @abstractmethod
    def submit(self, script: str, env_vars: Dict[str, str], name: str,
               track: Optional[str], gpus: int = 1, gpu_type: str = "H100",
               prefix: Optional[str] = None, first: bool = False,
               **kwargs) -> SubmitResult: ...

    @abstractmethod
    def queue(self) -> List[QueueItem]: ...

    def queue_meta(self) -> QueueMeta:
        """Freshness of the last queue() result. Default: live (always fresh)."""
        return QueueMeta(live=True)

    @abstractmethod
    def cancel(self, job_id: Optional[str] = None) -> ConnectorResult: ...

    @abstractmethod
    def status(self) -> List[QueueItem]: ...

    @abstractmethod
    def logs(self, job_id: Optional[str] = None, tail: bool = False,
             lines: int = 50) -> ConnectorResult: ...

    @abstractmethod
    def resume(self) -> ConnectorResult: ...

    @abstractmethod
    def clear(self) -> ConnectorResult: ...

    @abstractmethod
    def remove(self, index: int) -> ConnectorResult: ...

    def sweep(self, script: str, configs: List[Dict[str, str]], name: str,
              track: Optional[str], gpus: int = 1, gpu_type: str = "H100",
              prefix: Optional[str] = None, **kwargs) -> List[SubmitResult]:
        """Submit a parameter sweep. Default: call submit() per config."""
        results = []
        for i, cfg in enumerate(configs):
            r = self.submit(
                script, cfg, name, track, gpus=gpus, gpu_type=gpu_type,
                prefix=prefix, first=False, **kwargs,
            )
            results.append(r)
        return results

    def resolve_script(self, script: str) -> Optional[Path]:
        """Resolve script path against the appropriate root. Returns None if not found."""
        return None

    def infer_track(self, script: str) -> Optional[str]:
        """Derive track name from script path when no .track.json exists."""
        return None

    def check_unsynced(self, script: str) -> bool:
        """Return True if script has unsynced local changes. False if not applicable."""
        return False

    def sync_file(self, script: str) -> None:
        """Sync a file to the remote. No-op if not applicable."""
        pass


# =============================================================================
# SSH Connector
# =============================================================================


class SshConnector(SessionConnector):
    """Job operations via remote daemon RPC."""

    def __init__(self, session_name: str):
        self.session_name = session_name

    def submit(self, script: str, env_vars: Dict[str, str], name: str,
               track: Optional[str], gpus: int = 1, gpu_type: str = "H100",
               prefix: Optional[str] = None, first: bool = False,
               **kwargs) -> SubmitResult:
        import time, random
        from .queue import add_to_queue_via_daemon

        auto_start = kwargs.get("auto_start", True)
        experiment_id = int(time.time() * 1000) + random.randint(0, 999)
        result = add_to_queue_via_daemon(
            experiment_id, script, env_vars, track, gpus, gpu_type, name, prefix,
            first=first, auto_start=auto_start, session_name=self.session_name,
        )
        return SubmitResult(
            experiment_id=experiment_id if result.success else None,
            job_id=None,
            error=result.error if not result.success else None,
            started=result.started,
            position=result.position,
        )

    def sweep(self, script: str, configs: List[Dict[str, str]], name: str,
              track: Optional[str], gpus: int = 1, gpu_type: str = "H100",
              prefix: Optional[str] = None, **kwargs) -> List[SubmitResult]:
        results = []
        for i, cfg in enumerate(configs):
            r = self.submit(
                script, cfg, name, track, gpus=gpus, gpu_type=gpu_type,
                prefix=prefix, first=False, auto_start=(i == 0), **kwargs,
            )
            results.append(r)
        return results

    def queue(self) -> List[QueueItem]:
        from .queue import read_queue
        queued = read_queue(self.session_name)
        return [
            QueueItem(
                job_id=str(getattr(e, 'id', '')),
                script=e.script,
                state="queued",
                env_vars=e.env_vars or {},
                track=e.track,
                gpu_type=e.gpu_type,
                gpus=e.gpus,
                session_name=e.session_name,
            )
            for e in queued
        ]

    def queue_meta(self) -> QueueMeta:
        from .queue import read_queue_meta
        m = read_queue_meta(self.session_name)
        return QueueMeta(connected=bool(m["connected"]), synced_at=m["synced_at"], live=False)

    def cancel(self, job_id: Optional[str] = None) -> ConnectorResult:
        from .runner import cancel_experiment
        cancel_experiment(start_next=False, session_name=self.session_name)
        return ConnectorResult(success=True)

    def status(self) -> List[QueueItem]:
        from .tracker import get_running_experiments
        running = get_running_experiments(session_name=self.session_name)
        items = []
        for exp in running:
            items.append(QueueItem(
                job_id=str(exp.id),
                script=exp.script,
                state="running",
                env_vars=exp.env_vars or {},
                track=exp.track,
                gpu_type=exp.gpu_type,
                gpus=exp.gpus,
                session_name=exp.session_name,
            ))
        return items

    def logs(self, job_id: Optional[str] = None, tail: bool = False,
             lines: int = 50) -> ConnectorResult:
        # SSH logs are handled directly in CLI via tmux (require RemoteSession)
        # Return unsupported so CLI falls through to its existing tmux logic
        return ConnectorResult(unsupported=True, message="use_tmux")

    def resume(self) -> ConnectorResult:
        from .runner import resume_queue
        resume_queue(session_name=self.session_name)
        return ConnectorResult(success=True)

    def clear(self) -> ConnectorResult:
        from .queue import clear_queue_via_daemon
        cleared = clear_queue_via_daemon(session_name=self.session_name)
        if cleared is not None:
            return ConnectorResult(success=True, message=f"Cleared {cleared} experiment(s)")
        return ConnectorResult(success=False, message="Daemon not reachable")

    def remove(self, index: int) -> ConnectorResult:
        from .queue import remove_from_queue_via_daemon, read_queue
        queued = read_queue(session_name=self.session_name)
        if index < 1 or index > len(queued):
            return ConnectorResult(success=False, message=f"Invalid index: {index} (queue has {len(queued)} items)")
        exp = queued[index - 1]
        if remove_from_queue_via_daemon(index - 1, session_name=self.session_name):
            return ConnectorResult(success=True, message=f"Removed: {exp.script}")
        return ConnectorResult(success=False, message="Daemon not reachable")

    def resolve_script(self, script: str) -> Optional[Path]:
        p = get_repo_root() / script
        return p if p.exists() else None

    def check_unsynced(self, script: str) -> bool:
        from .sync import has_unsynced_changes
        return has_unsynced_changes(files=[script], session_name=self.session_name)

    def sync_file(self, script: str) -> None:
        from .remote_control import require_session
        from .sync import push_code
        remote = require_session(self.session_name)
        push_code(remote, message=None, files=[script])


# =============================================================================
# Iris Connector
# =============================================================================


_IRIS_AUTO_FORWARD_ENV = ["WANDB_API_KEY"]


class IrisConnector(SessionConnector):
    """Job operations via iris CLI."""

    needs_session_tracker = False

    def __init__(self, session_config: SessionConfig):
        self.sc = session_config
        self.session_name = session_config.name
        self.iris_binary = session_config.iris_binary or "iris"
        self.iris_config = session_config.iris_config
        self.iris_user = session_config.iris_user or ""
        self.iris_workspace = session_config.iris_workspace

    def _iris_cmd(self, *args: str) -> List[str]:
        cmd = [self.iris_binary]
        if self.iris_config:
            cmd.extend(["--config", self.iris_config])
        cmd.extend(args)
        return cmd

    def _run(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
        import os as _os
        cwd = self.iris_workspace or None
        return subprocess.run(
            self._iris_cmd(*args), capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )

    def resolve_script(self, script: str) -> Optional[Path]:
        if self.iris_workspace:
            p = Path(self.iris_workspace) / script
            return p if p.exists() else None
        return None

    def infer_track(self, script: str) -> Optional[str]:
        """Derive track from script path: experiments/grug/moe/launch.py → moe."""
        parts = Path(script).parts
        # Strip leading "experiments/grug/" and use the next component as track
        try:
            idx = list(parts).index("grug")
            if idx + 1 < len(parts) - 1:  # must have something after grug and before filename
                return parts[idx + 1]
        except ValueError:
            pass
        # Fallback: parent directory name
        parent = Path(script).parent.name
        return parent if parent and parent != "." else None

    def submit(self, script: str, env_vars: Dict[str, str], name: str,
               track: Optional[str], gpus: int = 1, gpu_type: str = "TPU",
               prefix: Optional[str] = None, first: bool = False,
               **kwargs) -> SubmitResult:
        import os as _os
        import uuid

        reserve = kwargs.get("reserve")
        module = kwargs.get("module")

        # Resolve script against iris workspace
        workspace = Path(self.iris_workspace) if self.iris_workspace else get_repo_root()
        abs_script = workspace / script
        code_hash = None
        if abs_script.exists():
            code_hash = hashlib.sha256(abs_script.read_bytes()).hexdigest()[:12]

        # Generate unique run ID for W&B tracking
        wandb_run_id = str(uuid.uuid4())

        # Auto-forward env vars (secrets — not persisted to DB)
        auto_env = {}
        for key in _IRIS_AUTO_FORWARD_ENV:
            val = _os.environ.get(key)
            if val:
                auto_env[key] = val
        auto_env["GRUG_RUN_ID"] = wandb_run_id
        cli_env = {**auto_env, **env_vars}

        # Create experiment in SQLite (user env + run ID, no secrets)
        stored_env = {**env_vars, "_wandb_run_id": wandb_run_id}
        exp_id = create_experiment(
            name=name, script=script, track=track, code_hash=code_hash,
            gpu_type="TPU", session_name=self.session_name, env_vars=stored_env,
        )
        update_experiment_status(exp_id, "queued")

        # Build iris command
        cmd_args = ["job", "run", "--no-wait"]
        if reserve:
            cmd_args.extend(["--reserve", reserve])
        region = kwargs.get("region")
        if region:
            for r in region:
                cmd_args.extend(["--region", r])
        for k, v in cli_env.items():
            cmd_args.extend(["-e", k, v])
        cmd_args.append("--")
        if module:
            cmd_args.extend(["python", "-m", module])
        else:
            cmd_args.extend(["python", script])

        try:
            result = self._run(*cmd_args, timeout=120)
        except subprocess.TimeoutExpired:
            return SubmitResult(experiment_id=exp_id, error="iris job run timed out")

        if result.returncode != 0:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            return SubmitResult(experiment_id=exp_id, error=f"iris job run failed: {err}")

        lines = result.stdout.strip().splitlines()
        if not lines:
            return SubmitResult(experiment_id=exp_id, error="No output from iris job run")
        iris_job_id = lines[-1].strip()

        # Use wandb_run_id as canonical remote_run_id (simple, no slashes)
        # Store iris job ID separately for cancel/logs operations
        update_experiment_metadata(exp_id, remote_run_id=wandb_run_id)
        conn = get_db()
        conn.execute(
            "UPDATE experiments SET env_vars = json_set(COALESCE(env_vars, '{}'), '$._iris_job_id', ?) WHERE id = ?",
            (iris_job_id, exp_id),
        )
        conn.commit()
        conn.close()
        self._write_log_header(wandb_run_id, script, abs_script, wandb_run_id, iris_job_id)

        return SubmitResult(experiment_id=exp_id, job_id=iris_job_id)

    def queue(self) -> List[QueueItem]:
        jobs = self._list_jobs_raw()
        items = []
        for j in jobs:
            name = j.get("name", "")
            if not name or ":" in name.split("/")[-1]:
                continue
            state = _IRIS_STATE_MAP.get(j.get("state", ""), "")
            if state not in ("queued", "running"):
                continue
            submitted = j.get("submitted_at", {})
            epoch_ms = submitted.get("epoch_ms")
            created_at = None
            if epoch_ms:
                ts = int(epoch_ms) / 1000
                created_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            items.append(QueueItem(
                job_id=j.get("name", ""),
                script="",
                state=state,
                created_at=created_at,
                session_name=self.session_name,
            ))
        return items

    def cancel(self, job_id: Optional[str] = None) -> ConnectorResult:
        if not job_id:
            # Find most recent running job from iris
            items = self.queue()
            running = [i for i in items if i.state == "running"]
            if not running:
                return ConnectorResult(success=False, message="No running iris jobs to cancel")
            job_id = running[0].job_id

        result = self._run("job", "stop", job_id, timeout=30)
        if result.returncode != 0:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown"
            return ConnectorResult(success=False, message=f"iris job stop failed: {err}")

        # Update local SQLite — find by _iris_job_id in env_vars
        conn = get_db()
        row = conn.execute(
            "SELECT id FROM experiments WHERE session_name = ? AND json_extract(env_vars, '$._iris_job_id') = ?",
            (self.session_name, job_id),
        ).fetchone()
        conn.close()
        if row:
            update_experiment_status(row["id"], "cancelled")

        return ConnectorResult(success=True, message=job_id)

    def status(self) -> List[QueueItem]:
        items = self.queue()
        return [i for i in items if i.state in ("running", "queued")]

    def logs(self, job_id: Optional[str] = None, tail: bool = False,
             lines: int = 50) -> ConnectorResult:
        if not job_id:
            items = self.queue()
            running = [i for i in items if i.state == "running"]
            if not running:
                return ConnectorResult(success=False, message="No running iris jobs. Provide a job ID.")
            job_id = running[0].job_id

        if tail:
            cmd = self._iris_cmd("job", "logs", "-f", job_id)
            subprocess.run(cmd, cwd=self.iris_workspace or None)
            return ConnectorResult(success=True)
        else:
            result = self._run(
                "job", "logs", "--max-lines", str(lines), "--no-tail", job_id,
                timeout=30,
            )
            if result.returncode == 0:
                print(result.stdout)
                return ConnectorResult(success=True)
            return ConnectorResult(success=False, message="Failed to fetch logs")

    def resume(self) -> ConnectorResult:
        return ConnectorResult(unsupported=True, message="Iris sessions manage scheduling automatically.")

    def clear(self) -> ConnectorResult:
        return ConnectorResult(unsupported=True, message="Not supported for iris sessions. Use 'nanorun job cancel' to stop individual jobs.")

    def remove(self, index: int) -> ConnectorResult:
        return ConnectorResult(unsupported=True, message="Not supported for iris sessions. Use 'nanorun job cancel' to stop individual jobs.")

    # --- internal ---

    def _list_jobs_raw(self) -> List[dict]:
        result = self._run("job", "list", "--prefix", f"/{self.iris_user}/", "--json", timeout=30)
        if result.returncode != 0:
            return []
        try:
            return json.loads(result.stdout)
        except (ValueError, json.JSONDecodeError):
            return []

    def _write_log_header(self, run_id: str, script: str, abs_script: Path,
                          wandb_run_id: str, iris_job_id: str):
        config_dir = Config.get_config_dir()
        logs_dir = config_dir / "logs" / self.session_name
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_path = logs_dir / f"{run_id}.txt"

        code_content = abs_script.read_text() if abs_script.exists() else "(script not found)"
        now = datetime.now(timezone.utc).isoformat()

        # Construct wandb URL from session config (first project in comma-separated list)
        wandb_entity = getattr(self.sc, "wandb_entity", "") or ""
        wandb_project = (getattr(self.sc, "wandb_project", "") or "").split(",")[0].strip()
        wandb_url = f"https://wandb.ai/{wandb_entity}/{wandb_project}/runs/{wandb_run_id}" if wandb_entity and wandb_project else ""

        header = (
            f"--- CODE START ---\n"
            f"{code_content}\n"
            f"--- CODE END ---\n\n"
            f"--- METADATA ---\n"
            f"iris_job_id: {iris_job_id}\n"
            f"wandb_run_id: {wandb_run_id}\n"
            f"wandb_url: {wandb_url}\n"
            f"submitted_at: {now}\n"
            f"--- METADATA END ---\n\n"
        )
        log_path.write_text(header)


# =============================================================================
# Factory
# =============================================================================


def get_connector(session_name: str) -> SessionConnector:
    """Get the appropriate connector for a session."""
    sc = Config.load_session(session_name)
    if sc and sc.session_type == "iris":
        return IrisConnector(sc)
    return SshConnector(session_name)
