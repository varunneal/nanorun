"""Hub integration for log and weight storage.

Dispatches to a configured backend (HuggingFace Buckets, S3, etc.)
based on [hub] section in nanorun.toml.

Storage layout (same across all backends):
  logs/{session}/{run_id}.txt              - Training log files
  logs/{session}/mappings/*.jsonl           - Experiment mappings
  weights/{session}/{experiment_id}/       - Model checkpoints
"""

import json
import logging
import math
import os
import warnings
from pathlib import Path
from typing import List, Optional

from .project_config import load_project_config

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message="Cannot enable progress bars", module="huggingface_hub")

log = logging.getLogger("local_daemon")


def _get_hub_config() -> dict:
    return load_project_config().get("hub", {})


def _get_backend() -> str:
    return _get_hub_config().get("backend", "hf")


# =========================================================================
# Backend: HuggingFace Buckets
# =========================================================================


class _HfBackend:
    def __init__(self):
        from huggingface_hub import (
            HfApi, bucket_info, create_bucket, list_bucket_tree, sync_bucket,
        )
        self._HfApi = HfApi
        self._bucket_info = bucket_info
        self._create_bucket = create_bucket
        self._list_bucket_tree = list_bucket_tree
        self._sync_bucket = sync_bucket

        config = _get_hub_config()
        self.bucket_id = config.get("bucket_id", "")
        self.bucket_handle = f"hf://buckets/{self.bucket_id}"
        self._api = HfApi()

    def ping(self) -> bool:
        try:
            info = self._api.whoami()
            return bool(info.get("name"))
        except Exception:
            return False

    def check_auth(self) -> Optional[str]:
        try:
            info = self._api.whoami()
            return info.get("name")
        except Exception:
            return None

    def ensure_bucket(self) -> str:
        self._create_bucket(self.bucket_id, exist_ok=True)
        return self.bucket_handle

    def get_bucket_info(self):
        return self._bucket_info(self.bucket_id)

    def sync_logs_up(self, local_logs_dir: Path, session: str) -> None:
        self._sync_bucket(
            str(local_logs_dir),
            f"{self.bucket_handle}/logs/{session}",
            include=["*.txt", "*.jsonl", "mappings/*.jsonl", "queue/*.jsonl"],
            quiet=True,
        )

    def sync_queue_up(self, local_logs_dir: Path, session: str) -> None:
        # Scoped upload of just the queue snapshot segments (event-driven fast path).
        # Same root/target as sync_logs_up so the remote layout matches; the narrower
        # include means only the queue subtree is diffed.
        self._sync_bucket(
            str(local_logs_dir),
            f"{self.bucket_handle}/logs/{session}",
            include=["queue/*.jsonl"],
            quiet=True,
        )

    def sync_logs_down(self, local_logs_dir: Path, session: str, include: Optional[List[str]] = None) -> None:
        local_logs_dir.mkdir(parents=True, exist_ok=True)
        self._sync_bucket(
            f"{self.bucket_handle}/logs/{session}",
            str(local_logs_dir),
            include=include or ["*.txt", "*.jsonl", "mappings/*.jsonl", "queue/*.jsonl"],
            quiet=True,
        )

    def list_logs(self, session: str) -> List[str]:
        prefix = f"logs/{session}"
        run_ids = []
        for item in self._list_bucket_tree(self.bucket_id, prefix=prefix, recursive=True):
            if item.type == "file" and item.path.endswith(".txt"):
                name = item.path.removeprefix(f"{prefix}/").removesuffix(".txt")
                run_ids.append(name)
        return run_ids

    def upload_log(self, local_path: Path, run_id: str, session: str) -> None:
        prefix = f"logs/{session}"
        self._api.batch_bucket_files(
            self.bucket_id,
            add=[(str(local_path), f"{prefix}/{run_id}.txt")],
        )

    def download_log(self, run_id: str, local_path: Path, session: str) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = f"logs/{session}"
        self._api.download_bucket_files(
            self.bucket_id,
            files=[(f"{prefix}/{run_id}.txt", str(local_path))],
        )

    def list_weights(self, experiment_id: int, session: str) -> List[str]:
        prefix = f"weights/{session}/{experiment_id}"
        filenames = []
        for item in self._list_bucket_tree(self.bucket_id, prefix=prefix, recursive=True):
            if item.type == "file":
                filenames.append(item.path.removeprefix(f"{prefix}/"))
        return filenames

    def upload_weight(self, local_path: Path, experiment_id: int, filename: str, session: str) -> None:
        prefix = f"weights/{session}/{experiment_id}"
        self._api.batch_bucket_files(
            self.bucket_id,
            add=[(str(local_path), f"{prefix}/{filename}")],
        )

    def download_weight(self, experiment_id: int, filename: str, local_path: Path, session: str) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = f"weights/{session}/{experiment_id}"
        self._api.download_bucket_files(
            self.bucket_id,
            files=[(f"{prefix}/{filename}", str(local_path))],
        )

    def get_local_token(self) -> Optional[str]:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            return token_path.read_text().strip()
        return None


# =========================================================================
# Backend: S3 (and S3-compatible: GCS, R2, MinIO)
# =========================================================================


class _S3Backend:
    def __init__(self):
        import boto3
        config = _get_hub_config()
        self.bucket_name = config.get("bucket_id", "")
        self.prefix = config.get("prefix", "")
        kwargs = {}
        if config.get("endpoint_url"):
            kwargs["endpoint_url"] = config["endpoint_url"]
        if config.get("region"):
            kwargs["region_name"] = config["region"]
        self._s3 = boto3.client("s3", **kwargs)

    def _key(self, *parts: str) -> str:
        segments = [self.prefix] + list(parts) if self.prefix else list(parts)
        return "/".join(s.strip("/") for s in segments if s)

    def ping(self) -> bool:
        try:
            self._s3.head_bucket(Bucket=self.bucket_name)
            return True
        except Exception:
            return False

    def check_auth(self) -> Optional[str]:
        try:
            sts = __import__("boto3").client("sts")
            identity = sts.get_caller_identity()
            return identity.get("Arn")
        except Exception:
            return None

    def ensure_bucket(self) -> str:
        try:
            self._s3.head_bucket(Bucket=self.bucket_name)
        except Exception:
            self._s3.create_bucket(Bucket=self.bucket_name)
        return f"s3://{self.bucket_name}"

    def get_bucket_info(self):
        return {"bucket": self.bucket_name, "prefix": self.prefix}

    def sync_logs_up(self, local_logs_dir: Path, session: str) -> None:
        prefix = self._key("logs", session)
        for path in local_logs_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in (".txt", ".jsonl"):
                continue
            key = f"{prefix}/{path.relative_to(local_logs_dir)}"
            local_size = path.stat().st_size
            # Skip if remote already has same size
            try:
                resp = self._s3.head_object(Bucket=self.bucket_name, Key=key)
                if resp["ContentLength"] == local_size:
                    continue
            except self._s3.exceptions.ClientError:
                pass
            self._s3.upload_file(str(path), self.bucket_name, key)

    def sync_queue_up(self, local_logs_dir: Path, session: str) -> None:
        # Scoped variant of sync_logs_up: walk only the queue/ subdir but keep the key
        # rooted at local_logs_dir so the queue/ path component is preserved remotely.
        queue_dir = local_logs_dir / "queue"
        if not queue_dir.exists():
            return
        prefix = self._key("logs", session)
        for path in queue_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix != ".jsonl":
                continue
            key = f"{prefix}/{path.relative_to(local_logs_dir)}"
            local_size = path.stat().st_size
            try:
                resp = self._s3.head_object(Bucket=self.bucket_name, Key=key)
                if resp["ContentLength"] == local_size:
                    continue
            except self._s3.exceptions.ClientError:
                pass
            self._s3.upload_file(str(path), self.bucket_name, key)

    def sync_logs_down(self, local_logs_dir: Path, session: str, include: Optional[List[str]] = None) -> None:
        """Incremental download using Range GETs — only fetches new bytes for growing files."""
        local_logs_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._key("logs", session)
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key.removeprefix(prefix + "/")
                if not (rel.endswith(".txt") or rel.endswith(".jsonl")):
                    continue
                remote_size = obj["Size"]
                local_path = local_logs_dir / rel
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_size = local_path.stat().st_size if local_path.exists() else 0
                if local_size >= remote_size:
                    continue
                if local_size == 0:
                    self._s3.download_file(self.bucket_name, key, str(local_path))
                else:
                    # Range GET: fetch only new bytes and append
                    resp = self._s3.get_object(
                        Bucket=self.bucket_name, Key=key,
                        Range=f"bytes={local_size}-",
                    )
                    with open(local_path, "ab") as f:
                        for chunk in resp["Body"].iter_chunks(1024 * 64):
                            f.write(chunk)

    def list_logs(self, session: str) -> List[str]:
        prefix = self._key("logs", session) + "/"
        run_ids = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".txt") and "/" not in key.removeprefix(prefix):
                    run_ids.append(key.removeprefix(prefix).removesuffix(".txt"))
        return run_ids

    def upload_log(self, local_path: Path, run_id: str, session: str) -> None:
        key = self._key("logs", session, f"{run_id}.txt")
        self._s3.upload_file(str(local_path), self.bucket_name, key)

    def download_log(self, run_id: str, local_path: Path, session: str) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        key = self._key("logs", session, f"{run_id}.txt")
        self._s3.download_file(self.bucket_name, key, str(local_path))

    def list_weights(self, experiment_id: int, session: str) -> List[str]:
        prefix = self._key("weights", session, str(experiment_id)) + "/"
        filenames = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                filenames.append(obj["Key"].removeprefix(prefix))
        return filenames

    def upload_weight(self, local_path: Path, experiment_id: int, filename: str, session: str) -> None:
        key = self._key("weights", session, str(experiment_id), filename)
        self._s3.upload_file(str(local_path), self.bucket_name, key)

    def download_weight(self, experiment_id: int, filename: str, local_path: Path, session: str) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        key = self._key("weights", session, str(experiment_id), filename)
        self._s3.download_file(self.bucket_name, key, str(local_path))

    def get_local_token(self) -> Optional[str]:
        return None


# =========================================================================
# Backend: Local filesystem (zero deps, works with NFS/sshfs)
# =========================================================================


class _LocalBackend:
    def __init__(self):
        import shutil
        self._shutil = shutil
        config = _get_hub_config()
        self.root = Path(config.get("path", "~/.nanorun/hub")).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def ping(self) -> bool:
        return self.root.is_dir()

    def check_auth(self) -> Optional[str]:
        return "local" if self.root.is_dir() else None

    def ensure_bucket(self) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        return str(self.root)

    def get_bucket_info(self):
        return {"path": str(self.root)}

    def sync_logs_up(self, local_logs_dir: Path, session: str) -> None:
        dst_dir = self._dir("logs", session)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in local_logs_dir.rglob("*"):
            if not src.is_file():
                continue
            if src.suffix not in (".txt", ".jsonl"):
                continue
            rel = src.relative_to(local_logs_dir)
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src_size = src.stat().st_size
            dst_size = dst.stat().st_size if dst.exists() else 0
            if dst_size >= src_size:
                continue
            if dst_size == 0:
                self._shutil.copy2(src, dst)
            else:
                with open(src, "rb") as sf:
                    sf.seek(dst_size)
                    new_bytes = sf.read()
                with open(dst, "ab") as df:
                    df.write(new_bytes)

    def sync_queue_up(self, local_logs_dir: Path, session: str) -> None:
        # Scoped variant of sync_logs_up: walk only the queue/ subdir, keeping rel rooted
        # at local_logs_dir so the queue/ path component is preserved at the destination.
        queue_dir = local_logs_dir / "queue"
        if not queue_dir.exists():
            return
        dst_dir = self._dir("logs", session)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in queue_dir.rglob("*"):
            if not src.is_file():
                continue
            if src.suffix != ".jsonl":
                continue
            rel = src.relative_to(local_logs_dir)
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src_size = src.stat().st_size
            dst_size = dst.stat().st_size if dst.exists() else 0
            if dst_size >= src_size:
                continue
            if dst_size == 0:
                self._shutil.copy2(src, dst)
            else:
                with open(src, "rb") as sf:
                    sf.seek(dst_size)
                    new_bytes = sf.read()
                with open(dst, "ab") as df:
                    df.write(new_bytes)

    def sync_logs_down(self, local_logs_dir: Path, session: str, include: Optional[List[str]] = None) -> None:
        local_logs_dir.mkdir(parents=True, exist_ok=True)
        src_dir = self._dir("logs", session)
        if not src_dir.exists():
            return
        for src in src_dir.rglob("*"):
            if not src.is_file():
                continue
            if not (src.name.endswith(".txt") or src.name.endswith(".jsonl")):
                continue
            rel = src.relative_to(src_dir)
            dst = local_logs_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src_size = src.stat().st_size
            dst_size = dst.stat().st_size if dst.exists() else 0
            if dst_size >= src_size:
                continue
            if dst_size == 0:
                self._shutil.copy2(src, dst)
            else:
                with open(src, "rb") as sf:
                    sf.seek(dst_size)
                    new_bytes = sf.read()
                with open(dst, "ab") as df:
                    df.write(new_bytes)

    def list_logs(self, session: str) -> List[str]:
        logs_dir = self._dir("logs", session)
        if not logs_dir.exists():
            return []
        return [p.stem for p in logs_dir.glob("*.txt")]

    def upload_log(self, local_path: Path, run_id: str, session: str) -> None:
        dst = self._dir("logs", session, f"{run_id}.txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._shutil.copy2(local_path, dst)

    def download_log(self, run_id: str, local_path: Path, session: str) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        src = self._dir("logs", session, f"{run_id}.txt")
        self._shutil.copy2(src, local_path)

    def list_weights(self, experiment_id: int, session: str) -> List[str]:
        weights_dir = self._dir("weights", session, str(experiment_id))
        if not weights_dir.exists():
            return []
        return [p.name for p in weights_dir.iterdir() if p.is_file()]

    def upload_weight(self, local_path: Path, experiment_id: int, filename: str, session: str) -> None:
        dst = self._dir("weights", session, str(experiment_id), filename)
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._shutil.copy2(local_path, dst)

    def download_weight(self, experiment_id: int, filename: str, local_path: Path, session: str) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        src = self._dir("weights", session, str(experiment_id), filename)
        self._shutil.copy2(src, local_path)

    def get_local_token(self) -> Optional[str]:
        return None


# =========================================================================
# Backend: Iris (TPU jobs via iris CLI)
# =========================================================================


def _iris_job_id_to_filename(job_id: str) -> str:
    """Strip the leading /{user}/ prefix from an iris job_id for use as filename.

    Any remaining slashes are replaced with __ to prevent directory creation.
    """
    parts = job_id.strip("/").split("/", 1)
    name = parts[1] if len(parts) > 1 else parts[0]
    return name.replace("/", "__")


_IRIS_STATE_MAP = {
    "JOB_STATE_PENDING": "queued",
    "JOB_STATE_BUILDING": "queued",
    "JOB_STATE_RUNNING": "running",
    "JOB_STATE_SUCCEEDED": "completed",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_KILLED": "cancelled",
    "JOB_STATE_WORKER_FAILED": "failed",
    "JOB_STATE_UNSCHEDULABLE": "failed",
}


class _IrisBackend:
    """Hub backend for Iris sessions. Fetches logs via iris CLI."""

    def __init__(self, session_config):
        self.iris_config = session_config.iris_config
        self.iris_binary = session_config.iris_binary or "iris"
        self.iris_user = session_config.iris_user or ""
        self.iris_workspace = getattr(session_config, "iris_workspace", None)
        self.wandb_project = getattr(session_config, "wandb_project", None)
        self.wandb_entity = getattr(session_config, "wandb_entity", None)
        self._finalized_jobs: set = set()

    def _iris_cmd(self, *args: str) -> List[str]:
        cmd = [self.iris_binary]
        if self.iris_config:
            cmd.extend(["--config", self.iris_config])
        cmd.extend(args)
        return cmd

    def _run_iris(self, *args: str, timeout: int = 60) -> "subprocess.CompletedProcess":
        import subprocess
        cmd = self._iris_cmd(*args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=self.iris_workspace)

    def ping(self) -> bool:
        try:
            result = self._run_iris("job", "list", "--prefix", f"/{self.iris_user}/", "--json", timeout=30)
            return result.returncode == 0
        except Exception:
            return False

    def check_auth(self) -> Optional[str]:
        if self.ping():
            return self.iris_user or "iris"
        return None

    def list_jobs(self, *, strict: bool = False) -> List[dict]:
        """Call iris job list and return parsed JSON array."""
        result = self._run_iris("job", "list", "--prefix", f"/{self.iris_user}/", "--json", timeout=30)
        if result.returncode != 0:
            msg = f"iris job list failed with exit {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
            if strict:
                raise RuntimeError(msg)
            log.warning("[hub] %s", msg)
            return []
        try:
            jobs = json.loads(result.stdout)
        except (ValueError, json.JSONDecodeError) as e:
            msg = f"iris job list returned invalid JSON: {e}"
            if strict:
                raise RuntimeError(msg) from e
            log.warning("[hub] %s", msg)
            return []
        if not isinstance(jobs, list):
            msg = f"iris job list returned {type(jobs).__name__}, expected list"
            if strict:
                raise RuntimeError(msg)
            log.warning("[hub] %s", msg)
            return []
        return jobs

    def sync_logs_down(self, local_logs_dir: Path, session: str, include: Optional[List[str]] = None) -> None:
        """Fetch metrics from W&B and write to local log files in nanorun-parseable format.

        remote_run_id is now the wandb_run_id (12-char UUID). The iris job ID
        is stored in env_vars._iris_job_id for status reconciliation.
        """
        local_logs_dir.mkdir(parents=True, exist_ok=True)

        from .tracker import get_db
        conn = get_db()
        rows = conn.execute(
            """
            SELECT
                e.id, e.remote_run_id, e.status, e.env_vars,
                EXISTS (
                    SELECT 1 FROM crash_logs c WHERE c.experiment_id = e.id
                ) AS has_crash_log,
                EXISTS (
                    SELECT 1 FROM metrics m
                    WHERE m.experiment_id = e.id AND m.is_final_step = 1
                ) AS has_final_metric
            FROM experiments e
            WHERE e.remote_run_id IS NOT NULL AND e.session_name = ?
            """,
            (session,),
        ).fetchall()
        conn.close()
        rows = [dict(row) for row in rows]

        if not rows:
            return

        # Build job state mapping from iris job list (keyed by iris job ID)
        # Include both parent jobs (has_children) and direct-run jobs (no children)
        # Exclude child tasks (name ends with :reservation:, :subtask, etc.)
        jobs = self.list_jobs(strict=False)
        job_states = {}
        for j in jobs:
            name = j.get("name", "")
            if not name or ":" in name.split("/")[-1]:
                continue
            job_states[name] = _IRIS_STATE_MAP.get(j.get("state", ""), "")

        # Fetch W&B metrics for each tracked experiment
        projects = [p.strip() for p in (self.wandb_project or "").split(",") if p.strip()]
        if not self.wandb_entity or not projects:
            raise RuntimeError("iris W&B sync requires wandb_entity and wandb_project in the session config")

        def _as_int(value, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return default

        try:
            import wandb
            api = wandb.Api()
        except Exception as e:
            log.warning(f"[hub] W&B unavailable for iris sync: {e}")
            return

        row_errors = 0

        for row in rows:
            try:
                wandb_run_id = row["remote_run_id"]
                local_status = row["status"]
                try:
                    env = json.loads(row["env_vars"]) if row["env_vars"] else {}
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"experiment {row['id']} has invalid env_vars JSON: {e}") from e
                iris_job_id = env.get("_iris_job_id")

                # Reconcile status from iris job list
                if iris_job_id:
                    iris_status = job_states.get(iris_job_id, "")
                    if iris_status and iris_status != local_status:
                        if not (local_status == "completed" and iris_status == "failed"):
                            from .tracker import update_experiment_status
                            update_experiment_status(row["id"], iris_status)

                    # Fetch crash log for failed jobs that don't have one yet.
                    effective_status = iris_status or local_status
                    if effective_status == "failed" and not row["has_crash_log"]:
                        from .tracker import set_crash_log
                        result = self._run_iris("job", "logs", "--no-tail", iris_job_id, timeout=30)
                        if result.returncode == 0 and result.stdout.strip():
                            set_crash_log(row["id"], result.stdout[-4000:])
                        elif result.returncode != 0:
                            log.warning(
                                "[hub] Failed to fetch iris crash log for experiment %s (%s): %s",
                                row["id"],
                                iris_job_id,
                                result.stderr.strip() or result.stdout.strip(),
                            )

                if wandb_run_id in self._finalized_jobs:
                    continue

                # Permanently skip: terminal + we already have the final datapoint
                if local_status in ("completed", "failed", "cancelled") and row["has_final_metric"]:
                    self._finalized_jobs.add(wandb_run_id)
                    continue

                # Soft skip: terminal without final metric — don't hit W&B, but
                # don't add to _finalized_jobs so we re-check if status changes back
                effective_status = (iris_status if iris_job_id else "") or local_status
                if effective_status in ("completed", "failed", "cancelled"):
                    continue

                local_path = local_logs_dir / f"{wandb_run_id}.txt"

                # For legacy experiments without _wandb_run_id in env, fall back
                lookup_id = env.get("_wandb_run_id") or wandb_run_id
                if not lookup_id:
                    continue

                wb_run = None
                project_for_run = None
                lookup_errors = []
                for project in projects:
                    try:
                        found = api.runs(f"{self.wandb_entity}/{project}", filters={"display_name": lookup_id}, per_page=1)
                        if found:
                            wb_run = found[0]
                            project_for_run = project
                            break
                    except Exception as e:
                        lookup_errors.append(f"{project}: {e}")
                if not wb_run:
                    if lookup_errors:
                        raise RuntimeError(f"W&B lookup failed for {lookup_id}: {'; '.join(lookup_errors)}")
                    continue

                hist = wb_run.history(
                    keys=["eval/paloma/macro_loss", "_step", "_runtime"],
                    pandas=False,
                )

                # Also fetch train loss + step time (sampled — W&B returns ~500 points)
                train_hist = wb_run.history(
                    keys=["train/loss", "throughput/duration", "_step"],
                    pandas=False,
                )

                hist = list(hist or [])
                train_hist = list(train_hist or [])
                if not hist and not train_hist:
                    continue

                # Get configured total steps from run config (not summary._step which is just current progress)
                trainer_cfg = wb_run.config.get("trainer", {}) if wb_run.config else {}
                if isinstance(trainer_cfg, dict):
                    nested_trainer_cfg = trainer_cfg.get("trainer", {})
                    total_steps = (
                        nested_trainer_cfg.get("num_train_steps") if isinstance(nested_trainer_cfg, dict) else None
                    ) or trainer_cfg.get("num_train_steps")
                else:
                    total_steps = None
                if not total_steps:
                    all_steps = [_as_int(h.get("_step", 0)) for h in hist + train_hist if h.get("_step") is not None]
                    total_steps = wb_run.summary.get("_step") or max(all_steps, default=0)
                total_steps = _as_int(total_steps)
                if total_steps <= 0:
                    continue

                lines = []
                for h in hist:
                    step = h.get("_step", 0)
                    loss = h.get("eval/paloma/macro_loss")
                    runtime = h.get("_runtime", 0)
                    if loss is not None:
                        try:
                            step = _as_int(step)
                            loss = float(loss)
                            runtime = float(runtime or 0)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(loss):
                            continue
                        elapsed_ms = int(runtime * 1000)
                        lines.append(f"step:{step}/{total_steps} val_loss:{loss:.6f} train_time:{elapsed_ms}ms")

                for h in train_hist:
                    step = h.get("_step", 0)
                    loss = h.get("train/loss")
                    duration = h.get("throughput/duration")
                    if loss is not None:
                        try:
                            step = _as_int(step)
                            loss = float(loss)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(loss):
                            continue
                        suffix = ""
                        if duration is not None:
                            try:
                                suffix = f" step_avg:{float(duration) * 1000:.0f}ms"
                            except (TypeError, ValueError):
                                pass
                        lines.append(f"step:{step}/{total_steps} train_loss:{loss:.6f}{suffix}")

                if not lines:
                    continue

                # Sort by step so val_loss and train_loss are interleaved chronologically
                lines.sort(key=lambda l: int(l.split("/")[0].split(":")[1]))

                # Preserve header, update wandb URL with correct project/run ID
                correct_wandb_url = f"https://wandb.ai/{self.wandb_entity}/{project_for_run}/runs/{wb_run.id}"
                header = ""
                if local_path.exists():
                    content = local_path.read_text()
                    marker = "--- METADATA END ---\n"
                    idx = content.find(marker)
                    if idx >= 0:
                        header_section = content[: idx + len(marker)]
                        import re as _re
                        if "wandb_url:" in header_section:
                            header_section = _re.sub(r"wandb_url: .*\n", f"wandb_url: {correct_wandb_url}\n", header_section)
                        else:
                            header_section = content[:idx] + f"wandb_url: {correct_wandb_url}\n" + content[idx:idx + len(marker)]
                        header = header_section + "\n"

                local_path.write_text(header + "\n".join(lines) + "\n")

                # Only finalize once W&B is done AND we've seen eval data near the end
                # (W&B history API has eventual consistency — state can be "finished"
                # before all history rows are queryable)
                if wb_run.state in ("finished", "crashed", "failed"):
                    last_eval_step = max((
                        _as_int(h.get("_step", 0))
                        for h in hist
                        if h.get("_step") is not None and h.get("eval/paloma/macro_loss") is not None
                    ), default=0)
                    if last_eval_step >= total_steps - 10:
                        self._finalized_jobs.add(wandb_run_id)
            except Exception as e:
                row_errors += 1
                log.warning(
                    "[hub] Iris W&B sync failed for experiment %s (%s): %s",
                    row["id"],
                    row["remote_run_id"],
                    e,
                    exc_info=True,
                )
                continue

        if row_errors and row_errors == len(rows):
            raise RuntimeError(f"Iris W&B sync failed for all {row_errors} experiments in session {session}")

    def sync_logs_up(self, local_logs_dir: Path, session: str) -> None:
        pass

    def sync_queue_up(self, local_logs_dir: Path, session: str) -> None:
        pass

    def list_logs(self, session: str) -> List[str]:
        jobs = self.list_jobs()
        return [_iris_job_id_to_filename(j["name"]) for j in jobs if j.get("name")]

    def upload_log(self, local_path: Path, run_id: str, session: str) -> None:
        pass

    def download_log(self, run_id: str, local_path: Path, session: str) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        full_job_id = f"/{self.iris_user}/{run_id}"
        result = self._run_iris("job", "logs", "--max-lines", "999999", "--no-tail", full_job_id, timeout=120)
        if result.returncode == 0:
            local_path.write_text(result.stdout)

    def list_weights(self, experiment_id: int, session: str) -> List[str]:
        return []

    def upload_weight(self, local_path: Path, experiment_id: int, filename: str, session: str) -> None:
        pass

    def download_weight(self, experiment_id: int, filename: str, local_path: Path, session: str) -> None:
        pass

    def get_local_token(self) -> Optional[str]:
        return None


# =========================================================================
# Backend dispatch
# =========================================================================

_BACKENDS = {
    "hf": _HfBackend,
    "s3": _S3Backend,
    "local": _LocalBackend,
}

_backend_instance = None


def _get_backend_instance():
    global _backend_instance
    if _backend_instance is None:
        name = _get_backend()
        cls = _BACKENDS.get(name)
        if cls is None:
            raise ValueError(f"Unknown hub backend: {name!r}. Available: {list(_BACKENDS.keys())}")
        _backend_instance = cls()
    return _backend_instance


# =========================================================================
# Public API — delegates to active backend
# =========================================================================


def ping() -> bool:
    return _get_backend_instance().ping()


def check_auth() -> Optional[str]:
    return _get_backend_instance().check_auth()


def ensure_bucket() -> str:
    return _get_backend_instance().ensure_bucket()


def get_bucket_info():
    return _get_backend_instance().get_bucket_info()


def sync_logs_up(local_logs_dir: Path, session: str) -> None:
    _get_backend_instance().sync_logs_up(local_logs_dir, session)


def sync_queue_up(local_logs_dir: Path, session: str) -> None:
    _get_backend_instance().sync_queue_up(local_logs_dir, session)


def sync_logs_down(local_logs_dir: Path, session: str, include: Optional[List[str]] = None) -> None:
    _get_backend_instance().sync_logs_down(local_logs_dir, session, include)


def list_logs(session: str) -> List[str]:
    return _get_backend_instance().list_logs(session)


def upload_log(local_path: Path, run_id: str, session: str) -> None:
    _get_backend_instance().upload_log(local_path, run_id, session)


def download_log(run_id: str, local_path: Path, session: str) -> None:
    _get_backend_instance().download_log(run_id, local_path, session)


def list_weights(experiment_id: int, session: str) -> List[str]:
    return _get_backend_instance().list_weights(experiment_id, session)


def upload_weight(local_path: Path, experiment_id: int, filename: str, session: str) -> None:
    _get_backend_instance().upload_weight(local_path, experiment_id, filename, session)


def download_weight(experiment_id: int, filename: str, local_path: Path, session: str) -> None:
    _get_backend_instance().download_weight(experiment_id, filename, local_path, session)


def get_local_token() -> Optional[str]:
    return _get_backend_instance().get_local_token()
