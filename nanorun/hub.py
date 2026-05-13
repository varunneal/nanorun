"""Hub integration for log and weight storage.

Dispatches to a configured backend (HuggingFace Buckets, S3, etc.)
based on [hub] section in nanorun.toml.

Storage layout (same across all backends):
  logs/{session}/{run_id}.txt              - Training log files
  logs/{session}/mappings/*.jsonl           - Experiment mappings
  weights/{session}/{experiment_id}/       - Model checkpoints
"""

import os
from pathlib import Path
from typing import List, Optional

from .project_config import load_project_config

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


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
            include=["*.txt", "*.jsonl", "mappings/*.jsonl"],
            quiet=True,
        )

    def sync_logs_down(self, local_logs_dir: Path, session: str, include: Optional[List[str]] = None) -> None:
        local_logs_dir.mkdir(parents=True, exist_ok=True)
        self._sync_bucket(
            f"{self.bucket_handle}/logs/{session}",
            str(local_logs_dir),
            include=include or ["*.txt", "*.jsonl", "mappings/*.jsonl"],
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
