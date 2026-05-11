"""HuggingFace Buckets integration for log and weight storage.

Bucket layout (varunneal/nanorun):
  logs/{session}/{run_id}.txt              - Training log files
  weights/{session}/{experiment_id}/       - Model checkpoints
"""

from pathlib import Path
from typing import List, Optional

from huggingface_hub import (
    HfApi,
    bucket_info,
    create_bucket,
    list_bucket_tree,
    sync_bucket,
)

BUCKET_ID = "varunneal/nanorun"
BUCKET_HANDLE = f"hf://buckets/{BUCKET_ID}"

_api = HfApi()


def _logs_prefix(session: str) -> str:
    return f"logs/{session}"


def _weights_prefix(session: str, experiment_id: int) -> str:
    return f"weights/{session}/{experiment_id}"


def ensure_bucket() -> str:
    """Ensure the nanorun bucket exists. Returns the bucket handle."""
    create_bucket(BUCKET_ID, exist_ok=True)
    return BUCKET_HANDLE


def get_bucket_info():
    """Get bucket metadata (size, file count, etc.)."""
    return bucket_info(BUCKET_ID)


# =========================================================================
# Log operations
# =========================================================================


def sync_logs_up(local_logs_dir: Path, session: str) -> None:
    """Push local logs directory to bucket. Syncs .txt and .jsonl files."""
    sync_bucket(
        str(local_logs_dir),
        f"{BUCKET_HANDLE}/{_logs_prefix(session)}",
        include=["*.txt", "*.jsonl", "mappings/*.jsonl"],
        quiet=True,
    )


def sync_logs_down(
    local_logs_dir: Path,
    session: str,
    include: Optional[List[str]] = None,
) -> None:
    """Pull logs from bucket to local directory."""
    local_logs_dir.mkdir(parents=True, exist_ok=True)
    sync_bucket(
        f"{BUCKET_HANDLE}/{_logs_prefix(session)}",
        str(local_logs_dir),
        include=include or ["*.txt", "*.jsonl", "mappings/*.jsonl"],
        quiet=True,
    )


def list_logs(session: str) -> List[str]:
    """List all log files for a session. Returns list of run_id strings."""
    prefix = _logs_prefix(session)
    run_ids = []
    for item in list_bucket_tree(BUCKET_ID, prefix=prefix, recursive=True):
        if item.type == "file" and item.path.endswith(".txt"):
            name = item.path.removeprefix(f"{prefix}/").removesuffix(".txt")
            run_ids.append(name)
    return run_ids


def upload_log(local_path: Path, run_id: str, session: str) -> None:
    """Upload a single log file to the bucket."""
    prefix = _logs_prefix(session)
    _api.batch_bucket_files(
        BUCKET_ID,
        add=[(str(local_path), f"{prefix}/{run_id}.txt")],
    )


def download_log(run_id: str, local_path: Path, session: str) -> None:
    """Download a single log file from the bucket."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = _logs_prefix(session)
    _api.download_bucket_files(
        BUCKET_ID,
        files=[(f"{prefix}/{run_id}.txt", str(local_path))],
    )


# =========================================================================
# Weight operations
# =========================================================================


def list_weights(experiment_id: int, session: str) -> List[str]:
    """List checkpoint files for an experiment. Returns list of filenames."""
    prefix = _weights_prefix(session, experiment_id)
    filenames = []
    for item in list_bucket_tree(BUCKET_ID, prefix=prefix, recursive=True):
        if item.type == "file":
            filenames.append(item.path.removeprefix(f"{prefix}/"))
    return filenames


def upload_weight(
    local_path: Path, experiment_id: int, filename: str, session: str
) -> None:
    """Upload a checkpoint file to the bucket."""
    prefix = _weights_prefix(session, experiment_id)
    _api.batch_bucket_files(
        BUCKET_ID,
        add=[(str(local_path), f"{prefix}/{filename}")],
    )


def download_weight(
    experiment_id: int, filename: str, local_path: Path, session: str
) -> None:
    """Download a checkpoint file from the bucket."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = _weights_prefix(session, experiment_id)
    _api.download_bucket_files(
        BUCKET_ID,
        files=[(f"{prefix}/{filename}", str(local_path))],
    )


# =========================================================================
# Auth
# =========================================================================


def check_auth() -> Optional[str]:
    """Check if HF token is available. Returns username or None."""
    try:
        info = _api.whoami()
        return info.get("name")
    except Exception:
        return None


def get_local_token() -> Optional[str]:
    """Read HF token from local cache."""
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None
