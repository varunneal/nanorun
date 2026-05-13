"""Project-level configuration from nanorun.toml at repo root."""

from pathlib import Path
from functools import lru_cache

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _find_repo_root() -> Path:
    """Walk up from this file to find the repo root (contains nanorun.toml or .git)."""
    p = Path(__file__).resolve().parent.parent
    while p != p.parent:
        if (p / "nanorun.toml").exists():
            return p
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_project_config() -> dict:
    root = _find_repo_root()
    toml_path = root / "nanorun.toml"
    if not toml_path.exists():
        return {}
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


def get_bucket_id() -> str:
    config = load_project_config()
    return config.get("hub", {}).get("bucket_id", "")


def get_repo_url() -> str:
    config = load_project_config()
    return config.get("repo", {}).get("url", "")
