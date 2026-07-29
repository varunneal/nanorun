"""Disposable worker process for one local Hub log sync."""

import argparse
import json
import sys
from typing import List, Optional

from .config import Config


RESULT_PREFIX = "NANORUN_HUB_SYNC_RESULT="


def sync_logs_down(session_name: str) -> Optional[List[str]]:
    """Pull one session's logs and return the backend's changed-file list."""
    from . import hub

    session_config = Config.load_session(session_name)
    session_logs_dir = Config.get_config_dir() / "logs" / session_name
    if session_config and session_config.session_type == "iris":
        backend = hub._IrisBackend(session_config)
        return backend.sync_logs_down(session_logs_dir, session_name)
    return hub.sync_logs_down(session_logs_dir, session_name)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_name")
    args = parser.parse_args(argv)

    try:
        changed = sync_logs_down(args.session_name)
    except Exception as exc:
        print(
            f"Hub sync failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(
        RESULT_PREFIX + json.dumps({"changed": changed}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
