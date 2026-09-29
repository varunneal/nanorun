"""Git synchronization through shell-only SSH without remote Git credentials."""

import shlex
import subprocess
import tempfile
from pathlib import Path


def sync_repository(remote, repo_path: str, repo_url: str, branch: str | None = None):
    """Clone/update a committed branch with a verified, usually incremental bundle.

    The controller still pushes to origin in the usual sync flow. The bundle
    delivers the same commit to gateways that cannot forward an SSH agent. A
    remote merge is always fast-forward-only; local edits and divergent commits
    are preserved and reported as errors.
    """
    from .config import get_repo_root
    from .remote_control import CommandResult

    repo = get_repo_root()

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)

    def failure(detail):
        return CommandResult("", detail, 1)

    if branch is None:
        result = git("branch", "--show-current")
        if result.returncode or not result.stdout.strip():
            return failure("Proxy sync requires a checked-out Git branch")
        branch = result.stdout.strip()
    valid = git("check-ref-format", "--branch", branch)
    if valid.returncode:
        return failure("Invalid Git branch for proxy sync")
    tip = git("rev-parse", f"refs/heads/{branch}")
    if tip.returncode:
        return failure(tip.stderr)
    if repo_path.startswith("~"):
        from .setup import resolve_repo_path
        repo_path = resolve_repo_path(remote, repo_path)
        if repo_path.startswith("~"):
            return failure("Could not resolve the remote home directory")
    path = shlex.quote(repo_path)
    head = remote.run(f"git -C {path} rev-parse HEAD 2>/dev/null", timeout=15)
    old = head.stdout.strip() if head.success else ""
    if old == tip.stdout.strip():
        return CommandResult("Already up to date", "", 0)
    revisions = [f"refs/heads/{branch}"]
    # A full bundle is needed only when there is no shared commit available.
    if old and git("merge-base", "--is-ancestor", old, tip.stdout.strip()).returncode == 0:
        revisions.append(f"^{old}")
    with tempfile.TemporaryDirectory(prefix="nanorun-proxy-git-") as temp:
        bundle = Path(temp) / "repo.bundle"
        result = git("bundle", "create", str(bundle), *revisions)
        if result.returncode:
            return failure(result.stderr)
        created = remote.run("mktemp /tmp/nanorun-git-XXXXXXXX.bundle", timeout=15)
        if not created.success or not created.stdout.strip():
            return failure(created.stderr or "Could not create remote Git bundle")
        remote_bundle = created.stdout.strip()
        quoted_bundle = shlex.quote(remote_bundle)
        try:
            uploaded = remote.upload_file(bundle, remote_bundle)
            if not uploaded.success:
                return uploaded
            ref = shlex.quote(f"refs/heads/{branch}:refs/remotes/origin/{branch}")
            script = (
                f"set -e\nif git -C {path} rev-parse --git-dir >/dev/null 2>&1; then\n"
                f"  git -C {path} fetch {quoted_bundle} {ref}\n"
                f"  git -C {path} merge --ff-only FETCH_HEAD\n"
                "else\n"
                f"  git clone --branch {shlex.quote(branch)} {quoted_bundle} {path}\n"
                f"  git -C {path} remote set-url origin {shlex.quote(repo_url)}\n"
                "fi\n"
                f"test \"$(git -C {path} rev-parse HEAD)\" = {shlex.quote(tip.stdout.strip())}\n"
            )
            return remote.run_script(script, timeout=180)
        finally:
            remote.run(f"rm -f -- {quoted_bundle}", timeout=15)
