"""Code sync operations - git push/pull."""

import hashlib
import re
import subprocess
import py_compile
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from .config import Config, SessionConfig
from .remote_control import RemoteSession, DaemonClient
from .lineage import (
    parse_parent_path,
    parse_kernels_path,
    parse_dependencies,
    read_local_file,
    compute_combined_hash,
    generate_combined_diff,
    store_diff,
)
from .script_manifest import (
    ManifestError,
    parse_script_manifest,
    resolve_repo_python_file,
)

console = Console()

LOCAL_BRANCH_PREFIX = "nanorun/local/"


def get_local_repo_path() -> Path:
    """Get the nanorun-platform repository path."""
    # The repo is always the parent of the nanorun package
    return Path(__file__).parent.parent


def _git(
    repo: Path,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def _git_error(action: str, result: subprocess.CompletedProcess[str]) -> ValueError:
    detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
    return ValueError(f"{action}: {detail}")


def _new_local_workspace_id(config: SessionConfig) -> str:
    """Build a path-safe, globally unique identity for a local session."""
    safe_name = re.sub(r"[^a-z0-9_-]+", "-", config.name.lower()).strip("-")
    safe_name = (safe_name or "local")[:40].rstrip("-")
    digest = hashlib.sha256(config.session_id.encode()).hexdigest()[:12]
    return f"{safe_name}-{digest}"


def ensure_local_session_branch(
    config: SessionConfig,
    *,
    switch_if_needed: bool,
) -> str:
    """Create/restore the branch owned by a local session.

    Legacy local configs are upgraded in place. Normal sync/job operations do
    not silently move an established worktree between branches; re-running
    ``session start --local`` is the explicit restoration path.
    """
    if config.session_type != "local":
        raise ValueError("A local Git branch can only be assigned to a local session")

    repo = Path(config.repo_path).expanduser().resolve()
    inside = _git(repo, ["rev-parse", "--show-toplevel"])
    if inside.returncode != 0:
        raise _git_error(f"Local session repository is not a Git worktree ({repo})", inside)

    if not config.started_at:
        config.started_at = datetime.now(timezone.utc).isoformat()

    if not config.workspace_id:
        if config.git_branch and config.git_branch.startswith(LOCAL_BRANCH_PREFIX):
            candidate = config.git_branch.removeprefix(LOCAL_BRANCH_PREFIX)
            if candidate and "/" not in candidate:
                config.workspace_id = candidate
        if not config.workspace_id:
            config.workspace_id = _new_local_workspace_id(config)

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", config.workspace_id):
        raise ValueError(
            f"Invalid local workspace identity in session config: {config.workspace_id!r}"
        )

    expected_branch = f"{LOCAL_BRANCH_PREFIX}{config.workspace_id}"
    if config.git_branch and config.git_branch != expected_branch:
        raise ValueError(
            "Local session Git branch does not match its workspace identity: "
            f"{config.git_branch!r} != {expected_branch!r}"
        )
    config.git_branch = expected_branch

    valid = _git(repo, ["check-ref-format", "--branch", expected_branch])
    if valid.returncode != 0:
        raise _git_error(f"Invalid local session branch {expected_branch!r}", valid)

    current_result = _git(repo, ["branch", "--show-current"])
    if current_result.returncode != 0:
        raise _git_error("Could not determine the current Git branch", current_result)
    current_branch = current_result.stdout.strip()

    exists = _git(
        repo,
        ["show-ref", "--verify", "--quiet", f"refs/heads/{expected_branch}"],
    ).returncode == 0

    if current_branch != expected_branch:
        if exists and not switch_if_needed:
            raise ValueError(
                f"Local session '{config.name}' owns branch '{expected_branch}', "
                f"but this worktree is on '{current_branch or 'detached HEAD'}'. "
                "Run 'nanorun session start --local' to restore its branch."
            )
        command = ["switch", expected_branch] if exists else ["switch", "-c", expected_branch]
        switched = _git(repo, command)
        if switched.returncode != 0:
            raise _git_error(
                f"Could not switch the local session to branch {expected_branch!r}",
                switched,
            )

    Config.save_session(config)
    return expected_branch


def ensure_local_daemon_namespace(
    config: SessionConfig,
    *,
    restart_for_code: bool = False,
) -> tuple[bool, bool]:
    """Restart a running local executor when its Hub namespace/code is stale."""
    from .remote_control import get_daemon_client

    daemon = get_daemon_client(config.name)
    if not daemon:
        return True, False
    try:
        if not daemon.is_daemon_running():
            return True, False
        status = daemon.get_status()
        if (
            restart_for_code
            or status.get("hub_session") != config.hub_namespace
        ):
            return daemon.restart_daemon(), True
        return True, False
    finally:
        daemon.close()


def expand_declared_files(files: list[str]) -> list[str]:
    """Expand selected entrypoints to their declared runtime source bundle."""

    repo_root = get_local_repo_path().resolve()
    expanded: list[str] = []
    for file_path in files:
        if file_path not in expanded:
            expanded.append(file_path)
        if not file_path.endswith(".py"):
            continue

        full_path = repo_root / file_path
        if not full_path.is_file():
            raise ManifestError(f"Selected file not found: {file_path}")
        manifest = parse_script_manifest(full_path.read_text())
        declared_paths = []
        if manifest.kernels:
            declared_paths.append(manifest.kernels)
        declared_paths.extend(path for _, path in manifest.dependencies)
        for declared_path in declared_paths:
            resolved = resolve_repo_python_file(repo_root, declared_path)
            if resolved not in expanded:
                expanded.append(resolved)
    return expanded


def get_changed_files() -> list[str]:
    """Get list of modified/staged files from git status.

    Returns:
        List of file paths relative to repo root
    """
    local_repo = get_local_repo_path()
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=local_repo,
        capture_output=True,
        text=True,
    )

    changed = []
    # Pattern: 2-char status (XY), optional space, then path
    # Examples: "M  file.py", " M file.py", "?? file.py", "R  old -> new"
    pattern = re.compile(r'^(..) ?(.+)$')

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue

        match = pattern.match(line)
        if not match:
            continue

        status, filepath = match.groups()
        # Skip deleted files (D in either position of status)
        if "D" in status:
            continue

        # Handle renamed files ("old -> new" or "\"old\" -> \"new\"")
        if " -> " in filepath:
            filepath = filepath.split(" -> ")[1]

        # Strip quotes from paths with spaces/special chars (git quotes these)
        if filepath.startswith('"') and filepath.endswith('"'):
            filepath = filepath[1:-1]

        # Expand directories to their contained files
        # Git shows untracked directories as "?? dir/" - we need the actual files
        if filepath.endswith("/"):
            dir_path = local_repo / filepath
            if dir_path.is_dir():
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        changed.append(str(file_path.relative_to(local_repo)))
                continue

        changed.append(filepath)

    return changed


def generate_lineage_diffs(changed_files: list[str] | None = None) -> int:
    """Generate diffs for scripts with parent declarations.

    Includes every declared dependency in the child code identity and diff.

    Args:
        changed_files: If provided, process Python files from this list,
            regardless of their directory. If None, process all repository
            Python files known to git.

    Returns:
        Number of diffs generated/updated
    """
    local_repo = get_local_repo_path()

    # Determine which files to process
    if changed_files is not None:
        changed_py = {
            f for f in changed_files
            if f.endswith(".py") and not Path(f).is_absolute()
        }
        if not changed_py:
            return 0
        files_to_process = set(changed_py)
    else:
        result = subprocess.run(
            [
                "git", "ls-files", "--cached", "--others",
                "--exclude-standard", "--", "*.py",
            ],
            cwd=local_repo,
            capture_output=True,
            text=True,
        )
        files_to_process = {
            path for path in result.stdout.splitlines() if path
        }

    diffs_generated = 0

    for rel_path in files_to_process:
        # Read and check for parent declaration
        content = read_local_file(rel_path)
        if content is None:
            continue

        parent_path = parse_parent_path(content)
        if parent_path is None:
            continue  # No parent - skip

        # Read parent content
        parent_content = read_local_file(parent_path)
        if parent_content is None:
            console.print(f"[yellow]Warning: Parent not found: {parent_path}[/yellow]")
            continue

        # Parse kernels from both child and parent
        child_kernels = parse_kernels_path(content)
        parent_kernels = parse_kernels_path(parent_content)
        child_dependencies = parse_dependencies(content)
        parent_dependencies = parse_dependencies(parent_content)
        try:
            child_kernels = (
                resolve_repo_python_file(local_repo, child_kernels)
                if child_kernels
                else None
            )
            parent_kernels = (
                resolve_repo_python_file(local_repo, parent_kernels)
                if parent_kernels
                else None
            )
            child_dependencies = {
                module: resolve_repo_python_file(local_repo, path)
                for module, path in child_dependencies.items()
            }
            parent_dependencies = {
                module: resolve_repo_python_file(local_repo, path)
                for module, path in parent_dependencies.items()
            }
        except ManifestError as error:
            console.print(f"[yellow]Warning: {error}[/yellow]")
            continue

        # Compute combined hash (entrypoint + all declared code files)
        child_hash = compute_combined_hash(
            rel_path,
            child_kernels,
            child_dependencies,
        )
        if child_hash is None:
            console.print(
                f"[yellow]Warning: Declared dependency not found for {rel_path}[/yellow]"
            )
            continue

        # Generate combined diff (entrypoint + all declared code files)
        diff_content = generate_combined_diff(
            rel_path, parent_path,
            child_kernels, parent_kernels,
            child_dependencies, parent_dependencies,
        )
        if diff_content:
            store_diff(child_hash, diff_content)
            diffs_generated += 1

    return diffs_generated


def check_python_syntax(files: list[str] | None = None) -> list[str]:
    """Check syntax of modified Python files using py_compile.

    Args:
        files: If provided, only check these files. If None, checks all modified files.

    Returns:
        List of error messages (empty if all files pass)
    """
    local_repo = get_local_repo_path()
    errors = []

    if files:
        # Check only the specified files
        for filepath in files:
            if not filepath.endswith(".py"):
                continue
            full_path = local_repo / filepath
            if not full_path.exists():
                continue
            try:
                py_compile.compile(str(full_path), doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"{filepath}: {e.msg}")
        return errors

    # Get list of modified/staged Python files
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=local_repo,
        capture_output=True,
        text=True,
    )

    # Pattern: 2-char status (XY), optional space, then path
    pattern = re.compile(r'^(..) ?(.+)$')

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue

        match = pattern.match(line)
        if not match:
            continue

        status, filepath = match.groups()
        # Skip deleted files (D in either position of status)
        if "D" in status:
            continue

        # Handle renamed files ("old -> new")
        if " -> " in filepath:
            filepath = filepath.split(" -> ")[1]

        if not filepath.endswith(".py"):
            continue

        full_path = local_repo / filepath
        if not full_path.exists():
            continue

        try:
            py_compile.compile(str(full_path), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"{filepath}: {e.msg}")

    return errors


def _get_session_sync_file(session_name: str) -> Path:
    """Path to the file that stores the last-synced commit for a session."""
    from .config import Config
    return Config.get_sessions_dir() / session_name / "last_synced_commit"


def get_last_synced_commit(session_name: str) -> str | None:
    """Get the commit hash that was last synced to this session."""
    f = _get_session_sync_file(session_name)
    if f.exists():
        return f.read_text().strip() or None
    return None


def record_synced_commit(session_name: str) -> None:
    """Record the current HEAD as the last-synced commit for this session."""
    local_repo = get_local_repo_path()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=local_repo,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        f = _get_session_sync_file(session_name)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(result.stdout.strip())


def has_unsynced_changes(
    files: list[str] | None = None,
    session_name: str | None = None,
    expected_upstream: str | None = None,
) -> bool:
    """Check if there are uncommitted or unpushed changes relative to a session.

    Args:
        files: If provided, only check these specific files (paths relative to repo root).
            If None, checks for any uncommitted/unpushed changes.
        session_name: If provided, also checks whether the file has changed since
            the last sync to this specific session.
        expected_upstream: If provided, considers a differently configured
            upstream unsynced even when the trees currently match.

    Returns:
        True if there are local changes not synced to remote
    """
    local_repo = get_local_repo_path()

    # A new local-session branch is not synchronized until its upstream exists.
    upstream = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        cwd=local_repo,
        capture_output=True,
        text=True,
    )
    if upstream.returncode != 0:
        return True
    if expected_upstream and upstream.stdout.strip() != expected_upstream:
        return True

    if files:
        # Check if specific files have uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain", "--"] + files,
            cwd=local_repo,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            return True

        # Check if any unpushed commits touch these files
        result = subprocess.run(
            ["git", "diff", "--name-only", "@{u}..HEAD", "--"] + files,
            cwd=local_repo,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return True
        if result.stdout.strip():
            return True

        # Check if files changed since last sync to this session
        if session_name:
            last_commit = get_last_synced_commit(session_name)
            if last_commit:
                result = subprocess.run(
                    ["git", "diff", "--name-only", f"{last_commit}..HEAD", "--"] + files,
                    cwd=local_repo,
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    return True

        return False

    # Check for uncommitted changes
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=local_repo,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        return True

    # Check for unpushed commits
    result = subprocess.run(
        ["git", "log", "@{u}..", "--oneline"],
        cwd=local_repo,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return True
    if result.stdout.strip():
        return True

    return False


def push_local_code(
    config: SessionConfig,
    message: str = None,
    skip_syntax_check: bool = False,
    files: list[str] | None = None,
) -> None:
    """Commit selected code and publish it to a local session's owned branch."""
    branch = ensure_local_session_branch(config, switch_if_needed=False)
    local_repo = Path(config.repo_path).expanduser().resolve()

    if files:
        console.print(f"[cyan]Syncing {len(files)} file(s) to {branch}...[/cyan]")
        for file_path in files:
            console.print(f"  [dim]{file_path}[/dim]")
    else:
        console.print(f"[cyan]Syncing all changes to {branch}...[/cyan]")

    changed_files = get_changed_files()
    if files:
        files_lower = {file_path.lower() for file_path in files}
        changed_files = [
            file_path
            for file_path in changed_files
            if file_path.lower() in files_lower
        ]

    if changed_files:
        console.print("[dim]Local changes detected[/dim]")
        if not skip_syntax_check:
            syntax_errors = check_python_syntax(files=files)
            if syntax_errors:
                console.print("[red]Syntax errors found:[/red]")
                for error in syntax_errors:
                    console.print(f"  [red]{error}[/red]")
                raise SystemExit(1)

        commit_message = message
        if commit_message is None:
            commit_message = (
                "sync " + ", ".join(files)
                if files
                else "nanorun sync"
            )

        add_args = ["add", "--", *files] if files else ["add", "-A"]
        added = _git(local_repo, add_args)
        if added.returncode != 0:
            console.print(f"[red]{_git_error('Could not stage local changes', added)}[/red]")
            raise SystemExit(1)

        commit_args = ["commit", "-m", commit_message]
        if files:
            # Do not sweep unrelated paths that the user happened to stage
            # before invoking a targeted sync.
            commit_args.extend(["--only", "--", *files])
        committed = _git(local_repo, commit_args)
        if committed.returncode != 0:
            staged = _git(local_repo, ["diff", "--cached", "--quiet"])
            if staged.returncode != 0:
                console.print(
                    f"[red]{_git_error('Could not commit local changes', committed)}[/red]"
                )
                raise SystemExit(1)
            console.print("[dim]Nothing to commit[/dim]")
        else:
            console.print(f"[green]Committed: {commit_message}[/green]")

    console.print(f"[dim]Pushing {branch} to origin...[/dim]")
    pushed = _git(
        local_repo,
        ["push", "--set-upstream", "origin", f"HEAD:refs/heads/{branch}"],
    )
    if pushed.returncode != 0:
        console.print(f"[red]{_git_error(f'Could not push {branch}', pushed)}[/red]")
        raise SystemExit(1)

    record_synced_commit(config.name)
    console.print(f"[green]Published local session branch: {branch}[/green]")

    lineage_candidates = files if files else None
    diffs_count = generate_lineage_diffs(lineage_candidates)
    if diffs_count > 0:
        console.print(f"[dim]Generated {diffs_count} lineage diff(s)[/dim]")

    daemon_code_changed = bool(changed_files) and any(
        path in changed_files
        for path in ("nanorun/remote_daemon.py", "nanorun/script_manifest.py")
    )
    try:
        daemon_ready, daemon_restarted = ensure_local_daemon_namespace(
            config,
            restart_for_code=daemon_code_changed,
        )
        if not daemon_ready:
            console.print("[yellow]Failed to restart local execution daemon[/yellow]")
        elif daemon_restarted:
            console.print(
                "[green]Restarted local execution daemon "
                "(code or Hub namespace changed)[/green]"
            )
    except Exception as error:
        console.print(
            f"[yellow]Could not refresh local execution daemon: {error}[/yellow]"
        )


def push_code(remote: RemoteSession, message: str = None, skip_syntax_check: bool = False,
              files: list[str] | None = None) -> None:
    """Push local code changes to remote.

    Args:
        remote: Remote session to sync to.
        message: Commit message. Auto-generated if None.
        skip_syntax_check: Skip Python syntax validation.
        files: If provided, only stage and sync these specific files (paths relative to repo root).
            If None, stages all changes (git add -A).
    """
    local_repo = get_local_repo_path()

    if files:
        console.print(f"[cyan]Syncing {len(files)} file(s)...[/cyan]")
        for f in files:
            console.print(f"  [dim]{f}[/dim]")
    else:
        console.print(f"[cyan]Syncing all changes from {local_repo}...[/cyan]")

    # Capture changed files before committing (for lineage diff generation)
    changed_files = get_changed_files()

    if files:
        # Filter changed_files to only the files we're syncing (case-insensitive for macOS)
        files_lower = {f.lower() for f in files}
        changed_files = [f for f in changed_files if f.lower() in files_lower]

    if changed_files:
        console.print("[dim]Local changes detected[/dim]")

        # Check syntax of modified Python files
        if not skip_syntax_check:
            syntax_errors = check_python_syntax(files=files)
            if syntax_errors:
                console.print("[red]Syntax errors found:[/red]")
                for error in syntax_errors:
                    console.print(f"  [red]{error}[/red]")
                raise SystemExit(1)

        # Stage and commit
        if message is None:
            if files:
                message = "sync " + ", ".join(files)
            else:
                message = "nanorun sync"

        if files:
            subprocess.run(["git", "add", "--"] + files, cwd=local_repo)
        else:
            subprocess.run(["git", "add", "-A"], cwd=local_repo)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=local_repo,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            console.print(f"[green]Committed: {message}[/green]")
        else:
            console.print("[dim]Nothing to commit[/dim]")

    # Step 2: Push to remote
    console.print("[dim]Pushing to origin...[/dim]")
    result = subprocess.run(
        ["git", "push"],
        cwd=local_repo,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 and "Everything up-to-date" not in result.stderr:
        console.print(f"[yellow]Push warning: {result.stderr}[/yellow]")
    else:
        console.print("[green]Pushed to origin[/green]")

    # Step 3: Pull on remote (use agent forwarding for git auth)
    console.print("[dim]Pulling on remote...[/dim]")
    remote_result = remote.run_with_agent(
        f"cd {remote.config.repo_path} && git pull",
        timeout=60
    )

    if remote_result.success:
        # Show what changed
        if "Already up to date" in remote_result.stdout:
            console.print("[dim]Remote already up to date[/dim]")
        else:
            console.print("[green]Remote updated[/green]")
            # Show changed files
            for line in remote_result.stdout.split("\n"):
                if "|" in line or "create mode" in line or "delete mode" in line:
                    console.print(f"  [dim]{line.strip()}[/dim]")
        record_synced_commit(remote.config.name)
    else:
        console.print(f"[red]Remote pull failed: {remote_result.stderr}[/red]")

    # Step 4: Generate lineage diffs. For a targeted sync, include clean
    # entrypoints too so changing only a declared dependency still creates the
    # diff keyed by the entrypoint's new combined hash.
    lineage_candidates = files if files else None
    diffs_count = generate_lineage_diffs(lineage_candidates)
    if diffs_count > 0:
        console.print(f"[dim]Generated {diffs_count} lineage diff(s)[/dim]")

    # Step 5: Restart the remote daemon when it or its shared manifest parser
    # changed.
    daemon_code_changed = changed_files and any(
        path in changed_files
        for path in ("nanorun/remote_daemon.py", "nanorun/script_manifest.py")
    )
    if daemon_code_changed:
        with DaemonClient(remote) as daemon:
            if daemon.is_daemon_running():
                try:
                    if daemon.restart_daemon():
                        console.print("[green]Restarted remote daemon (code changed)[/green]")
                    else:
                        console.print("[yellow]Failed to restart daemon[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]Could not restart daemon: {e}[/yellow]")
