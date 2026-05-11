"""Code sync operations - git push/pull."""

import re
import subprocess
import py_compile
from pathlib import Path

from rich.console import Console

from .remote_control import RemoteSession, DaemonClient
from .lineage import (
    parse_parent_path,
    parse_kernels_path,
    read_local_file,
    compute_file_hash,
    compute_combined_hash,
    generate_diff,
    generate_combined_diff,
    store_diff,
    get_diffs_dir,
)

console = Console()


def get_local_repo_path() -> Path:
    """Get the nanorun-platform repository path."""
    # The repo is always the parent of the nanorun package
    return Path(__file__).parent.parent


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

    Supports kernels versioning: if a script declares `kernels: path/to/kernels.py`,
    the diff will include changes to both the script and its kernels file.

    Args:
        changed_files: If provided, only process these files (and files that
            declare them as parents or reference them as kernels). If None,
            processes all files in experiments/.

    Returns:
        Number of diffs generated/updated
    """
    local_repo = get_local_repo_path()
    script_dirs = [local_repo / "experiments", local_repo / "interp"]
    script_dirs = [d for d in script_dirs if d.exists()]

    if not script_dirs:
        return 0

    script_prefixes = ("experiments/", "interp/")

    # Determine which files to process
    if changed_files is not None:
        changed_py = {f for f in changed_files if f.endswith(".py") and f.startswith(script_prefixes)}
        if not changed_py:
            return 0
        # Only process the changed files themselves (skip expensive child scan)
        files_to_process = set(changed_py)
    else:
        files_to_process = set()
        for script_dir in script_dirs:
            files_to_process.update(
                str(py_file.relative_to(local_repo))
                for py_file in script_dir.rglob("*.py")
            )

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

        # Compute combined hash (script + kernels)
        child_hash = compute_combined_hash(rel_path, child_kernels)

        # Generate combined diff (script + kernels)
        diff_content = generate_combined_diff(
            rel_path, parent_path,
            child_kernels, parent_kernels
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


def has_unsynced_changes(files: list[str] | None = None) -> bool:
    """Check if there are uncommitted or unpushed changes.

    Args:
        files: If provided, only check these specific files (paths relative to repo root).
            If None, checks for any uncommitted/unpushed changes.

    Returns:
        True if there are local changes not synced to remote
    """
    local_repo = get_local_repo_path()

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
    if result.stdout.strip():
        return True

    return False


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
    else:
        console.print(f"[red]Remote pull failed: {remote_result.stderr}[/red]")

    # Step 4: Generate lineage diffs for changed scripts with parent declarations
    diffs_count = generate_lineage_diffs(changed_files if changed_files else None)
    if diffs_count > 0:
        console.print(f"[dim]Generated {diffs_count} lineage diff(s)[/dim]")

    # Step 5: Restart remote daemon only if daemon code was changed
    # The daemon is self-contained in remote_daemon.py with no local imports
    daemon_code_changed = changed_files and "nanorun/remote_daemon.py" in changed_files
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
