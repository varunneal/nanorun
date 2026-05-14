"""CLI entry point for nanorun."""

import sys
from pathlib import Path

import click
from click.shell_completion import CompletionItem
from rich.console import Console
from rich.table import Table

from .config import Config, SessionConfig, Track, discover_tracks, get_track, get_repo_root, infer_track_from_path
from .remote_control import RemoteSession, get_session, require_session, resolve_session_config

console = Console()


def _resolve_script_path(script: str) -> str:
    """Resolve a script path relative to repo root."""
    repo_root = get_repo_root()
    try:
        return str(Path(script).resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return script


def _resolve_session(ctx, param, value):
    """Click callback: resolve session name to active session if not specified."""
    if value:
        sc = Config.load_session(value)
        if not sc:
            console.print(f"[red]Session '{value}' not found.[/red]")
            raise SystemExit(1)
        return value
    name = Config.get_active_session_name()
    if name:
        return name
    from .remote_control import _find_sole_connected_session
    sole = _find_sole_connected_session()
    if sole:
        return sole
    console.print("[red]No active session. Run 'nanorun session start <host>' first.[/red]")
    raise SystemExit(1)


def session_option(f):
    """Decorator to add --session option. Always resolves to a valid session name."""
    f = click.option(
        "--session", "session_name", default=None,
        callback=_resolve_session, is_eager=True, expose_value=True,
        help="Session name (default: active session)",
    )(f)
    return f


class RichGroup(click.Group):
    """Custom Click group that displays errors in red."""

    def main(self, *args, standalone_mode=True, **kwargs):
        try:
            return super().main(*args, standalone_mode=False, **kwargs)
        except click.ClickException as e:
            err_console = Console(stderr=True)
            if hasattr(e, 'ctx') and e.ctx:
                err_console.print(f"[dim]Usage: {e.ctx.command_path} [OPTIONS] {' '.join(p.name.upper() for p in e.ctx.command.params if isinstance(p, click.Argument))}[/dim]")
                err_console.print(f"[dim]Try '{e.ctx.command_path} --help' for help.[/dim]")
                err_console.print()
            err_console.print(f"[red]Error: {e.format_message()}[/red]")
            if standalone_mode:
                raise SystemExit(e.exit_code)
            raise


def complete_track_names(ctx, param, incomplete):
    """Shell completion for track names."""
    tracks = discover_tracks()
    return [
        CompletionItem(t.name, help=t.description or t.directory)
        for t in tracks
        if t.name.startswith(incomplete)
    ]


@click.group(cls=RichGroup)
@click.version_option()
def cli():
    """nanorun - Experiment testing platform for NanoGPT speedruns."""
    pass


# ============================================================================
# Session commands
# ============================================================================

@cli.group()
def session():
    """Manage remote SSH sessions."""
    pass


@session.command("start")
@click.argument("host")
@click.option("--port", "-p", default=22, help="SSH port")
@click.option("--gpu-type", default=None, help="Override GPU type (H100, H200, GH200, DGX_SPARK)")
@click.option("--key-file", "-i", default=None, help="Path to SSH private key file")
@click.option("--ssh-option", "-o", multiple=True, help="Extra SSH -o options (e.g. -o IdentitiesOnly=yes)")
@click.option("--name", "-n", default=None, help="Session name (auto-generated if not specified)")
def session_start(host: str, port: int, gpu_type: str, key_file: str, ssh_option: tuple, name: str):
    """Connect to a remote machine.

    HOST should be in format user@hostname or just hostname (defaults to root@).
    """
    from .setup import detect_gpu_type, detect_gpu_count

    if "@" in host:
        user, hostname = host.split("@", 1)
    else:
        user, hostname = "root", host

    # Remove from known_hosts to avoid host key verification errors
    # (common when cloud IPs get reassigned to different machines)
    import subprocess
    subprocess.run(
        ["ssh-keygen", "-R", hostname],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Generate session name if not provided
    if not name:
        name = Config.next_session_name()

    console.print(f"[cyan]Connecting to {user}@{hostname}:{port}...[/cyan]")

    session_config = SessionConfig(
        name=name,
        host=hostname,
        user=user,
        port=port,
        key_file=key_file,
        ssh_options=list(ssh_option) if ssh_option else None,
    )

    remote = RemoteSession(session_config)

    # Test connection
    success, msg = remote.test_connection()
    if not success:
        console.print(f"[red]Connection failed: {msg}[/red]")
        raise SystemExit(1)

    console.print("[green]Connection successful![/green]")

    # Detect or use provided GPU type
    if gpu_type:
        session_config.gpu_type = gpu_type
        console.print(f"[dim]GPU type: {gpu_type} (override)[/dim]")
    else:
        detected_type = detect_gpu_type(remote)
        session_config.gpu_type = detected_type
        console.print(f"[dim]GPU type: {detected_type} (detected)[/dim]")

    # Detect GPU count
    gpu_count = detect_gpu_count(remote)
    session_config.gpu_count = gpu_count
    console.print(f"[dim]GPU count: {gpu_count} (detected)[/dim]")

    # Check for tmux
    if not remote.check_tmux():
        console.print("[yellow]Warning: tmux not found on remote. Some features may not work.[/yellow]")
    else:
        console.print("[dim]tmux available[/dim]")
        # Create session
        if remote.create_tmux_session():
            console.print(f"[dim]tmux session '{session_config.tmux_session}' ready[/dim]")

    # Save session
    config = Config(session=session_config)
    config.save()

    console.print(f"[green]Session '{name}' started! Config saved to .nanorun/sessions/{name}.json[/green]")


@session.command("stop")
def session_stop():
    """Disconnect from remote machine and cleanup."""
    config = Config.load()
    if not config.session:
        console.print("[yellow]No active session.[/yellow]")
        return

    remote = RemoteSession(config.session)
    host = f"{config.session.user}@{config.session.host}"

    # Kill tmux session
    if remote.tmux_session_exists():
        if remote.kill_tmux_session():
            console.print(f"[dim]Killed tmux session[/dim]")

    Config.clear_active()
    console.print(f"[green]Disconnected from {host}[/green]")


@session.command("cleanup")
def session_cleanup():
    """Remove all disconnected sessions.

    Checks each session's daemon state. If marked 'disconnected', removes
    the session config and its state directory.
    """
    import shutil
    from .local_daemon import SessionState

    sessions = Config.list_sessions()
    if not sessions:
        console.print("[dim]No sessions configured[/dim]")
        return

    removed = 0
    for sc in sessions:
        state = SessionState.load(sc.name)
        if state.status == "disconnected":
            # Remove session config
            Config.delete_session(sc.name)
            # Remove per-session state dir
            state_dir = Config.get_session_state_dir(sc.name)
            if state_dir.exists():
                shutil.rmtree(state_dir, ignore_errors=True)
            console.print(f"  Removed [bold]{sc.name}[/bold] ({sc.user}@{sc.host})")
            removed += 1

    if removed:
        console.print(f"\n[green]Cleaned up {removed} disconnected session(s)[/green]")
    else:
        console.print("[dim]No disconnected sessions to clean up[/dim]")


@session.command("status")
@session_option
def session_status(session_name):
    """Show current session status."""
    config = Config(session=resolve_session_config(session_name)) if session_name else Config.load()

    if not config.session:
        console.print("[yellow]No active session.[/yellow]")
        console.print("Run 'nanorun session start <host>' to connect.")
        return

    s = config.session
    remote = RemoteSession(s)

    table = Table(title="Session Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Session", s.name)
    table.add_row("Host", f"{s.user}@{s.host}:{s.port}")
    table.add_row("GPU Type", s.gpu_type)
    table.add_row("GPU Count", str(getattr(s, 'gpu_count', 1)))
    table.add_row("CUDA Version", s.cuda_version or "[dim]not detected[/dim]")
    table.add_row("Has sudo", "yes" if s.has_sudo else "no")
    table.add_row("Repo path", s.repo_path)

    # Check connection
    success, _ = remote.test_connection()
    table.add_row("Connection", "[green]connected[/green]" if success else "[red]disconnected[/red]")

    # Check tmux
    if success:
        if remote.tmux_session_exists():
            windows = remote.get_tmux_windows()
            table.add_row("tmux session", f"[green]active[/green] ({len(windows)} windows)")
        else:
            table.add_row("tmux session", "[yellow]no session[/yellow]")

    console.print(table)


@session.command("list")
def session_list():
    """List all saved sessions."""
    sessions = Config.list_sessions()
    active_name = Config.get_active_session_name()

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        console.print("Run 'nanorun session start <host>' to create one.")
        return

    from .local_daemon import SessionState

    table = Table(title="Sessions")
    table.add_column("Name", style="cyan")
    table.add_column("Host")
    table.add_column("GPU")
    table.add_column("Status")
    table.add_column("Active", justify="center")

    for s in sessions:
        is_active = "[green]*[/green]" if s.name == active_name else ""
        state = SessionState.load(s.name)
        status_color = {"connected": "green", "connecting": "yellow", "disconnected": "red"}.get(state.status, "dim")
        status_str = f"[{status_color}]{state.status}[/{status_color}]"
        table.add_row(
            s.name,
            f"{s.user}@{s.host}:{s.port}",
            f"{s.gpu_count}x {s.gpu_type}",
            status_str,
            is_active,
        )

    console.print(table)


@session.command("switch")
@click.argument("name")
def session_switch(name: str):
    """Switch the active session.

    NAME is the session name (see 'nanorun session list').
    """
    try:
        Config.set_active_session(name)
        session = Config.load_session(name)
        console.print(f"[green]Switched to session '{name}' ({session.user}@{session.host})[/green]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sessions = Config.list_sessions()
        if sessions:
            names = ", ".join(s.name for s in sessions)
            console.print(f"[dim]Available: {names}[/dim]")
        raise SystemExit(1)


@session.command("attach")
@session_option
def session_attach(session_name):
    """Attach to the remote tmux session."""
    remote = require_session(session_name)
    if not remote.tmux_session_exists():
        console.print("[yellow]No tmux session. Creating one...[/yellow]")
        remote.create_tmux_session()
    remote.attach_tmux()


@session.command("setup")
@click.option("--verify", is_flag=True, help="Only verify the setup, don't install anything")
@click.option("--interactive", "-i", is_flag=True, help="Prompt for confirmation at each step")
@session_option
def session_setup(verify: bool, interactive: bool, session_name):
    """Set up the remote machine for training."""
    from .setup import run_setup, verify_setup

    remote = require_session(session_name)

    if verify:
        verify_setup(remote)
    else:
        run_setup(remote, auto_yes=not interactive)


# ============================================================================
# Track commands
# ============================================================================

@cli.group()
def track():
    """Manage experiment tracks/workstreams."""
    pass


@track.command("create")
@click.argument("name")
@click.argument("directory", required=False)
@click.option("-d", "--description", default="", help="Description of this track")
def track_create(name: str, directory: str | None, description: str):
    """Create a new experiment track.

    NAME is a short identifier (e.g., 'lr-sweep', 'attention-mods').
    DIRECTORY is the folder for this track's experiments (defaults to 'experiments/<name>/').
    """
    # Default directory to experiments/<name> if not specified
    if directory is None:
        directory = f"experiments/{name}"

    # Check if track with this name already exists
    if get_track(name):
        console.print(f"[red]Track '{name}' already exists[/red]")
        raise SystemExit(1)

    # Create track (this creates .track.json in the directory)
    t = Track.create(name, directory, description)
    console.print(f"[green]Created track '{name}' -> {directory}[/green]")

    if description:
        console.print(f"[dim]{description}[/dim]")



@track.command("list")
def track_list():
    """List all experiment tracks."""
    tracks = discover_tracks()

    if not tracks:
        console.print("[yellow]No tracks defined.[/yellow]")
        console.print("Create one with: nanorun track create <name> <directory>")
        return

    table = Table(title="Experiment Tracks")
    table.add_column("Name", style="cyan")
    table.add_column("Directory")
    table.add_column("Description")

    for t in tracks:
        table.add_row(
            t.name,
            t.directory,
            t.description or "[dim]-[/dim]",
        )

    console.print(table)


@track.command("delete")
@click.argument("name", shell_complete=complete_track_names)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def track_delete(name: str, yes: bool):
    """Delete a track (removes .track.json, does not delete other files)."""
    t = get_track(name)
    if not t:
        console.print(f"[red]Track '{name}' not found[/red]")
        raise SystemExit(1)

    if not yes:
        from rich.prompt import Confirm
        if not Confirm.ask(f"Delete track '{name}'? (only .track.json will be removed)"):
            return

    t.delete()
    console.print(f"[green]Deleted track '{name}'[/green]")


@track.command("info")
@click.argument("name", shell_complete=complete_track_names)
def track_info(name: str):
    """Show details about a track."""
    t = get_track(name)
    if not t:
        console.print(f"[red]Track '{name}' not found[/red]")
        raise SystemExit(1)

    console.print(f"[bold cyan]{t.name}[/bold cyan]")
    console.print(f"  Directory: {t.directory}")
    if t.description:
        console.print(f"  Description: {t.description}")
    console.print(f"  Created: {t.created_at}")

    # Show files in directory
    dir_path = get_repo_root() / t.directory
    if dir_path.exists():
        py_files = list(dir_path.glob("*.py"))
        console.print(f"  Files: {len(py_files)} Python files")
        for f in py_files[:5]:
            console.print(f"    [dim]{f.name}[/dim]")
        if len(py_files) > 5:
            console.print(f"    [dim]... and {len(py_files) - 5} more[/dim]")


# ============================================================================
# Sync commands
# ============================================================================

@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("-m", "--message", default=None, help="Commit message for local changes")
@click.option("--all", "sync_all", is_flag=True, help="Sync all changes (git add -A)")
@click.option("--no-verify", is_flag=True, help="Skip Python syntax check")
@session_option
def sync(files: tuple, message: str, sync_all: bool, no_verify: bool, session_name):
    """Push local code changes to remote (git commit + push + pull).

    By default, syncs only the specified file(s). Use --all to sync everything.

    Examples:
        nanorun sync experiments/muon/train.py
        nanorun sync experiments/muon/train.py experiments/muon/utils.py
        nanorun sync --all -m "big refactor"
    """
    from .sync import push_code
    from pathlib import Path

    if not files and not sync_all:
        console.print("[red]Specify file(s) to sync, or use --all to sync everything.[/red]")
        console.print("[dim]  nanorun sync experiments/muon/train.py[/dim]")
        console.print("[dim]  nanorun sync --all[/dim]")
        raise SystemExit(1)

    remote = require_session(session_name)

    if sync_all:
        push_code(remote, message, skip_syntax_check=no_verify)
    else:
        repo_root = get_repo_root()
        rel_files = []
        for f in files:
            try:
                rel = str(Path(f).resolve().relative_to(repo_root.resolve()))
            except ValueError:
                console.print(f"[red]File {f} is not inside the repo[/red]")
                raise SystemExit(1)
            rel_files.append(rel)
        push_code(remote, message, skip_syntax_check=no_verify, files=rel_files)


# ============================================================================
# Job commands
# ============================================================================

@cli.group()
def job():
    """Manage experiment jobs (add, sweep, status, cancel, resume)."""
    pass


def _prepare_job_submission(script: str, gpus: int | None, session_name: str):
    """Resolve script path, prompt to sync if needed, and fill in GPU/track defaults.

    Returns (script_rel, gpus, gpu_type, track). Prints status for sync/gpu/track
    exactly as the old inline code did.
    """
    from .sync import has_unsynced_changes, push_code

    script_rel = _resolve_script_path(script)

    if has_unsynced_changes(files=[script_rel]):
        console.print(f"[yellow]{script_rel} not synced with remote.[/yellow]")
        if click.confirm("Sync before adding?", default=True):
            remote = require_session(session_name)
            push_code(remote, message=None, files=[script_rel])
        else:
            console.print("[dim]Proceeding without sync...[/dim]")

    sc = resolve_session_config(session_name)
    gpu_type = sc.gpu_type if sc else "H100"
    if gpus is None:
        gpus = sc.gpu_count if sc else 1
        console.print(f"[dim]GPUs: {gpus} (from session)[/dim]")

    track = infer_track_from_path(script_rel)
    if track:
        console.print(f"[dim]Track: {track}[/dim]")

    return script_rel, gpus, gpu_type, track


def _enqueue_experiment(
    script_rel: str, env_dict: dict, name: str, track: str | None,
    gpus: int, gpu_type: str, prefix: str | None,
    first: bool, auto_start: bool, session_name: str,
):
    """Generate an experiment ID and enqueue via the remote daemon. Returns the result."""
    import time, random
    from .queue import add_to_queue_via_daemon

    # Timestamp + random suffix; local DB row is created later when the daemon sees it.
    experiment_id = int(time.time() * 1000) + random.randint(0, 999)
    return add_to_queue_via_daemon(
        experiment_id, script_rel, env_dict, track, gpus, gpu_type, name, prefix,
        first=first, auto_start=auto_start, session_name=session_name,
    )


@job.command("add")
@click.argument("script", type=click.Path(exists=True))
@click.option("--env", "-e", multiple=True, help="Environment variables (KEY=VALUE)")
@click.option("--name", "-n", default=None, help="Experiment name")
@click.option("--gpus", "-g", default=None, type=int, help="Number of GPUs (defaults to session gpu_count)")
@click.option("--prefix", "-p", default=None, help="Command prefix (e.g., 'nsys profile --trace=cuda,nvtx -o trace')")
@click.option("--first", "-f", is_flag=True, help="Add to front of queue instead of end")
@session_option
def job_add(script: str, env: tuple, name: str, gpus: int | None, prefix: str, first: bool, session_name):
    """Add an experiment to the queue.

    Track is auto-inferred from script path (if directory has .track.json).
    If nothing is currently running, the experiment will start immediately.

    Examples:
        nanorun job add experiments/records/train.py --env LR=0.03
        nanorun job add experiments/lr/train.py --gpus 8
        nanorun job add train.py --prefix 'nsys profile --trace=cuda,nvtx -o trace'
        nanorun job add train.py --first  # Add to front of queue
    """
    script_rel, gpus, gpu_type, track = _prepare_job_submission(script, gpus, session_name)

    env_dict = dict(e.split("=", 1) for e in env) if env else {}
    if not name:
        name = Path(script).stem

    result = _enqueue_experiment(
        script_rel, env_dict, name, track, gpus, gpu_type, prefix,
        first=first, auto_start=True, session_name=session_name,
    )
    if not result.success:
        console.print(f"[red]Failed to add to queue: {result.error}[/red]")
        return

    position_msg = "front" if first else f"position {result.position}"
    console.print(f"[green]Added to queue ({position_msg})[/green]")
    if result.started:
        console.print("[cyan]Nothing was running - started experiment[/cyan]")


@job.command("sweep")
@click.argument("script", type=click.Path(exists=True))
@click.option("--env", "-e", multiple=True, help="Environment variables with sweep values (KEY=V1,V2,V3)")
@click.option("--name", "-n", default=None, help="Sweep name prefix")
@click.option("--gpus", "-g", default=None, type=int, help="Number of GPUs (defaults to session gpu_count)")
@click.option("--prefix", "-p", default=None, help="Command prefix (e.g., 'nsys profile --trace=cuda,nvtx -o trace')")
@session_option
def job_sweep(script: str, env: tuple, name: str, gpus: int | None, prefix: str, session_name):
    """Add a parameter sweep to the queue.

    Track is auto-inferred from script path (if directory has .track.json).
    If nothing is currently running, the first experiment will start immediately.

    Examples:
        nanorun job sweep experiments/lr/train.py --env LR=0.01,0.02,0.03
        nanorun job sweep train.py --env LR=0.01,0.02 --env BETA=0.9,0.95
    """
    from .runner import parse_sweep_env, generate_sweep_configs

    script_rel, gpus, gpu_type, track = _prepare_job_submission(script, gpus, session_name)

    configs = generate_sweep_configs(parse_sweep_env(env))
    console.print(f"[cyan]Adding {len(configs)} configurations to queue[/cyan]")

    exp_name = name if name else Path(script).stem
    added = 0
    started = False
    for i, cfg in enumerate(configs):
        result = _enqueue_experiment(
            script_rel, cfg, exp_name, track, gpus, gpu_type, prefix,
            first=False, auto_start=(i == 0), session_name=session_name,
        )
        if result.success:
            added += 1
            if result.started:
                started = True

    console.print(f"[green]Added {added} experiments to queue[/green]")
    if started:
        console.print("[cyan]Nothing was running - started first experiment[/cyan]")


@job.command("cancel")
@session_option
def job_cancel(session_name):
    """Cancel current experiment and pause queue.

    Use 'job resume' to start the next queued experiment.
    """
    from .runner import cancel_experiment
    cancel_experiment(start_next=False, session_name=session_name)


@job.command("resume")
@session_option
def job_resume(session_name):
    """Resume queue processing (start next if paused)."""
    from .runner import resume_queue
    resume_queue(session_name=session_name)


@job.command("status")
@click.option("--daemon", "-d", is_flag=True, help="Also show daemon status from remote")
@session_option
def job_status(daemon: bool, session_name):
    """Show current job and queue status."""
    from .tracker import get_running_experiments, get_latest_metric
    from .queue import read_queue, get_queue_state, get_daemon_status

    running = get_running_experiments(session_name=session_name)
    queue_state = get_queue_state()
    queued = read_queue(session_name)

    # Currently running experiment (local view)
    if running:
        exp = running[0]
        metric = get_latest_metric(exp.id)
        console.print(f"[bold cyan]Currently Running[/bold cyan] [dim]#{exp.id}[/dim]")
        console.print(f"  Name: {exp.name}")
        console.print(f"  Script: {exp.script}")
        console.print(f"  Window: [dim]{exp.tmux_window}[/dim]")
        if metric:
            progress = f"{metric.step}/{metric.total_steps}" if metric.total_steps else str(metric.step)
            console.print(f"  Progress: {progress}")
            if metric.val_loss:
                console.print(f"  Val Loss: {metric.val_loss:.4f}")
    else:
        console.print("[dim]No experiment currently running (local)[/dim]")

    console.print()

    # Queue status and contents (local view)
    state_color = "green" if queue_state == "active" else "yellow"
    if queued:
        console.print(f"[bold cyan]Queue[/bold cyan] [{state_color}]{queue_state}[/{state_color}] [dim]({len(queued)} pending)[/dim]")
        for i, exp in enumerate(queued[:5], 1):
            env_str = ", ".join(f"{k}={v}" for k, v in exp.env_vars.items()) if exp.env_vars else ""
            console.print(f"  {i}. {exp.script} {env_str}")
        if len(queued) > 5:
            console.print(f"  [dim]... and {len(queued) - 5} more[/dim]")
    else:
        console.print(f"[bold cyan]Queue[/bold cyan] [{state_color}]{queue_state}[/{state_color}] [dim](empty)[/dim]")

    # Daemon status (optional, requires connection)
    if daemon:
        console.print()
        daemon_status = get_daemon_status(session_name=session_name)
        if daemon_status:
            d_status = daemon_status.get("status", "unknown")
            d_color = {"idle": "dim", "running": "green", "paused": "yellow"}.get(d_status, "white")
            console.print(f"[bold cyan]Daemon[/bold cyan] [{d_color}]{d_status}[/{d_color}]")
            if daemon_status.get("current_experiment_id"):
                console.print(f"  Experiment: #{daemon_status.get('current_experiment_id')}")
                console.print(f"  Window: [dim]{daemon_status.get('current_window')}[/dim]")
                if daemon_status.get("current_run_id"):
                    console.print(f"  Run ID: [dim]{daemon_status.get('current_run_id')}[/dim]")
            console.print(f"  Remote queue: {daemon_status.get('queue_length', 0)} pending")

            # GPU processes
            gpu_procs = daemon_status.get("gpu_processes", [])
            if gpu_procs:
                total_mem = sum(p.get("memory_mb", 0) for p in gpu_procs)
                console.print(f"  GPU: [yellow]{len(gpu_procs)} process(es), {total_mem}MB[/yellow]")
            else:
                console.print(f"  GPU: [green]idle[/green]")
        else:
            console.print("[bold cyan]Daemon[/bold cyan] [dim]not connected[/dim]")


@job.command("logs")
@click.option("--tail", "-f", is_flag=True, help="Continuously watch logs")
@click.option("--window", "-w", default=None, help="Specific tmux window to watch")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
@session_option
def job_logs(tail: bool, window: str, lines: int, session_name):
    """View raw experiment output from tmux.

    Useful for debugging errors, warnings, or watching training progress.
    """
    import time
    from .tracker import get_running_experiments

    remote = require_session(session_name)

    # Default to running experiment's window
    if window is None:
        running = get_running_experiments(session_name=session_name)
        if running and running[0].tmux_window:
            window = running[0].tmux_window
        else:
            console.print("[yellow]No running experiment found[/yellow]")
            console.print("[dim]Use --window to specify a tmux window[/dim]")
            return

    if tail:
        console.print("[dim]Press Ctrl+C to stop watching[/dim]")
        try:
            while True:
                output = remote.get_tmux_output(window, lines=lines)
                # Use raw print to preserve ANSI colors from training script
                print("\033[2J\033[H", end="")  # Clear screen
                print(output)
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        output = remote.get_tmux_output(window, lines=lines)
        # Use raw print to preserve ANSI colors from training script
        print(output)


@job.command("ps")
@session_option
def job_ps(session_name):
    """Show GPU processes on remote machine.

    Lists all processes using the GPU via nvidia-smi.
    The daemon will automatically kill orphan GPU processes when idle.
    """
    from .remote_control import get_daemon_client, DaemonError

    daemon = get_daemon_client(session_name)
    if not daemon:
        console.print("[red]No active session[/red]")
        return

    try:
        if not daemon.ensure_running():
            console.print("[red]Could not connect to daemon[/red]")
            return

        procs = daemon.get_gpu_processes()

        if not procs:
            console.print("[green]No GPU processes running[/green]")
            return

        table = Table(title="GPU Processes")
        table.add_column("PID", style="cyan", justify="right")
        table.add_column("Process Name")
        table.add_column("Memory (MB)", justify="right")

        total_mem = 0
        for p in procs:
            table.add_row(
                str(p["pid"]),
                p["name"],
                str(p["memory_mb"]),
            )
            total_mem += p["memory_mb"]

        console.print(table)
        console.print(f"[dim]Total: {len(procs)} process(es), {total_mem}MB GPU memory[/dim]")

    except DaemonError as e:
        console.print(f"[red]Error: {e}[/red]")


@job.group("queue", invoke_without_command=True)
@click.option("--flat", is_flag=True, help="Plain text output (no table, no truncation)")
@click.option("--session", "session_name", default=None, help="Session name (omit to show all sessions)")
@click.pass_context
def job_queue(ctx, flat, session_name):
    """Show or manage the experiment queue."""
    ctx.ensure_object(dict)
    ctx.obj["session_name"] = session_name
    if ctx.invoked_subcommand is not None:
        return

    # Default behavior: show queue
    from .queue import read_queue, get_queue_state

    queued = read_queue(session_name)
    state = get_queue_state()

    console.print(f"[bold]Queue Status:[/bold] {state}")
    console.print()

    if not queued:
        console.print("[dim]Queue is empty[/dim]")
        console.print("Add experiments with: nanorun job add <script>")
        return

    if flat:
        show_session = session_name is None and len(set(e.session_name for e in queued)) > 1
        for i, exp in enumerate(queued, 1):
            env_str = " ".join(f"{k}={v}" for k, v in exp.env_vars.items()) if exp.env_vars else ""
            parts = [f"{i}.", exp.script]
            if env_str:
                parts.append(env_str)
            if show_session and exp.session_name:
                parts.append(f"({exp.session_name})")
            print(" ".join(parts))
        return

    show_session = session_name is None and len(set(e.session_name for e in queued)) > 1
    table = Table(title=f"Queued Experiments ({len(queued)})")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Script", style="cyan")
    table.add_column("Env")
    table.add_column("Track")
    table.add_column("GPUs", justify="right")
    if show_session:
        table.add_column("Session", style="magenta")

    for i, exp in enumerate(queued, 1):
        env_str = ", ".join(f"{k}={v}" for k, v in exp.env_vars.items()) if exp.env_vars else "[dim]-[/dim]"
        row = [
            str(i),
            exp.script,
            env_str[:30] + "..." if len(env_str) > 30 else env_str,
            exp.track or "[dim]-[/dim]",
            f"{exp.gpus}x {exp.gpu_type}",
        ]
        if show_session:
            row.append(exp.session_name or "-")
        table.add_row(*row)

    console.print(table)


@job.command("clear")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@session_option
def job_clear(yes: bool, session_name):
    """Clear all queued experiments."""
    from .queue import clear_queue_via_daemon, read_queue

    queued = read_queue(session_name)
    if len(queued) == 0:
        console.print("[dim]Queue is already empty[/dim]")
        return

    if not yes:
        if not click.confirm(f"Clear {len(queued)} queued experiment(s)?"):
            return

    cleared = clear_queue_via_daemon(session_name=session_name)
    if cleared is not None:
        console.print(f"[green]Cleared {cleared} experiment(s) from queue[/green]")
    else:
        console.print("[red]Failed to clear queue - daemon not reachable[/red]")


@job.command("remove")
@click.argument("index", type=int)
@click.option("--session", "session_name", default=None, help="Session name (omit to use combined queue index)")
def job_remove(index: int, session_name):
    """Remove experiment at INDEX from queue (1-indexed)."""
    from .queue import remove_from_queue_via_daemon, read_queue, _read_session_queue

    queued = read_queue(session_name)
    if index < 1 or index > len(queued):
        console.print(f"[red]Invalid index: {index} (queue has {len(queued)} items)[/red]")
        return

    exp = queued[index - 1]
    target_session = exp.session_name or session_name
    if not target_session:
        console.print("[red]Cannot determine session for this item[/red]")
        return

    # Find the item's index within its own session's queue
    session_queue = _read_session_queue(target_session)
    session_index = None
    for i, sq_exp in enumerate(session_queue):
        if sq_exp.script == exp.script and sq_exp.env_vars == exp.env_vars:
            session_index = i
            break

    if session_index is None:
        console.print("[red]Could not find item in session queue[/red]")
        return

    if remove_from_queue_via_daemon(session_index, session_name=target_session):
        console.print(f"[green]Removed: {exp.script}[/green]")
    else:
        console.print("[red]Failed to remove from queue - daemon not reachable[/red]")


# ============================================================================
# Local daemon commands
# ============================================================================

@cli.group("local")
def local_daemon():
    """Manage local daemon (metrics sync, experiment tracking)."""
    pass


# -----------------------------------------------------------------------------
# Crash notification helpers (used by local daemon status/crashes commands)
# -----------------------------------------------------------------------------

def get_recent_crashes(unseen_only: bool = False, session_name: str = None) -> list:
    """Get recent crash notifications for one or all sessions."""
    from .local_daemon import safe_json_load, get_crashes_file
    if session_name:
        sessions = [session_name]
    else:
        sessions = [sc.name for sc in Config.list_sessions()] or [None]
    all_crashes = []
    for name in sessions:
        crashes = safe_json_load(get_crashes_file(name), default=[])
        if unseen_only:
            crashes = [c for c in crashes if not c.get("seen", False)]
        all_crashes.extend(crashes)
    return all_crashes


def mark_crashes_seen(session_name: str = None):
    """Mark all crashes as seen for one or all sessions."""
    import json
    from .local_daemon import safe_json_load, get_crashes_file
    if session_name:
        sessions = [session_name]
    else:
        sessions = [sc.name for sc in Config.list_sessions()] or [None]
    for name in sessions:
        crashes_file = get_crashes_file(name)
        crashes = safe_json_load(crashes_file, default=None)
        if crashes is None:
            continue
        for crash in crashes:
            crash["seen"] = True
        crashes_file.write_text(json.dumps(crashes, indent=2))


@local_daemon.command("start")
@click.option(
    "--background/--foreground",
    "background",
    default=False,
    help="Run in background or foreground (default)",
)
@click.option("--no-dashboard", is_flag=True, help="Don't start the dashboard server")
@click.option("--dashboard-port", "-p", default=8080, help="Dashboard port (default: 8080)")
def local_daemon_start(background: bool, no_dashboard: bool, dashboard_port: int):
    """Start the local daemon.

    The daemon polls the remote for experiment status and syncs metrics
    to the local database. It automatically detects when experiments
    start (including from queue) and finish.

    Also starts the web dashboard (localhost:8080) unless --no-dashboard is passed.

    Polling intervals: status/queue=1s, metrics=3s, logs=10s
    """
    from .local_daemon import is_daemon_running, LocalMetricsDaemon, get_daemon_pid
    import subprocess

    if is_daemon_running():
        console.print("[yellow]Local daemon is already running[/yellow]")
        console.print("[dim]Use 'nanorun local status' to check state[/dim]")
        return

    if background:
        # Run as background process
        cmd = [sys.executable, "-m", "nanorun.local_daemon"]
        if no_dashboard:
            cmd.append("--no-dashboard")
        if dashboard_port != 8080:
            cmd.extend(["--dashboard-port", str(dashboard_port)])
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Wait for daemon to start (may take a moment to write PID file)
        import time
        for _ in range(10):
            time.sleep(0.3)
            if is_daemon_running():
                pid = get_daemon_pid()
                console.print(f"[green]Local daemon started (PID: {pid})[/green]")
                return

        console.print("[red]Failed to start local daemon[/red]")
    else:
        # Run in foreground (blocking)
        console.print("[cyan]Starting local daemon in foreground...[/cyan]")
        console.print("[dim]Press Ctrl+C to stop[/dim]")
        daemon_instance = LocalMetricsDaemon(
            dashboard_port=dashboard_port, no_dashboard=no_dashboard
        )
        daemon_instance.run()


@local_daemon.command("stop")
def local_daemon_stop():
    """Stop the local daemon."""
    from .local_daemon import is_daemon_running, stop_daemon, get_daemon_pid

    if not is_daemon_running():
        console.print("[yellow]Local daemon is not running[/yellow]")
        return

    pid = get_daemon_pid()
    if stop_daemon():
        console.print(f"[green]Local daemon stopped (was PID: {pid})[/green]")
    else:
        console.print("[red]Failed to stop local daemon[/red]")


@local_daemon.command("status")
def local_daemon_status():
    """Show local daemon status."""
    from .local_daemon import is_daemon_running, get_daemon_pid, SessionState

    running = is_daemon_running()

    if running:
        pid = get_daemon_pid()
        console.print(f"[bold green]Local daemon running[/bold green] [dim](PID: {pid})[/dim]")
    else:
        console.print("[dim]Local daemon not running[/dim]")

    # Show per-session status
    sessions = Config.list_sessions()
    if sessions:
        console.print()
        for sc in sessions:
            state = SessionState.load(sc.name)
            status_color = {"connected": "green", "connecting": "yellow", "disconnected": "red"}.get(state.status, "dim")
            console.print(f"  [bold]{sc.name}[/bold] [{status_color}]{state.status}[/{status_color}]")
            if state.tracking_experiment_id:
                console.print(f"    Tracking: [cyan]experiment #{state.tracking_experiment_id}[/cyan]")
                if state.tracking_run_id:
                    console.print(f"    Log: [dim]{state.tracking_run_id}.txt[/dim]")
            if state.metrics_synced:
                console.print(f"    Metrics synced: {state.metrics_synced}")
            if state.last_error:
                console.print(f"    Last error: [yellow]{state.last_error}[/yellow]")

    # Show recent crashes (across all sessions)
    all_crashes = get_recent_crashes(unseen_only=True)
    if all_crashes:
        console.print(f"\n[bold red]Recent Crashes ({len(all_crashes)} unseen):[/bold red]")
        for crash in all_crashes[-3:]:
            exp_id = crash.get("experiment_id")
            timestamp = crash.get("timestamp", "")[:19]
            console.print(f"  [red]Experiment {exp_id}[/red] failed at {timestamp}")
        console.print(f"\n  [dim]Run 'nanorun local crashes' for full crash logs[/dim]")
        mark_crashes_seen()


@local_daemon.command("logs")
@click.option("--tail", "-t", is_flag=True, help="Follow local daemon events log")
@click.option("-n", "num_lines", default=50, show_default=True, help="Number of lines to show")
@session_option
def local_daemon_logs(tail: bool, num_lines: int, session_name: str):
    """Show local daemon operator events (timestamped).

    Without --session, shows events from all sessions merged by time.
    With --session, shows events for that session only.
    """
    from .local_daemon import get_events_file
    import subprocess

    if session_name:
        # Single session
        events_file = get_events_file(session_name)
        events_file.parent.mkdir(exist_ok=True)
        events_file.touch(exist_ok=True)

        if tail:
            console.print(f"[dim]Tailing {events_file} (Ctrl+C to stop)[/dim]")
            try:
                subprocess.run(["tail", "-n", str(num_lines), "-f", str(events_file)])
            except KeyboardInterrupt:
                pass
            return

        lines = events_file.read_text().splitlines()
    else:
        # All sessions — merge event lines (they're already timestamped)
        all_lines = []
        for sc in Config.list_sessions():
            ef = get_events_file(sc.name)
            if ef.exists():
                all_lines.extend(ef.read_text().splitlines())
        # Sort by timestamp prefix [HH:MM:SS]
        all_lines.sort()
        lines = all_lines

        if tail:
            # For tail across all sessions, show the active session's events or just use daemon.log
            from .local_daemon import PATHS
            daemon_log = PATHS.log_file
            console.print(f"[dim]Tailing daemon log (all sessions). Use --session for single session. (Ctrl+C to stop)[/dim]")
            try:
                subprocess.run(["tail", "-n", str(num_lines), "-f", str(daemon_log)])
            except KeyboardInterrupt:
                pass
            return

    if not lines:
        console.print("[dim]No local daemon events yet[/dim]")
        return

    for line in lines[-num_lines:]:
        console.print(line)


@local_daemon.command("restart")
@click.pass_context
def local_daemon_restart(ctx):
    """Restart the local daemon (stop + start with same options)."""
    ctx.invoke(local_daemon_stop)
    ctx.invoke(local_daemon_start)


@local_daemon.command("crashes")
@click.option("--all", "-a", "show_all", is_flag=True, help="Show all crashes, not just unseen")
@click.argument("experiment_id", required=False, type=int)
def local_daemon_crashes(show_all: bool, experiment_id: int):
    """Show crash logs from failed experiments.

    If EXPERIMENT_ID is provided, shows full crash log for that experiment.
    Otherwise lists recent crashes.
    """
    if experiment_id:
        # Show specific crash
        crashes = get_recent_crashes(unseen_only=False)
        crash = next((c for c in crashes if c.get("experiment_id") == experiment_id), None)

        if not crash:
            console.print(f"[yellow]No crash log found for experiment {experiment_id}[/yellow]")
            console.print("[dim]Try 'nanorun daemon crash-log {id}' to fetch from remote[/dim]")
            return

        console.print(f"\n[bold red]Experiment {experiment_id} Crash Log[/bold red]")
        console.print(f"[dim]Time:[/dim] {crash.get('timestamp', '')[:19]}")

        if crash.get("crash_log_snippet"):
            console.print("\n[dim]--- Crash Output ---[/dim]")
            console.print(crash["crash_log_snippet"])
            console.print("[dim]--- End ---[/dim]")
        else:
            console.print("\n[dim](No crash output captured)[/dim]")
        return

    # List crashes
    crashes = get_recent_crashes(unseen_only=not show_all)

    if not crashes:
        if show_all:
            console.print("[dim]No crash logs recorded[/dim]")
        else:
            console.print("[green]No unseen crashes[/green]")
            console.print("[dim]Use --all to see all recorded crashes[/dim]")
        return

    console.print(f"\n[bold]{'All' if show_all else 'Unseen'} Crashes ({len(crashes)}):[/bold]")
    for crash in crashes:
        exp_id = crash.get("experiment_id")
        timestamp = crash.get("timestamp", "")[:19]
        seen = crash.get("seen", False)

        status = "[dim](seen)[/dim]" if seen else "[red](new)[/red]"
        console.print(f"\n  [bold]Experiment {exp_id}[/bold] {status}")
        console.print(f"    Time: [dim]{timestamp}[/dim]")

    console.print(f"\n[dim]Run 'nanorun local crashes <id>' for full crash log[/dim]")
    mark_crashes_seen()


# ============================================================================
# Remote daemon commands
# ============================================================================

@cli.group()
def daemon():
    """Manage the remote nanorun daemon."""
    pass


@daemon.command("status")
@session_option
def daemon_status(session_name):
    """Show remote daemon status."""
    from .remote_control import get_daemon_client, DaemonError

    client = get_daemon_client(session_name)
    if not client:
        console.print("[red]No active session[/red]")
        return

    # Check if daemon window exists
    result = client.remote.run(
        "tmux list-windows -t nanorun -F '#W' 2>/dev/null | grep -q '^daemon$'",
        timeout=10,
    )
    window_exists = result.success

    if not window_exists:
        console.print("[bold red]Daemon not running[/bold red]")
        console.print("[dim]Start with: nanorun daemon restart[/dim]")
        return

    # Try to get status from daemon
    try:
        status = client.get_status()
        d_status = status.get("status", "unknown")
        d_color = {"idle": "green", "running": "cyan", "paused": "yellow"}.get(d_status, "white")

        console.print(f"[bold green]Daemon running[/bold green]")
        console.print(f"  Status: [{d_color}]{d_status}[/{d_color}]")

        if status.get("current_experiment_id"):
            console.print(f"  Experiment: #{status.get('current_experiment_id')}")
            console.print(f"  Window: [dim]{status.get('current_window')}[/dim]")
            if status.get("current_run_id"):
                console.print(f"  Run ID: [dim]{status.get('current_run_id')}[/dim]")

        console.print(f"  Queue: {status.get('queue_length', 0)} pending")

        # GPU processes
        gpu_procs = status.get("gpu_processes", [])
        if gpu_procs:
            total_mem = sum(p.get("memory_mb", 0) for p in gpu_procs)
            console.print(f"  GPU: [yellow]{len(gpu_procs)} process(es), {total_mem}MB[/yellow]")
            for p in gpu_procs:
                console.print(f"    PID {p['pid']}: {p['name']} ({p['memory_mb']}MB)")
        else:
            console.print(f"  GPU: [green]idle[/green]")

    except DaemonError as e:
        console.print(f"[bold yellow]Daemon window exists but not responsive[/bold yellow]")
        console.print(f"[dim]Error: {e}[/dim]")
        console.print("[dim]Try: nanorun daemon restart[/dim]")


@daemon.command("stop")
@session_option
def daemon_stop(session_name):
    """Stop the remote daemon."""
    from .remote_control import get_daemon_client

    client = get_daemon_client(session_name)
    if not client:
        console.print("[red]No active session[/red]")
        return

    if client.stop_daemon():
        console.print("[green]Daemon stopped[/green]")
    else:
        console.print("[yellow]Daemon was not running[/yellow]")


@daemon.command("restart")
@session_option
def daemon_restart(session_name):
    """Restart the remote daemon (picks up new code)."""
    from .remote_control import get_daemon_client, DaemonError

    client = get_daemon_client(session_name)
    if not client:
        console.print("[red]No active session[/red]")
        return

    try:
        if client.restart_daemon():
            console.print("[green]Daemon restarted successfully[/green]")
        else:
            console.print("[yellow]Daemon started but may not be responsive yet[/yellow]")
    except DaemonError as e:
        console.print(f"[red]Failed to restart daemon: {e}[/red]")


@daemon.command("logs")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
@click.option("--tail", "-f", is_flag=True, help="Continuously watch logs")
@session_option
def daemon_logs(lines: int, tail: bool, session_name):
    """View remote daemon logs from tmux."""
    import time
    from .remote_control import get_daemon_client

    client = get_daemon_client(session_name)
    if not client:
        console.print("[red]No active session[/red]")
        return

    if tail:
        console.print("[dim]Press Ctrl+C to stop watching[/dim]")
        try:
            while True:
                result = client.remote.run(
                    f"tmux capture-pane -t nanorun:daemon -p -S -{lines}",
                    timeout=10,
                )
                print("\033[2J\033[H", end="")  # Clear screen
                print(result.stdout if result.success else "[No output]")
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        result = client.remote.run(
            f"tmux capture-pane -t nanorun:daemon -p -S -{lines}",
            timeout=10,
        )
        if result.success:
            print(result.stdout)
        else:
            console.print("[yellow]Could not capture daemon output[/yellow]")


# ============================================================================
# Hub commands
# ============================================================================

@cli.group()
def hub():
    """Interact with HuggingFace Buckets hub (logs, weights)."""
    pass


@hub.command("status")
def hub_status():
    """Show hub bucket info."""
    from . import hub as hub_mod

    username = hub_mod.check_auth()
    if not username:
        console.print("[red]Not authenticated. Run: huggingface-cli login[/red]")
        return

    console.print(f"[green]Authenticated as:[/green] {username}")
    try:
        info = hub_mod.get_bucket_info()
        console.print(f"[green]Bucket:[/green] {hub_mod.BUCKET_ID}")
        console.print(f"[green]Size:[/green] {info.size:,} bytes")
        console.print(f"[green]Files:[/green] {info.total_files:,}")
    except Exception as e:
        console.print(f"[yellow]Bucket not found or error: {e}[/yellow]")


@hub.command("logs")
@click.argument("experiment_id", type=int)
def hub_logs(experiment_id: int):
    """Fetch and display a log from the hub."""
    from . import hub as hub_mod
    from .tracker import get_experiment
    from .config import Config

    exp = get_experiment(experiment_id)
    if not exp:
        console.print(f"[red]Experiment {experiment_id} not found[/red]")
        return
    if not exp.remote_run_id:
        console.print(f"[red]Experiment {experiment_id} has no run_id yet[/red]")
        return
    if not exp.session_name:
        console.print(f"[red]Experiment {experiment_id} has no session_name[/red]")
        return

    local_path = Config.get_config_dir() / "logs" / exp.session_name / f"{exp.remote_run_id}.txt"
    try:
        hub_mod.download_log(exp.remote_run_id, local_path, exp.session_name)
        print(local_path.read_text())
    except Exception as e:
        console.print(f"[red]Failed to fetch log: {e}[/red]")


@hub.command("weights")
@click.argument("experiment_id", type=int)
@click.option("--download", is_flag=True, help="Download weights locally")
@click.option("--output", "-o", default=None, help="Download directory (default: .nanorun/weights/{experiment_id}/)")
def hub_weights(experiment_id: int, download: bool, output: str):
    """List or download model weights from the hub."""
    from . import hub as hub_mod
    from .tracker import get_experiment
    from .config import Config

    exp = get_experiment(experiment_id)
    if not exp:
        console.print(f"[red]Experiment {experiment_id} not found[/red]")
        return
    if not exp.session_name:
        console.print(f"[red]Experiment {experiment_id} has no session_name[/red]")
        return

    try:
        filenames = hub_mod.list_weights(experiment_id, exp.session_name)
    except Exception as e:
        console.print(f"[red]Failed to list weights: {e}[/red]")
        return

    if not filenames:
        console.print(f"[yellow]No weights found for experiment {experiment_id}[/yellow]")
        return

    if not download:
        console.print(f"[green]Weights for experiment {experiment_id}:[/green]")
        for f in sorted(filenames):
            console.print(f"  {f}")
        console.print(f"\n[dim]Use --download to fetch locally[/dim]")
        return

    out_dir = Path(output) if output else Config.get_config_dir() / "weights" / str(experiment_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(filenames):
        console.print(f"  Downloading {f}...")
        hub_mod.download_weight(experiment_id, f, out_dir / f, exp.session_name)
    console.print(f"[green]Downloaded {len(filenames)} files to {out_dir}[/green]")


# ============================================================================
# Exec command
# ============================================================================

@cli.command("exec")
@click.argument("command", nargs=-1)
@click.option("-t", "--timeout", default=30, type=int, help="Timeout in seconds (0 = no timeout)")
@session_option
def exec_cmd(command: tuple, timeout: int, session_name):
    """Run a command on the remote machine.

    Reads from stdin if no command given (pipe mode):

        echo "nvidia-smi" | nanorun exec

        cat script.sh | nanorun exec

    Or pass the command directly:

        nanorun exec nvidia-smi

        nanorun exec -- ls -la ~/nanorun/logs/
    """
    import sys

    if command:
        cmd_str = " ".join(command)
    elif not sys.stdin.isatty():
        cmd_str = sys.stdin.read().strip()
        if not cmd_str:
            console.print("[red]No command provided via stdin.[/red]")
            raise SystemExit(1)
    else:
        console.print("[red]No command provided. Pass a command or pipe via stdin.[/red]")
        console.print("[dim]Usage: nanorun exec <command>  or  echo 'cmd' | nanorun exec[/dim]")
        raise SystemExit(1)

    remote = require_session(session_name)
    t = timeout if timeout > 0 else None
    result = remote.run(cmd_str, timeout=t)

    if result.stdout:
        sys.stdout.write(result.stdout)
        if not result.stdout.endswith("\n"):
            sys.stdout.write("\n")
    if result.stderr:
        sys.stderr.write(result.stderr)
        if not result.stderr.endswith("\n"):
            sys.stderr.write("\n")

    raise SystemExit(result.returncode)


# ============================================================================
# Dashboard command
# ============================================================================

@cli.command()
@click.option("--port", "-p", default=8080, help="Port to run dashboard on")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def dashboard(port: int, no_browser: bool):
    """Start the web dashboard."""
    from .dashboard import run_dashboard
    run_dashboard(port=port, open_browser=not no_browser)


if __name__ == "__main__":
    cli()
