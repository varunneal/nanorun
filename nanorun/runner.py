"""Job control for remote experiments."""

import itertools
from typing import Dict, List, Optional

from rich.console import Console

from .remote_control import DaemonError, get_daemon_client
from .tracker import (
    apply_authoritative_experiment_status,
    get_running_experiments,
)
from .queue import set_queue_state

console = Console()


def parse_sweep_env(env_args: tuple) -> Dict[str, List[str]]:
    """Parse sweep environment variables.

    Input: ("LR=0.01,0.02", "BETA=0.9,0.95")
    Output: {"LR": ["0.01", "0.02"], "BETA": ["0.9", "0.95"]}
    """
    result = {}
    for arg in env_args:
        key, values = arg.split("=", 1)
        result[key] = values.split(",")
    return result


def generate_sweep_configs(sweep_vars: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Generate all combinations for a sweep."""
    if not sweep_vars:
        return [{}]

    keys = list(sweep_vars.keys())
    value_lists = [sweep_vars[k] for k in keys]

    configs = []
    for combo in itertools.product(*value_lists):
        config = dict(zip(keys, combo))
        configs.append(config)

    return configs


# =============================================================================
# Job control functions
# =============================================================================

def cancel_experiment(start_next: bool = True, session_name: Optional[str] = None) -> bool:
    """Cancel the currently running experiment via daemon.

    Args:
        start_next: If True, start next queued experiment after cancelling.
                   If False, leave queue paused (for pause behavior).
        session_name: Session name, or None for active session.

    Returns:
        True if an experiment was cancelled, False otherwise.
    """
    # Get local running experiments to update local DB (scoped to this incarnation)
    from .config import Config
    running = get_running_experiments(
        session_name=session_name, session_id=Config.session_id_for(session_name),
    )
    if not running:
        console.print("[yellow]No experiment currently running[/yellow]")
        return False

    exp = running[0]  # Cancel the first (should only be one)
    console.print(f"[cyan]Cancelling experiment: {exp.name}[/cyan]")

    # Cancel via daemon RPC (daemon kills tmux window and updates remote state)
    daemon = get_daemon_client(session_name)
    if not daemon:
        console.print("[red]Could not reach daemon; cancellation was not confirmed[/red]")
        return False

    try:
        with daemon:
            result = daemon.cancel(pause=not start_next)
    except DaemonError as e:
        console.print(f"[red]Could not reach daemon: {e}[/red]")
        return False

    if not result.get("success"):
        console.print(
            f"[yellow]Daemon did not cancel experiment: "
            f"{result.get('error', 'unknown')}[/yellow]"
        )
        return False

    cancelled_id = result.get("cancelled_experiment_id")
    has_correlated_id = (
        isinstance(cancelled_id, int)
        and not isinstance(cancelled_id, bool)
        and cancelled_id > 0
    )
    if has_correlated_id:
        # The acknowledgement names the authority-owned attempt. Any different
        # local "running" projection is stale, not the experiment we cancelled.
        for local_exp in running:
            if local_exp.id != cancelled_id:
                apply_authoritative_experiment_status(local_exp.id, "unknown")
        apply_authoritative_experiment_status(cancelled_id, "cancelled")
        console.print(
            f"[green]Experiment {cancelled_id} marked as cancelled[/green]"
        )
    else:
        console.print(
            "[yellow]Cancellation succeeded remotely, but its experiment ID "
            "was not reported; local status is unchanged[/yellow]"
        )

    # Handle local queue state
    if not start_next:
        set_queue_state("paused")
        console.print("[yellow]Queue paused[/yellow]")

    return True


def resume_queue(session_name: Optional[str] = None) -> bool:
    """Resume queue processing via daemon.

    Args:
        session_name: Session name, or None for active session.

    Returns:
        True if an experiment was started, False otherwise.
    """
    # Update local queue state
    set_queue_state("active")
    console.print("[green]Queue resumed[/green]")

    # Resume via daemon
    daemon = get_daemon_client(session_name)
    if daemon:
        with daemon:
            try:
                result = daemon.resume()
                if result.get("success"):
                    status = result.get("status", "unknown")
                    if status == "running":
                        console.print("[dim]Daemon started next experiment[/dim]")
                        return True
                    else:
                        console.print("[dim]Daemon queue resumed (idle)[/dim]")
            except DaemonError as e:
                console.print(f"[yellow]Could not reach daemon: {e}[/yellow]")

    # Check if experiment already running locally (scoped to this incarnation)
    from .config import Config
    running = get_running_experiments(
        session_name=session_name, session_id=Config.session_id_for(session_name),
    )
    if running:
        console.print(f"[dim]Experiment '{running[0].name}' still running - queue will continue when it finishes[/dim]")
        return False

    return False
