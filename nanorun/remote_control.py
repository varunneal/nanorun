"""Remote control for nanorun - SSH transport and daemon communication.

This module provides:
- RemoteSession: SSH connection management via Paramiko
- DaemonClient: Client for communicating with the remote daemon via WebSocket RPC

The remote daemon (remote_daemon.py) runs on the GPU machine and exposes a
WebSocket RPC server on localhost:9321.  CLI commands and the local daemon
use DaemonClient (backed by RpcClient) to talk to it over an SSH tunnel.
"""

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import paramiko
from rich.console import Console

from .config import Config, SessionConfig
from .rpc_client import RpcClient, RpcError
from .rpc_types import Method


console = Console()


TMUX_SESSION = "nanorun"
DAEMON_WINDOW = "daemon"


# =============================================================================
# SSH Transport Layer
# =============================================================================


@dataclass
class CommandResult:
    """Result of a remote command execution."""
    stdout: str
    stderr: str
    returncode: int

    @property
    def success(self) -> bool:
        return self.returncode == 0


class RemoteSession:
    """Manages SSH connection and tmux sessions on remote machine using Paramiko."""

    def __init__(self, config: SessionConfig):
        self.config = config
        self._client: Optional[paramiko.SSHClient] = None
        self._lock = threading.Lock()
        self._connect_timeout = 10

    def _get_client(self) -> paramiko.SSHClient:
        """Get or create the SSH client connection."""
        with self._lock:
            if self._client is not None:
                # Check if connection is still alive
                transport = self._client.get_transport()
                if transport is not None and transport.is_active():
                    return self._client
                # Connection dead, close and reconnect
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

            # Create new connection
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Build connection kwargs
            connect_kwargs = dict(
                hostname=self.config.host,
                port=self.config.port,
                username=self.config.user,
                timeout=self._connect_timeout,
                allow_agent=True,
                look_for_keys=True,
            )
            if self.config.key_file:
                connect_kwargs["key_filename"] = os.path.expanduser(self.config.key_file)
                connect_kwargs["look_for_keys"] = False

            try:
                client.connect(**connect_kwargs)
            except Exception as e:
                raise ConnectionError(f"Failed to connect to {self.config.user}@{self.config.host}:{self.config.port}: {e}")

            self._client = client
            return client

    def close(self):
        """Close the SSH connection."""
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    def run(self, command: str, timeout: Optional[int] = 30) -> CommandResult:
        """Run a command on the remote machine."""
        try:
            client = self._get_client()

            # Execute command
            stdin, stdout, stderr = client.exec_command(
                command,
                timeout=timeout,
            )

            # Read output
            stdout_str = stdout.read().decode('utf-8', errors='replace')
            stderr_str = stderr.read().decode('utf-8', errors='replace')
            returncode = stdout.channel.recv_exit_status()

            return CommandResult(
                stdout=stdout_str,
                stderr=stderr_str,
                returncode=returncode,
            )
        except paramiko.SSHException as e:
            # Connection issue - close and let next call reconnect
            self.close()
            return CommandResult(
                stdout="",
                stderr=f"SSH error: {e}",
                returncode=-1,
            )
        except TimeoutError:
            return CommandResult(
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                returncode=-1,
            )
        except Exception as e:
            return CommandResult(
                stdout="",
                stderr=str(e),
                returncode=-1,
            )

    def run_with_agent(self, command: str, timeout: Optional[int] = 30) -> CommandResult:
        """Run a command with SSH agent forwarding.

        Uses subprocess SSH for commands that need agent forwarding (e.g., git operations).
        Slower than run() but forwards your local SSH keys.
        """
        ssh_cmd = self._build_subprocess_ssh_command() + [command]
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return CommandResult(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                returncode=-1,
            )
        except Exception as e:
            return CommandResult(
                stdout="",
                stderr=str(e),
                returncode=-1,
            )

    def run_interactive(self, command: str) -> int:
        """Run a command interactively (with TTY).

        Falls back to subprocess SSH for interactive commands.
        """
        ssh_cmd = self._build_subprocess_ssh_command() + ["-t", command]
        result = subprocess.run(ssh_cmd)
        return result.returncode

    def _build_ssh_options(self) -> list[str]:
        """Build common SSH options for subprocess commands."""
        opts = [
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
        ]
        if self.config.key_file:
            opts.extend(["-i", os.path.expanduser(self.config.key_file)])
        for opt in (self.config.ssh_options or []):
            opts.extend(["-o", opt])
        if self.config.port != 22:
            opts.extend(["-p", str(self.config.port)])
        return opts

    def _build_subprocess_ssh_command(self) -> list[str]:
        """Build SSH command for subprocess (used for interactive commands)."""
        cmd = ["ssh", "-A"]
        cmd.extend(self._build_ssh_options())
        cmd.append(f"{self.config.user}@{self.config.host}")
        return cmd

    def test_connection(self) -> Tuple[bool, str]:
        """Test if we can connect to the remote machine."""
        try:
            result = self.run("echo 'nanorun connection test'", timeout=15)
            if result.success and "nanorun connection test" in result.stdout:
                return True, "Connection successful"
            return False, result.stderr or "Connection failed"
        except ConnectionError as e:
            return False, str(e)

    def check_tmux(self) -> bool:
        """Check if tmux is available on remote."""
        result = self.run("which tmux")
        return result.success

    def tmux_session_exists(self) -> bool:
        """Check if our tmux session exists."""
        result = self.run(f"tmux has-session -t {self.config.tmux_session} 2>/dev/null")
        return result.success

    def create_tmux_session(self) -> bool:
        """Create a new tmux session in the repo directory."""
        if self.tmux_session_exists():
            return True
        result = self.run(
            f"tmux new-session -d -s {self.config.tmux_session} -c {self.config.repo_path}"
        )
        return result.success

    def kill_tmux_session(self) -> bool:
        """Kill the tmux session."""
        result = self.run(f"tmux kill-session -t {self.config.tmux_session}")
        return result.success

    def run_in_tmux(self, command: str, window_name: Optional[str] = None) -> bool:
        """Run a command in the tmux session."""
        if not self.tmux_session_exists():
            self.create_tmux_session()

        if window_name:
            # Create new window with name, starting in repo directory
            self.run(
                f"tmux new-window -t {self.config.tmux_session} -n {window_name} -c {self.config.repo_path}"
            )
            target = f"{self.config.tmux_session}:{window_name}"
        else:
            target = self.config.tmux_session

        # Send command to tmux
        escaped_cmd = command.replace("'", "'\\''")
        result = self.run(
            f"tmux send-keys -t {target} '{escaped_cmd}' Enter"
        )
        return result.success

    def get_tmux_windows(self) -> list[str]:
        """Get list of tmux windows."""
        result = self.run(
            f"tmux list-windows -t {self.config.tmux_session} -F '#W' 2>/dev/null"
        )
        if result.success:
            return [w.strip() for w in result.stdout.strip().split("\n") if w.strip()]
        return []

    def get_tmux_output(self, window: Optional[str] = None, lines: int = 50) -> str:
        """Capture recent output from tmux pane."""
        target = self.config.tmux_session
        if window:
            target = f"{target}:{window}"
        result = self.run(
            f"tmux capture-pane -t {target} -p -S -{lines}"
        )
        return result.stdout if result.success else ""

    def attach_tmux(self) -> int:
        """Attach to tmux session interactively."""
        return self.run_interactive(f"tmux attach -t {self.config.tmux_session}")

def _find_sole_connected_session() -> Optional[str]:
    """If exactly one session is connected, return its name."""
    from .local_daemon import SessionState
    sessions = Config.list_sessions()
    connected = [s.name for s in sessions if SessionState.load(s.name).status == "connected"]
    if len(connected) == 1:
        return connected[0]
    return None


def resolve_session_config(session_name: Optional[str] = None) -> Optional[SessionConfig]:
    """Resolve a session config by name, falling back to active session.

    Falls back to the sole connected session if the active session is
    disconnected or unset.

    Args:
        session_name: Explicit session name, or None to use the active session.

    Returns:
        SessionConfig or None if no session found.
    """
    if session_name:
        return Config.load_session(session_name)
    config = Config.load()
    if config.session:
        return config.session
    # No active session — try sole connected session
    sole = _find_sole_connected_session()
    if sole:
        return Config.load_session(sole)
    return None


def get_session(session_name: Optional[str] = None) -> Optional[RemoteSession]:
    """Get a remote session. Uses named session or falls back to active."""
    sc = resolve_session_config(session_name)
    if sc:
        return RemoteSession(sc)
    return None


def require_session(session_name: Optional[str] = None) -> RemoteSession:
    """Get a session or raise an error."""
    session = get_session(session_name)
    if not session:
        if session_name:
            console.print(f"[red]Session '{session_name}' not found.[/red]")
        else:
            console.print("[red]No active session. Run 'nanorun session start <host>' first.[/red]")
        raise SystemExit(1)
    return session


# =============================================================================
# Daemon Client (WebSocket RPC)
# =============================================================================


class DaemonError(Exception):
    """Error communicating with daemon."""
    pass


class DaemonClient:
    """Client for communicating with the remote nanorun daemon via WebSocket RPC.

    Manages an SSH tunnel and WebSocket connection.  Provides the same public API
    as the previous file-based IPC client so existing callers continue to work.
    """

    def __init__(self, remote: RemoteSession):
        self.remote = remote
        self._rpc: Optional[RpcClient] = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ----- internal helpers -----

    def _ensure_connected(self):
        """Ensure RPC client is connected, reconnecting if needed."""
        if self._rpc and self._rpc.connected:
            return
        if self._rpc:
            self._rpc.close()
        self._rpc = RpcClient(self.remote.config)
        self._rpc.connect(timeout=8)

    def _call(self, method: Method, timeout: float = 30.0, **params) -> Dict[str, Any]:
        """Make an RPC call with automatic retry on connection loss.

        Returns the result dict on success.  On RPC error (daemon returned an
        error response) returns ``{"success": False, "error": msg}`` so that
        callers written against the old dict-based API keep working.

        Raises DaemonError on connection/timeout failures.
        """
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                self._ensure_connected()
                return self._rpc.call(method, timeout=timeout, **params)
            except RpcError as e:
                # Server explicitly returned an error — don't retry.
                return {"success": False, "error": e.message}
            except (ConnectionError, TimeoutError, OSError) as e:
                last_error = e
                # Connection lost — tear down and retry once.
                if self._rpc:
                    self._rpc.close()
                    self._rpc = None
                if attempt == 0:
                    continue
        raise DaemonError(str(last_error))

    def close(self, stop_tunnel: bool = False):
        """Close the WebSocket connection.

        By default the shared SSH tunnel is left running for other consumers.
        Pass stop_tunnel=True to also tear down the tunnel.
        """
        if self._rpc:
            self._rpc.close(stop_tunnel=stop_tunnel)
            self._rpc = None

    # ----- daemon lifecycle -----

    def is_daemon_running(self) -> bool:
        """Check if the remote daemon is responsive (RPC ping)."""
        try:
            self._ensure_connected()
            result = self._rpc.call(Method.PING, timeout=5)
            return result.get("pong", False)
        except Exception:
            return False

    def ensure_running(self) -> bool:
        """Ensure daemon is running, starting it if needed.

        Returns True if daemon is running, False if failed to start.
        """
        if self.is_daemon_running():
            return True

        console.print("[dim]Starting remote daemon...[/dim]")

        # Ensure tmux session exists
        self.remote.run(
            f"tmux has-session -t {TMUX_SESSION} 2>/dev/null || "
            f"tmux new-session -d -s {TMUX_SESSION}",
            timeout=10,
        )

        # Start daemon in a new tmux window, passing session name.
        # tee to a persistent log so we still have errors after tmux scrollback
        # is lost on restart. -u makes Python unbuffered so output lands promptly.
        session_name = self.remote.config.name
        daemon_cmd = (
            f"cd ~/nanorun && mkdir -p .daemon && source .venv/bin/activate && "
            f"python -u -m nanorun.remote_daemon --session {session_name} 2>&1 "
            f"| tee -a .daemon/daemon.log"
        )
        result = self.remote.run(
            f"tmux new-window -t {TMUX_SESSION} -n {DAEMON_WINDOW} '{daemon_cmd}'",
            timeout=10,
        )

        if not result.success:
            console.print(f"[red]Failed to start daemon: {result.stderr}[/red]")
            return False

        # Wait for daemon to initialize and become responsive.
        for _ in range(20):  # up to 10 seconds
            time.sleep(0.5)
            if self.is_daemon_running():
                console.print("[green]Daemon started[/green]")
                return True

        console.print("[yellow]Daemon may not have started properly[/yellow]")
        return False

    def stop_daemon(self) -> bool:
        """Stop the remote daemon by killing its tmux window.

        Returns True if daemon was stopped, False if not running.
        """
        result = self.remote.run(
            f"tmux list-windows -t {TMUX_SESSION} -F '#W' 2>/dev/null "
            f"| grep -q '^{DAEMON_WINDOW}$'",
            timeout=10,
        )
        if not result.success:
            return False  # Not running

        result = self.remote.run(
            f"tmux kill-window -t {TMUX_SESSION}:{DAEMON_WINDOW}",
            timeout=10,
        )

        if result.success:
            self.close(stop_tunnel=True)
            time.sleep(0.5)
            return True

        return False

    def restart_daemon(self) -> bool:
        """Restart the remote daemon.

        Returns True if daemon is running after restart.
        """
        console.print("[dim]Restarting remote daemon...[/dim]")

        if self.stop_daemon():
            console.print("[dim]Stopped existing daemon[/dim]")
            time.sleep(1.0)

        return self.ensure_running()

    # =========================================================================
    # High-level API
    # =========================================================================

    def run_experiment(
        self,
        experiment_id: int,
        script: str,
        env_vars: Dict[str, str],
        gpus: int = 1,
        gpu_type: str = "H100",
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start an experiment via daemon.

        Returns dict with: success, code_hash, window_name, error
        """
        return self._call(
            Method.RUN,
            experiment_id=experiment_id,
            script=script,
            env_vars=env_vars,
            gpus=gpus,
            gpu_type=gpu_type,
            name=name,
        )

    def add_to_queue(
        self,
        experiment_id: int,
        script: str,
        env_vars: Dict[str, str],
        gpus: int = 1,
        gpu_type: str = "H100",
        name: Optional[str] = None,
        track: Optional[str] = None,
        cmd_prefix: Optional[str] = None,
        first: bool = False,
        auto_start: bool = False,
    ) -> Dict[str, Any]:
        """Add experiment to remote queue.

        Returns dict with: success, position, daemon_status, started
        """
        return self._call(
            Method.QUEUE_ADD,
            experiment_id=experiment_id,
            script=script,
            env_vars=env_vars,
            gpus=gpus,
            gpu_type=gpu_type,
            name=name,
            track=track,
            cmd_prefix=cmd_prefix,
            first=first,
            auto_start=auto_start,
        )

    def cancel(self, pause: bool = False) -> Dict[str, Any]:
        """Cancel current experiment."""
        return self._call(Method.CANCEL, pause=pause)

    def pause(self) -> Dict[str, Any]:
        """Pause queue processing."""
        return self._call(Method.PAUSE)

    def resume(self) -> Dict[str, Any]:
        """Resume queue processing."""
        return self._call(Method.RESUME)

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status including current experiment, queue, and GPU processes."""
        return self._call(Method.STATUS)

    def get_queue(self) -> Dict[str, Any]:
        """Get full queue list."""
        return self._call(Method.QUEUE_LIST)

    def clear_queue(self) -> Dict[str, Any]:
        """Clear all queued experiments."""
        return self._call(Method.QUEUE_CLEAR)

    def remove_from_queue(self, index: int) -> Dict[str, Any]:
        """Remove item at index from queue."""
        return self._call(Method.QUEUE_REMOVE, index=index)

    def set_queue(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Replace entire remote queue with provided items."""
        return self._call(Method.QUEUE_SET, items=items)

    def get_mapping(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get mapping for an experiment. Returns mapping dict or None."""
        response = self._call(Method.GET_MAPPING, experiment_id=experiment_id)
        if response.get("success"):
            return response.get("mapping")
        return None

    def list_mappings(self) -> list:
        """List all experiment mappings on remote."""
        response = self._call(Method.LIST_MAPPINGS)
        return response.get("mappings", [])

    def ping(self) -> bool:
        """Check if daemon is responsive."""
        try:
            response = self._call(Method.PING, timeout=5)
            return response.get("pong", False)
        except DaemonError:
            return False

    def get_gpu_processes(self) -> list:
        """Get list of GPU processes on remote."""
        response = self._call(Method.GPU_PROCESSES, timeout=10)
        return response.get("gpu_processes", [])

    def get_crash_log(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get crash log for an experiment."""
        response = self._call(Method.GET_CRASH_LOG, experiment_id=experiment_id, timeout=30)
        if response.get("success"):
            return response
        return None

    def list_crash_logs(self) -> list:
        """List available crash logs."""
        response = self._call(Method.LIST_CRASH_LOGS, timeout=10)
        return response.get("crash_logs", [])


def get_daemon_client(session_name: Optional[str] = None) -> Optional[DaemonClient]:
    """Get daemon client for a session.

    Args:
        session_name: Explicit session name, or None to use the active session.

    Returns None if session not found.
    """
    remote = get_session(session_name)
    if not remote:
        return None
    return DaemonClient(remote)
