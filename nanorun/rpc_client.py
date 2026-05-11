"""WebSocket RPC client for communicating with remote nanorun daemon.

Architecture:
  - One SSH tunnel per session, shared between CLI commands and the local daemon.
    The tunnel forwards a random local port to localhost:9321 on the remote.
  - Each consumer (CLI command, local daemon) opens its own WebSocket over the
    shared tunnel.  Multiple WebSocket connections to the same daemon are fine.
  - Tunnel ownership is tracked via a lock file at .nanorun/tunnels/{session}.json.
    The process that starts the tunnel owns it; other processes reuse the port.

Usage (CLI — short-lived):
    with RpcClient(session_config) as client:
        result = client.call(Method.PING)
    # WebSocket closed, tunnel left running for others.

Usage (local daemon — long-lived):
    client = RpcClient(session_config)
    client.connect()
    while running:
        status = client.call(Method.STATUS)
    client.close()
"""

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

from .config import Config, SessionConfig
from .rpc_types import (
    RPC_PORT,
    EventMessage,
    Method,
    Request,
    Response,
    parse_message,
)


class RpcError(Exception):
    """Error returned by the remote daemon in an RPC response."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"RPC error ({code}): {message}")


def _find_free_port() -> int:
    """Find a free local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _port_is_open(port: int) -> bool:
    """Check if a local TCP port is accepting connections."""
    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except (ConnectionRefusedError, OSError, TimeoutError):
        return False


def _pid_is_alive(pid: int) -> bool:
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# =========================================================================
# Tunnel lock file management
# =========================================================================

def _tunnels_dir() -> Path:
    d = Config.get_config_dir() / "tunnels"
    d.mkdir(exist_ok=True)
    return d


def _tunnel_lock_path(session_name: str) -> Path:
    return _tunnels_dir() / f"{session_name}.json"


def _read_tunnel_lock(session_name: str) -> Optional[Dict[str, Any]]:
    """Read tunnel lock file.  Returns {"pid": int, "local_port": int} or None."""
    path = _tunnel_lock_path(session_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if "pid" in data and "local_port" in data:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _write_tunnel_lock(session_name: str, pid: int, local_port: int) -> None:
    path = _tunnel_lock_path(session_name)
    path.write_text(json.dumps({"pid": pid, "local_port": local_port}))


def _remove_tunnel_lock(session_name: str) -> None:
    path = _tunnel_lock_path(session_name)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


# =========================================================================
# SSH Tunnel (shared per session)
# =========================================================================


class SshTunnel:
    """Manages an SSH port-forwarding tunnel (ssh -N -L).

    Before starting a new tunnel, checks if one already exists for this session
    (via a lock file).  If so, reuses the existing tunnel's port.
    """

    def __init__(self, session: SessionConfig, remote_port: int = RPC_PORT):
        self._session = session
        self._remote_port = remote_port
        self._proc: Optional[subprocess.Popen] = None
        self._owns_tunnel: bool = False
        self.local_port: Optional[int] = None

    def start(self) -> int:
        """Start or reuse an SSH tunnel. Returns the local port number.

        Raises ConnectionError if the SSH process exits immediately.
        """
        # Check for an existing shared tunnel.
        existing = _read_tunnel_lock(self._session.name)
        if existing:
            pid = existing["pid"]
            port = existing["local_port"]
            if _pid_is_alive(pid) and _port_is_open(port):
                self.local_port = port
                self._owns_tunnel = False
                return port
            # Stale lock — clean up.
            _remove_tunnel_lock(self._session.name)

        # Start a new tunnel.
        self.local_port = _find_free_port()

        cmd = [
            "ssh",
            "-N",
            "-L",
            f"{self.local_port}:localhost:{self._remote_port}",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
        ]
        if self._session.key_file:
            cmd.extend(["-i", os.path.expanduser(self._session.key_file)])
        for opt in (self._session.ssh_options or []):
            cmd.extend(["-o", opt])
        if self._session.port != 22:
            cmd.extend(["-p", str(self._session.port)])
        cmd.append(f"{self._session.user}@{self._session.host}")

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            # Detach from parent's process group so the tunnel survives CLI exit.
            start_new_session=True,
        )

        # Give SSH time to establish the tunnel.
        time.sleep(0.75)
        if self._proc.poll() is not None:
            stderr = ""
            if self._proc.stderr:
                stderr = self._proc.stderr.read().decode(errors="replace").strip()
            raise ConnectionError(f"SSH tunnel failed: {stderr}")

        self._owns_tunnel = True
        _write_tunnel_lock(self._session.name, self._proc.pid, self.local_port)
        return self.local_port

    def stop(self):
        """Stop the SSH tunnel — only if we own it."""
        if self._owns_tunnel and self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
            _remove_tunnel_lock(self._session.name)
        self._proc = None
        self._owns_tunnel = False
        self.local_port = None

    @property
    def alive(self) -> bool:
        if self._owns_tunnel:
            return self._proc is not None and self._proc.poll() is None
        # Borrowed tunnel — check via lock file.
        if self.local_port is None:
            return False
        existing = _read_tunnel_lock(self._session.name)
        if not existing:
            return False
        return _pid_is_alive(existing["pid"]) and _port_is_open(existing["local_port"])

    def __del__(self):
        self.stop()


def kill_tunnel(session_name: str) -> bool:
    """Kill an existing shared tunnel for a session.  Used during cleanup.

    Returns True if a tunnel was killed.
    """
    existing = _read_tunnel_lock(session_name)
    if not existing:
        return False
    pid = existing["pid"]
    if _pid_is_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    _remove_tunnel_lock(session_name)
    return True


# =========================================================================
# RPC Client
# =========================================================================


class RpcClient:
    """Synchronous WebSocket RPC client for the remote nanorun daemon.

    Lifecycle: connect() -> call() ... -> close().

    The SSH tunnel is shared per session.  close() tears down the WebSocket but
    leaves the tunnel running for other consumers.  Call close(kill_tunnel=True)
    or use kill_tunnel() to also stop the tunnel.
    """

    def __init__(self, session: SessionConfig):
        self._session = session
        self._tunnel: Optional[SshTunnel] = None
        self._ws = None
        self._event_callbacks: List[Callable[[EventMessage], None]] = []

    # ----- connection management -----

    def connect(self, timeout: float = 10.0):
        """Establish (or reuse) SSH tunnel and connect WebSocket.

        Raises ConnectionError if the tunnel or WebSocket handshake fails.
        """
        if self._ws is not None:
            return

        # Get or create the shared tunnel.
        if not self._tunnel or not self._tunnel.alive:
            if self._tunnel:
                self._tunnel.stop()
            self._tunnel = SshTunnel(self._session)
            self._tunnel.start()

        # Connect WebSocket with retry (daemon may still be booting).
        deadline = time.time() + timeout
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                self._ws = ws_connect(
                    f"ws://localhost:{self._tunnel.local_port}",
                    open_timeout=min(3, max(0.5, deadline - time.time())),
                    ping_interval=None,
                    max_size=2**24,  # 16MB — LIST_MAPPINGS can be large
                )
                return
            except Exception as e:
                last_error = e
                time.sleep(0.5)

        raise ConnectionError(
            f"Could not connect to daemon WebSocket: {last_error}"
        )

    def close(self, stop_tunnel: bool = False):
        """Close the WebSocket connection.

        By default, leaves the SSH tunnel running for other consumers.
        Pass stop_tunnel=True to also tear down the tunnel (e.g. on daemon shutdown).
        """
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._tunnel and stop_tunnel:
            self._tunnel.stop()
            self._tunnel = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    # ----- RPC calls -----

    def call(self, method: Method, timeout: float = 30.0, **params) -> Dict[str, Any]:
        """Send an RPC request and wait for the matching response.

        Events received while waiting are dispatched to registered callbacks.

        Returns:
            The ``result`` dict from the Response on success.

        Raises:
            RpcError: on an error response from the daemon.
            ConnectionError: if the WebSocket is not connected or drops.
            TimeoutError: if no response arrives within *timeout* seconds.
        """
        if not self._ws:
            raise ConnectionError("Not connected to daemon")

        request = Request(method=method, params=params)
        try:
            self._ws.send(request.to_json())
        except ConnectionClosed:
            self._ws = None
            raise ConnectionError("WebSocket connection closed")

        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"RPC call {method.value} timed out after {timeout}s"
                )
            try:
                raw = self._ws.recv(timeout=remaining)
            except ConnectionClosed:
                self._ws = None
                raise ConnectionError("WebSocket connection closed")
            except TimeoutError:
                raise TimeoutError(
                    f"RPC call {method.value} timed out after {timeout}s"
                )

            msg = parse_message(raw)

            if isinstance(msg, Response) and msg.id == request.id:
                if msg.error:
                    raise RpcError(msg.error["code"], msg.error["message"])
                return msg.result or {}
            elif isinstance(msg, EventMessage):
                for cb in self._event_callbacks:
                    try:
                        cb(msg)
                    except Exception:
                        pass

    # ----- event listening -----

    def recv_event(self, timeout: float = 5.0) -> Optional[EventMessage]:
        """Block until an event is received or timeout expires.

        Non-event messages (unexpected responses) are silently discarded.
        Returns None on timeout.
        Raises ConnectionError if WebSocket drops.
        """
        if not self._ws:
            raise ConnectionError("Not connected to daemon")
        try:
            raw = self._ws.recv(timeout=timeout)
        except TimeoutError:
            return None
        except ConnectionClosed:
            self._ws = None
            raise ConnectionError("WebSocket connection closed")

        msg = parse_message(raw)
        if isinstance(msg, EventMessage):
            for cb in self._event_callbacks:
                try:
                    cb(msg)
                except Exception:
                    pass
            return msg
        return None

    def on_event(self, callback: Callable[[EventMessage], None]):
        """Register a callback for unsolicited events from the daemon."""
        self._event_callbacks.append(callback)

    # ----- context manager -----

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()  # WebSocket only — tunnel stays.

    def __del__(self):
        self.close()
