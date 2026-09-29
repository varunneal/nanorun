"""Framed shell transport for SSH gateways without exec or TCP forwarding.

RunPod's gateway provides an interactive terminal, rather than a complete SSH
server. Keep frames below the terminal's canonical input limit and encode binary
traffic: intermediate terminals may translate CR/LF even after stty raw.
"""

import base64
import hashlib
import re
import socket
import threading
import time
import uuid
from pathlib import Path

FRAME_BYTES = 768


def _frames(data: bytes) -> bytes:
    return b"".join(base64.b64encode(data[i:i + FRAME_BYTES]) + b"\n"
                    for i in range(0, len(data), FRAME_BYTES))


def _receive_until(channel, marker: bytes, timeout: float | None, initial=b"") -> tuple[bytes, bytes]:
    deadline = None if timeout is None else time.monotonic() + timeout
    output = bytearray(initial)
    while True:
        # Frames/markers contain no CR; tolerate the gateway's CR/LF conversion.
        normalized = bytes(output).replace(b"\r", b"")
        index = normalized.find(marker)
        if index >= 0:
            return normalized[:index], normalized[index + len(marker):]
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("SSH shell did not return its completion marker")
        if channel.recv_ready():
            block = channel.recv(65536)
            if not block:
                raise ConnectionError("SSH shell closed before its completion marker")
            output.extend(block)
        elif channel.closed or channel.exit_status_ready():
            raise ConnectionError("SSH shell closed before its completion marker")
        else:
            time.sleep(0.01)


def open_shell(client, timeout: float = 10, forward_agent: bool = False):
    channel = client.get_transport().open_session(timeout=timeout)
    try:
        channel.settimeout(timeout)
        channel.get_pty(term="dumb", width=200, height=50)
        if forward_agent:
            from paramiko.agent import AgentRequestHandler
            # Keep the handler alive for the lifetime of this channel.
            channel._nanorun_agent_handler = AgentRequestHandler(channel)
        channel.invoke_shell()
        marker = uuid.uuid4().hex
        # Split the marker so an echoed startup command cannot match it.
        channel.sendall((
            "stty -echo; PS1=; PS2=; unset PROMPT_COMMAND; "
            "bind 'set enable-bracketed-paste off' 2>/dev/null; "
            f"printf '\\n%s%s\\n' {marker[:16]} {marker[16:]}\n"
        ).encode())
        _receive_until(channel, f"\n{marker}\n".encode(), timeout)
        return channel
    except BaseException:
        channel.close()
        raise


def run_script(client, script: str, timeout: float | None, forward_agent=False):
    from .remote_control import CommandResult
    data = script.encode()
    marker = uuid.uuid4().hex
    digest = hashlib.sha256(data).hexdigest()
    program = f'''import sys,base64,hashlib,os,tty,tempfile,subprocess,signal
tty.setraw(0)
fd,path=tempfile.mkstemp(prefix="nanorun-script-")
def interrupted(signum,frame): raise SystemExit(1)
signal.signal(signal.SIGHUP,interrupted)
signal.signal(signal.SIGTERM,interrupted)
try:
 remaining={len(data)}
 h=hashlib.sha256()
 with os.fdopen(fd,"wb") as f:
  print("\\n__READY__",flush=True)
  while remaining:
   line=sys.stdin.buffer.readline()
   if not line: raise EOFError("script upload interrupted")
   block=base64.b64decode(line.strip(),validate=True)
   if not block or len(block)>remaining: raise ValueError("invalid script frame")
   f.write(block); h.update(block); remaining-=len(block)
 if h.hexdigest()!={digest!r}: raise ValueError("script checksum mismatch")
 env=os.environ.copy()
 env["PATH"]=os.path.expanduser("~/.local/bin")+":"+env.get("PATH","")
 status=subprocess.run(["bash",path],env=env).returncode
finally:
 os.unlink(path)
print("\\n{marker}_END_"+str(status),flush=True)
'''
    channel, pending = _open_program(
        client, program, min(timeout, 10) if timeout else 10, forward_agent
    )
    try:
        channel.settimeout(timeout)
        for offset in range(0, len(data), FRAME_BYTES * 64):
            channel.sendall(_frames(data[offset:offset + FRAME_BYTES * 64]))
        output, rest = _receive_until(channel, f"\n{marker}_END_".encode(), timeout, initial=pending)
        # Exit status may arrive in a separate packet from the marker prefix.
        while b"\n" not in rest:
            more = channel.recv(1024)
            if not more:
                raise ConnectionError("SSH shell omitted its command exit status")
            rest += more.replace(b"\r", b"")
        status = rest.split(b"\n", 1)[0]
        if not re.fullmatch(rb"-?\d+", status):
            raise ConnectionError("SSH shell returned an invalid exit status")
        return CommandResult(output.decode(errors="replace"), "", int(status))
    finally:
        channel.close()


def _open_program(client, program: str, timeout: float, forward_agent=False):
    channel = open_shell(client, min(timeout, 10), forward_agent)
    marker = uuid.uuid4().hex
    program = program.replace("__READY__", marker)
    encoded = base64.b64encode(program.encode()).decode()
    try:
        # The program stays connected to the terminal on stdin/stdout.
        # Its encoded source is supplied by a substitution, not a stdin pipe.
        if len(encoded) > 3000:
            raise ValueError("SSH relay program exceeds the terminal line limit")
        channel.sendall((f'exec python3 -u -c "$(printf %s {encoded} | base64 -d)"\n').encode())
        _, pending = _receive_until(channel, f"\n{marker}\n".encode(), timeout)
        channel.settimeout(None)
        return channel, pending
    except BaseException:
        channel.close()
        raise


def upload_file(client, local_path: Path, remote_path: str, timeout: float = 600):
    from .remote_control import CommandResult
    size = local_path.stat().st_size
    digest = hashlib.sha256()
    with local_path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    expected = digest.hexdigest()
    program = f'''import sys,base64,hashlib,os,tty
tty.setraw(0)
path={remote_path!r}
remaining={size}
h=hashlib.sha256()
try:
 with open(path,"wb") as f:
  print("\\n__READY__",flush=True)
  while remaining:
   line=sys.stdin.buffer.readline()
   if not line: raise EOFError("upload interrupted")
   data=base64.b64decode(line.strip(),validate=True)
   if not data or len(data)>remaining: raise ValueError("invalid upload frame")
   f.write(data); h.update(data); remaining-=len(data)
 if h.hexdigest()!={expected!r}: raise ValueError("upload checksum mismatch")
 print("\\nNANORUN_UPLOAD_OK",flush=True)
except BaseException:
 try: os.unlink(path)
 except OSError: pass
 raise
'''
    channel, pending = _open_program(client, program, min(timeout, 15))
    try:
        channel.settimeout(timeout)
        with local_path.open("rb") as stream:
            for block in iter(lambda: stream.read(FRAME_BYTES * 64), b""):
                channel.sendall(_frames(block))
        _receive_until(channel, b"\nNANORUN_UPLOAD_OK\n", timeout, initial=pending)
        return CommandResult("Uploaded and verified", "", 0)
    finally:
        channel.close()


class ShellSocket:
    """A socket connected to remote localhost through a framed SSH shell.

    Each WebSocket owns its SSH connection. No local listener, shared tunnel
    state, remote credential copy, or exposed daemon port is required.
    """

    def __init__(self, session, remote_port: int):
        from .remote_control import RemoteSession
        self.remote = RemoteSession(session)
        self.remote_port = remote_port
        self.channel = None
        self.socket = None
        self._peer = None
        self._lock = threading.Lock()

    def start(self, timeout: float = 10) -> socket.socket:
        program = f'''import socket,select,sys,os,base64,tty
tty.setraw(0)
s=socket.create_connection(("127.0.0.1",{self.remote_port}),timeout={timeout!r})
s.settimeout(None)
print("\\n__READY__",flush=True)
buf=b""
while True:
 ready,_,_=select.select([0,s],[],[])
 if 0 in ready:
  data=os.read(0,65536)
  if not data: break
  buf+=data
  while b"\\n" in buf:
   line,buf=buf.split(b"\\n",1)
   s.sendall(base64.b64decode(line.strip(),validate=True))
 if s in ready:
  data=s.recv(49152)
  if not data: break
  for i in range(0,len(data),{FRAME_BYTES}):
   sys.stdout.buffer.write(base64.b64encode(data[i:i+{FRAME_BYTES}])+b"\\n")
  sys.stdout.buffer.flush()
'''
        try:
            self.remote._connect_timeout = timeout
            self.channel, pending = _open_program(self.remote._get_client(), program, timeout)
            # websockets sets TCP_NODELAY; Unix socketpair sockets reject it
            # on macOS. Use a short-lived loopback listener for a TCP pair.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                listener.settimeout(timeout)
                self.socket = socket.create_connection(listener.getsockname(), timeout=timeout)
                self._peer, _ = listener.accept()
                self.socket.settimeout(None)
            threading.Thread(target=self._send, daemon=True).start()
            threading.Thread(target=self._receive, args=(pending,), daemon=True).start()
            return self.socket
        except BaseException:
            self.close()
            raise

    def _send(self):
        try:
            while True:
                block = self._peer.recv(FRAME_BYTES * 64)
                if not block:
                    break
                self.channel.sendall(_frames(block))
        except (OSError, EOFError, AttributeError):
            pass
        finally:
            self.close()

    def _receive(self, pending: bytes):
        try:
            buffer = pending
            while True:
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    self._peer.sendall(base64.b64decode(line.strip(), validate=True))
                block = self.channel.recv(65536)
                if not block:
                    break
                buffer += block
        except (OSError, EOFError, ValueError, AttributeError):
            pass
        finally:
            self.close()

    def close(self):
        with self._lock:
            for sock in (self.socket, self._peer):
                if sock:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    sock.close()
            if self.channel:
                self.channel.close()
            self.remote.close()
