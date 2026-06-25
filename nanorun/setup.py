"""Machine setup and provisioning for remote GPU machines (H100, H200, GH200, DGX Spark)."""

import re
import time
from pathlib import PurePosixPath
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from .remote_control import RemoteSession, DaemonClient
from .config import Config

console = Console()

# ─── Detection helpers ────────────────────────────────────────────────────────


def detect_cuda_version(remote: RemoteSession) -> Optional[str]:
    """Detect CUDA version from nvcc (preferred) or nvidia-smi (fallback)."""
    result = remote.run("nvcc --version 2>/dev/null")
    if result.success:
        match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
        if match:
            return _cuda_version_to_torch_tag(match.group(1))

    result = remote.run("nvidia-smi")
    if result.success:
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
        if match:
            return _cuda_version_to_torch_tag(match.group(1))

    return None


def _cuda_version_to_torch_tag(version: str) -> str:
    """Convert CUDA version string (e.g., '12.6') to torch tag (e.g., 'cu126')."""
    major_minor = version.split(".")
    major = int(major_minor[0])
    minor = int(major_minor[1]) if len(major_minor) > 1 else 0

    if major >= 13 and minor >= 2:
        return "cu132"
    elif major >= 13:
        return "cu130"
    elif major == 12 and minor >= 6:
        return "cu126"
    elif major == 12 and minor >= 4:
        return "cu124"
    else:
        return f"cu{major}{minor}"


def resolve_repo_path(remote: RemoteSession, configured_path: str) -> str:
    """Resolve repo path to an absolute path on the remote."""
    if not configured_path.startswith("~"):
        return configured_path
    result = remote.run("getent passwd $(whoami) | cut -d: -f6")
    if result.success and result.stdout.strip():
        home = result.stdout.strip()
        if home and home != "/":
            return configured_path.replace("~", home, 1)
    result = remote.run("id -u")
    if result.success and result.stdout.strip() == "0":
        return configured_path.replace("~", "/root", 1)
    return configured_path


def detect_gpu_type(remote: RemoteSession) -> str:
    """Detect GPU type from nvidia-smi output."""
    result = remote.run("nvidia-smi --query-gpu=name --format=csv,noheader | head -1")
    if result.success:
        name = result.stdout.strip().upper()
        if "GB10" in name or "DGX SPARK" in name:
            return "DGX_SPARK"
        elif "B200" in name:
            return "B200"
        elif "RTX PRO 6000" in name and "BLACKWELL" in name:
            return "RTX_PRO_6000"
        elif "BLACKWELL" in name or "B100" in name:
            return "BLACKWELL"
        elif "GH200" in name:
            return "GH200"
        elif "H200" in name:
            return "H200"
        elif "H100" in name:
            return "H100"
        elif "A100" in name:
            return "A100"
        elif "L4" in name:
            return "L4"
    return "H100"


def detect_gpu_count(remote: RemoteSession) -> int:
    """Detect number of GPUs from nvidia-smi output."""
    result = remote.run("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
    if result.success:
        try:
            count = int(result.stdout.strip())
            if count > 0:
                return count
        except ValueError:
            pass
    return 1


def detect_sudo(remote: RemoteSession) -> bool:
    """Check if sudo is available and needed."""
    result = remote.run("id -u")
    if result.success and result.stdout.strip() == "0":
        return False
    result = remote.run("which sudo")
    return result.success


# ─── Install commands ─────────────────────────────────────────────────────────

TORCH_VERSION = "2.12.0"
_INDEXED_CUDA_TAGS = {"cu126", "cu130", "cu132"}


def get_torch_install_cmd(cuda_version: str) -> str:
    if cuda_version in _INDEXED_CUDA_TAGS:
        return (
            f"uv pip install torch=={TORCH_VERSION}+{cuda_version} "
            f"--index-url https://download.pytorch.org/whl/{cuda_version}"
        )
    return f"uv pip install torch=={TORCH_VERSION}"


def get_flash_attn_install_cmd(cuda_version: str) -> str:
    torch_tag = "torch" + TORCH_VERSION.replace(".", "")
    wheel_tag = f"{cuda_version}_{torch_tag}"
    return (
        f"uv pip install flash_attn_3 "
        f"--find-links https://windreamer.github.io/flash-attention3-wheels/{wheel_tag}"
    )


# ─── Setup script generation ─────────────────────────────────────────────────


def _generate_setup_script(
    repo_path: str,
    home_dir: str,
    cuda_version: str,
    sudo_prefix: str,
    hf_token: Optional[str],
) -> str:
    """Generate a bash script that runs the entire setup in one shot on the remote.

    Uses background processes (&) and wait to maximize parallelism while
    respecting dependency ordering. Outputs structured status lines that we parse.
    Assumes repo is already cloned (done separately with agent forwarding).
    """
    torch_cmd = get_torch_install_cmd(cuda_version)
    flash_cmd = get_flash_attn_install_cmd(cuda_version)
    deps = "huggingface-hub[hf_xet] websockets tqdm numpy kernels==0.13.0 setuptools datasets tiktoken nvidia-cuda-nvcc"

    # HF auth block (only if token available)
    hf_block = ""
    if hf_token:
        import base64
        token_b64 = base64.b64encode(hf_token.encode()).decode()
        hf_block = f"""
# ── HF auth (parallel, just writes token file — login happens after deps) ──
(
  mkdir -p {home_dir}/.cache/huggingface
  echo '{token_b64}' | base64 -d > {home_dir}/.cache/huggingface/token
  echo "STATUS:hf_auth:OK:token written"
) &
PID_HF=$!
"""
    else:
        hf_block = """
echo "STATUS:hf_auth:FAIL:no local HF token"
PID_HF=""
"""

    script = f"""#!/bin/bash
set -o pipefail

REPO="{repo_path}"
HOME_DIR="{home_dir}"

# Ensure uv and venv python are findable in all subshells
export PATH="$HOME_DIR/.local/bin:/usr/local/bin:$REPO/.venv/bin:$PATH"
ACTIVATE="cd $REPO && source .venv/bin/activate"

# ─── PHASE 1: Foundation (apt || uv+clone) ────────────────────────────────

# SSH config for GitHub
mkdir -p $HOME_DIR/.ssh
grep -q "Host github.com" $HOME_DIR/.ssh/config 2>/dev/null || printf "Host github.com\\n    AddressFamily inet\\n" >> $HOME_DIR/.ssh/config

# Hostname fix
python3 -c "import socket; socket.gethostbyname(socket.gethostname())" 2>/dev/null || \\
  {sudo_prefix}sh -c 'echo "127.0.0.1 $(hostname)" >> /etc/hosts'

# ── apt (background) ──
(
  if dpkg -l git curl tmux rsync build-essential python3-dev 2>/dev/null | grep -c '^ii' | grep -q '^6$'; then
    echo "STATUS:apt:OK:already present"
  else
    OUT=$(DEBIAN_FRONTEND=noninteractive {sudo_prefix}apt-get update -qq -o Acquire::Languages=none 2>&1 && \\
    DEBIAN_FRONTEND=noninteractive {sudo_prefix}apt-get install -y -qq --no-install-recommends \\
      git curl tmux rsync build-essential python3-dev 2>&1)
    if [ $? -eq 0 ]; then echo "STATUS:apt:OK:installed"; else echo "STATUS:apt:FAIL:$(echo "$OUT" | tail -3 | tr '\\n' ' ')"; fi
  fi
) &
PID_APT=$!

# ── uv install (background, only needs curl) ──
(
  if which uv >/dev/null 2>&1 || $HOME_DIR/.local/bin/uv --version >/dev/null 2>&1; then
    echo "STATUS:uv:OK:already installed"
  else
    OUT=$(curl -LsSf https://astral.sh/uv/install.sh 2>&1 | sh 2>&1)
    if [ $? -eq 0 ]; then echo "STATUS:uv:OK:installed"; else echo "STATUS:uv:FAIL:$(echo "$OUT" | tail -2 | tr '\\n' ' ')"; fi
  fi
) &
PID_UV=$!

# Wait for uv (needed for venv)
wait $PID_UV

# Find uv
UV_BIN=$(which uv 2>/dev/null || echo "$HOME_DIR/.local/bin/uv")
export PATH="$(dirname $UV_BIN):$PATH"

# ── Venv ──
if [ -f "$REPO/.venv/bin/python" ]; then
  echo "STATUS:venv:OK:already exists"
else
  OUT=$(cd $REPO && uv venv --python 3.12 2>&1)
  if [ $? -eq 0 ]; then echo "STATUS:venv:OK:created"; else echo "STATUS:venv:FAIL:$(echo "$OUT" | tail -2 | tr '\\n' ' ')"; fi
fi

# Wait for apt (needed for build-essential in some pip installs)
wait $PID_APT

# ─── PHASE 2: Packages + data (maximally parallel) ───────────────────────

# ── torch (background) ──
(
  cd $REPO && source .venv/bin/activate && python -c "import torch; assert torch.__version__.startswith('{TORCH_VERSION}')" 2>/dev/null
  if [ $? -eq 0 ]; then
    echo "STATUS:torch:OK:already installed"
  else
    OUT=$(cd $REPO && source .venv/bin/activate && {torch_cmd} 2>&1)
    if [ $? -eq 0 ]; then echo "STATUS:torch:OK:installed"; else echo "STATUS:torch:FAIL:$(echo "$OUT" | tail -3 | tr '\\n' ' ')"; fi
  fi
) &
PID_TORCH=$!

# ── deps (background) ──
(
  OUT=$(cd $REPO && source .venv/bin/activate && uv pip install {deps} 2>&1)
  if [ $? -eq 0 ]; then echo "STATUS:deps:OK:installed"; else echo "STATUS:deps:FAIL:$(echo "$OUT" | tail -3 | tr '\\n' ' ')"; fi
) &
PID_DEPS=$!

{hf_block}

# Wait for deps (data needs huggingface-hub; hf login needs huggingface-hub)
wait $PID_DEPS

# HF login (quick, needs huggingface-hub from deps)
if [ -f "{home_dir}/.cache/huggingface/token" ]; then
  cd $REPO && source .venv/bin/activate && python -c "from huggingface_hub import login; login(token=open('{home_dir}/.cache/huggingface/token').read().strip(), add_to_git_credential=True)" 2>/dev/null
fi

# ── data download (background, uses xet from deps) ──
(
  if [ -f "$REPO/data/fineweb10B/fineweb_train_000024.bin" ]; then
    echo "STATUS:data:OK:already downloaded"
  else
    OUT=$(cd $REPO && source .venv/bin/activate && HF_XET_HIGH_PERFORMANCE=1 python $REPO/data/cached_fineweb10B.py 24 2>&1)
    if [ $? -eq 0 ]; then echo "STATUS:data:OK:downloaded 24 shards"; else echo "STATUS:data:FAIL:$(echo "$OUT" | tail -3 | tr '\\n' ' ')"; fi
  fi
) &
PID_DATA=$!

# Wait for torch (flash_attn needs it at runtime, install after)
wait $PID_TORCH

# ── flash_attn_3 ──
(
  cd $REPO && source .venv/bin/activate && python -c "import flash_attn_3" 2>/dev/null
  if [ $? -eq 0 ]; then
    echo "STATUS:flash_attn_3:OK:already installed"
  else
    OUT=$(cd $REPO && source .venv/bin/activate && {flash_cmd} 2>&1)
    if [ $? -eq 0 ]; then echo "STATUS:flash_attn_3:OK:installed"; else echo "STATUS:flash_attn_3:FAIL:$(echo "$OUT" | tail -3 | tr '\\n' ' ')"; fi
  fi
) &
PID_FLASH=$!

# ── CUDA symlink (non-critical, quick) ──
(
  CUDA_PKG=$($REPO/.venv/bin/python -c "import nvidia.cu13; import os; print(os.path.dirname(nvidia.cu13.__file__))" 2>/dev/null) && \\
  [ -n "$CUDA_PKG" ] && {sudo_prefix}ln -sfn $CUDA_PKG /usr/local/cuda 2>/dev/null
) &

# Wait for all remaining
wait $PID_FLASH
wait $PID_DATA
[ -n "$PID_HF" ] && wait $PID_HF

echo "STATUS:DONE"
"""
    return script


# ─── Setup implementation ─────────────────────────────────────────────────────


class SetupFailure:
    """A non-fatal failure that gets reported at the end."""
    def __init__(self, step: str, detail: str):
        self.step = step
        self.detail = detail


def run_setup(remote: RemoteSession, auto_yes: bool = False) -> None:
    """Run fast, non-interactive setup on remote machine.

    Ships a single bash script to the remote that handles all parallelism natively.
    Only 3 SSH round-trips: detect environment, run setup script, start daemon.
    """
    t0 = time.time()
    config = Config.load()
    failures: list[SetupFailure] = []

    console.print(Panel.fit(
        "[bold cyan]nanorun setup[/bold cyan]\n"
        "Single-script fast provisioning.",
        title="Setup"
    ))

    # ── Detect environment (1 SSH call) ─────────────────────────────────────────
    console.print("\n[bold]Detecting environment...[/bold]")

    detect_cmd = (
        'echo "CUDA:$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version:\\s*\\K[0-9.]+")"; '
        'echo "NVCC:$(nvcc --version 2>/dev/null | grep -oP "release\\s+\\K[0-9.]+")"; '
        'echo "UID:$(id -u)"; '
        'echo "HOME:$(getent passwd $(whoami) 2>/dev/null | cut -d: -f6)"'
    )
    result = remote.run(detect_cmd, timeout=10)
    detect_info = {}
    if result.success:
        for line in result.stdout.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                detect_info[k.strip()] = v.strip()

    # Parse CUDA version (prefer nvcc over nvidia-smi)
    cuda_version = None
    if detect_info.get("NVCC"):
        cuda_version = _cuda_version_to_torch_tag(detect_info["NVCC"])
    elif detect_info.get("CUDA"):
        cuda_version = _cuda_version_to_torch_tag(detect_info["CUDA"])
    if not cuda_version:
        failures.append(SetupFailure("CUDA detection", "Could not detect CUDA. Defaulting to cu130."))
        cuda_version = "cu130"
    console.print(f"  CUDA: [green]{cuda_version}[/green]")

    # Parse sudo needs
    needs_sudo = detect_info.get("UID") != "0"
    sudo_prefix = "sudo " if needs_sudo else ""
    console.print(f"  sudo: {'needed' if needs_sudo else 'not needed (root)'}")

    config.session.cuda_version = cuda_version
    config.session.has_sudo = needs_sudo
    config.save()

    # Resolve repo path from detected HOME
    repo_path = config.session.repo_path
    if repo_path.startswith("~"):
        detected_home = detect_info.get("HOME", "").strip()
        if detected_home and detected_home != "/":
            repo_path = repo_path.replace("~", detected_home, 1)
        elif detect_info.get("UID") == "0":
            repo_path = repo_path.replace("~", "/root", 1)
    if repo_path != config.session.repo_path:
        console.print(f"  [yellow]HOME misconfigured, using: {repo_path}[/yellow]")
        config.session.repo_path = repo_path
        config.save()

    home_dir = str(PurePosixPath(repo_path).parent)

    # Get repo URL
    from .project_config import get_repo_url
    repo_url = get_repo_url() or "git@github.com:varunneal/nanorun-private.git"

    # Get HF token
    from .hub import get_local_token
    hf_token = get_local_token()

    # ── Git clone/pull (needs agent forwarding, separate SSH call) ───────────
    console.print("\n[bold]Syncing repository...[/bold]")
    result = remote.run(f"test -d {repo_path} && echo exists")
    if "exists" in result.stdout:
        r = remote.run_with_agent(f"cd {repo_path} && git pull origin main", timeout=60)
        if r.success:
            console.print("  [green]repo: updated[/green]")
        else:
            failures.append(SetupFailure("repo", r.stderr[:200]))
            console.print(f"  [red]repo: pull FAILED[/red]")
    else:
        clone_cmd = (
            f"GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=accept-new' "
            f"git clone {repo_url} {repo_path}"
        )
        r = remote.run_with_agent(clone_cmd, timeout=60)
        if r.success:
            console.print("  [green]repo: cloned[/green]")
        else:
            failures.append(SetupFailure("repo", r.stderr[:200]))
            console.print(f"  [red]repo: clone FAILED[/red]")

    # ── Run setup script (1 SSH call, all parallelism inside) ─────────────────
    console.print("\n[bold]Running setup (parallel)...[/bold]")

    script = _generate_setup_script(
        repo_path=repo_path,
        home_dir=home_dir,
        cuda_version=cuda_version,
        sudo_prefix=sudo_prefix,
        hf_token=hf_token,
    )

    # Ship script via stdin and execute
    try:
        client = remote._get_client()
        stdin, stdout, stderr = client.exec_command("bash -s", timeout=600)
        stdin.write(script)
        stdin.channel.shutdown_write()
        stdout.channel.recv_exit_status()
        stdout_str = stdout.read().decode('utf-8', errors='replace')
        stderr_str = stderr.read().decode('utf-8', errors='replace')

        class _Result:
            success = True
            def __init__(self, out, err):
                self.stdout = out
                self.stderr = err
        result = _Result(stdout_str, stderr_str)
    except Exception as e:
        class _FailResult:
            success = False
            stdout = ""
            def __init__(self, err):
                self.stderr = str(err)
        result = _FailResult(e)

    # Parse structured status lines from output
    if result.success or result.stdout:
        output = result.stdout + result.stderr
        for line in output.splitlines():
            if not line.startswith("STATUS:"):
                continue
            parts = line.split(":", 3)
            if len(parts) < 4:
                continue
            _, name, status, detail = parts
            if name == "DONE":
                continue
            if status == "OK":
                console.print(f"  [green]{name}:[/green] {detail}")
            else:
                failures.append(SetupFailure(name, detail))
                console.print(f"  [red]{name}: FAILED — {detail}[/red]")
    else:
        failures.append(SetupFailure("setup script", f"Script failed: {result.stderr[:200]}"))
        console.print(f"  [red]Setup script failed to run[/red]")

    # ── Start daemon (1 SSH call) ─────────────────────────────────────────────
    console.print("\n[bold]Starting daemon...[/bold]")
    with DaemonClient(remote) as daemon:
        if daemon.is_daemon_running():
            console.print("  [green]daemon: already running[/green]")
        else:
            if daemon.restart_daemon():
                console.print("  [green]daemon: started[/green]")
            else:
                failures.append(SetupFailure("daemon", "Failed to start remote daemon"))
                console.print("  [red]daemon: FAILED to start[/red]")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")

    if failures:
        console.print(Panel(
            "\n".join(f"  [red]✗[/red] [bold]{f.step}[/bold]: {f.detail}" for f in failures),
            title="[bold red]FAILURES (fix manually)[/bold red]",
            border_style="red",
        ))
    else:
        console.print("[bold green]Setup complete — no failures.[/bold green]")


def verify_setup(remote: RemoteSession) -> bool:
    """Verify that the remote machine is properly set up."""
    config = Config.load()
    configured_path = config.session.repo_path if config.session else "~/nanorun"
    repo_path = resolve_repo_path(remote, configured_path)

    console.print(Panel.fit(
        "[bold cyan]Verifying setup...[/bold cyan]",
        title="Verify"
    ))

    all_good = True

    checks = [
        ("nvidia-smi", "nvidia-smi --query-gpu=name --format=csv,noheader", "GPU"),
        ("Python", f"cd {repo_path} && source .venv/bin/activate 2>/dev/null && python --version", "Python"),
        ("PyTorch", f"cd {repo_path} && source .venv/bin/activate 2>/dev/null && python -c 'import torch; print(torch.__version__)'", "PyTorch"),
        ("CUDA", f"cd {repo_path} && source .venv/bin/activate 2>/dev/null && python -c 'import torch; print(torch.cuda.is_available())'", "CUDA available"),
        ("Repo", f"test -d {repo_path} && echo 'exists'", "nanorun repo"),
        ("Data", f"test -f {repo_path}/data/fineweb10B/fineweb_val_000000.bin && echo 'exists'", "speedrun data"),
        ("SlowrunData", f"test -f {repo_path}/fineweb_data/fineweb_train.pt && echo 'exists'", "slowrun data"),
        ("GolfSP1024", f"test -f {repo_path}/experiments/parameter-golf/datasets/fineweb10B_sp1024/fineweb_val_000000.bin && echo 'exists'", "param golf sp1024"),
        ("GolfSP8192", f"test -f {repo_path}/experiments/parameter-golf/datasets/fineweb10B_sp8192/fineweb_val_000000.bin && echo 'exists'", "param golf sp8192"),
        ("HFAuth", f"cd {repo_path} && source .venv/bin/activate 2>/dev/null && hf auth whoami 2>/dev/null | head -1", "HuggingFace auth"),
    ]

    for name, cmd, desc in checks:
        result = remote.run(cmd, timeout=30)
        if result.success and result.stdout.strip():
            output = result.stdout.strip().split("\n")[0][:50]
            console.print(f"  [green]{desc}:[/green] {output}")
        else:
            console.print(f"  [red]{desc}:[/red] not found or failed")
            all_good = False

    if all_good:
        console.print("\n[bold green]All checks passed![/bold green]")
    else:
        console.print("\n[yellow]Some checks failed. Run 'nanorun setup' to fix.[/yellow]")

    return all_good
