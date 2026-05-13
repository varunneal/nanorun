"""Machine setup and provisioning for remote GPU machines (H100, H200, GH200, DGX Spark)."""

import re
from typing import Optional, Tuple

from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel

from .remote_control import RemoteSession, DaemonClient
from .config import Config

console = Console()


def detect_cuda_version(remote: RemoteSession) -> Optional[str]:
    """Detect CUDA version from nvcc (preferred) or nvidia-smi (fallback).

    nvcc --version shows the actual installed CUDA toolkit version.
    nvidia-smi shows the driver's maximum supported version, which may differ.
    """
    # Try nvcc first - this gives the actual installed CUDA toolkit version
    result = remote.run("nvcc --version 2>/dev/null")
    if result.success:
        # Look for "release X.Y" in nvcc output
        match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
        if match:
            version = match.group(1)
            return _cuda_version_to_torch_tag(version)

    # Fall back to nvidia-smi (shows driver's max supported version)
    result = remote.run("nvidia-smi")
    if result.success:
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
        if match:
            version = match.group(1)
            return _cuda_version_to_torch_tag(version)

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


def detect_gpu_type(remote: RemoteSession) -> str:
    """Detect GPU type from nvidia-smi output."""
    result = remote.run("nvidia-smi --query-gpu=name --format=csv,noheader | head -1")
    if result.success:
        name = result.stdout.strip().upper()
        # DGX Spark uses GB10 Grace Blackwell Superchip
        if "GB10" in name or "BLACKWELL" in name or "DGX SPARK" in name:
            return "DGX_SPARK"
        elif "GH200" in name:
            return "GH200"
        elif "H200" in name:
            return "H200"
        elif "H100" in name:
            return "H100"
    return "H100"  # Default fallback


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
    return 1  # Default fallback


def detect_sudo(remote: RemoteSession) -> bool:
    """Check if sudo is available and needed."""
    # Check if we're root
    result = remote.run("id -u")
    if result.success and result.stdout.strip() == "0":
        return False  # We're root, no sudo needed

    # Check if sudo is available
    result = remote.run("which sudo")
    return result.success


TORCH_VERSION = "2.12.0"

# Only these CUDA tags have dedicated PyTorch wheel indexes
_INDEXED_CUDA_TAGS = {"cu126", "cu130", "cu132"}


def get_torch_install_cmd(cuda_version: str) -> str:
    """Get the pip install command for torch with correct CUDA."""
    if cuda_version in _INDEXED_CUDA_TAGS:
        return (
            f"uv pip install torch=={TORCH_VERSION}+{cuda_version} "
            f"--index-url https://download.pytorch.org/whl/{cuda_version}"
        )
    return f"uv pip install torch=={TORCH_VERSION}"


def get_flash_attn_install_cmd(cuda_version: str) -> str:
    """Get the pip install command for flash_attn_3 with correct CUDA/torch wheel."""
    # Tag format: cu130_torch2100 (cuda tag + torch version without dots)
    torch_tag = "torch" + TORCH_VERSION.replace(".", "")
    wheel_tag = f"{cuda_version}_{torch_tag}"
    return (
        f"uv pip install flash_attn_3 "
        f"--find-links https://windreamer.github.io/flash-attention3-wheels/{wheel_tag}"
    )


def run_setup(remote: RemoteSession, auto_yes: bool = False) -> None:
    """Run interactive setup on remote machine."""
    config = Config.load()

    def confirm(prompt: str, default: bool = True) -> bool:
        """Ask for confirmation, or return True if auto_yes is set."""
        if auto_yes:
            return True
        return Confirm.ask(prompt, default=default)

    console.print(Panel.fit(
        "[bold cyan]nanorun setup[/bold cyan]\n"
        "This will set up the remote machine for NanoGPT training.",
        title="Setup"
    ))

    # Step 1: Detect environment
    console.print("\n[bold]Step 1: Detecting environment...[/bold]")

    # Detect CUDA
    cuda_version = detect_cuda_version(remote)
    if cuda_version:
        console.print(f"  [green]CUDA version detected: {cuda_version}[/green]")
    else:
        console.print("  [yellow]Could not detect CUDA version[/yellow]")
        cuda_version = Confirm.ask(
            "Enter CUDA version manually",
            default="cu130"
        )

    # Detect sudo
    needs_sudo = detect_sudo(remote)
    sudo_prefix = "sudo " if needs_sudo else ""
    console.print(f"  [dim]sudo: {'needed' if needs_sudo else 'not needed (running as root)'}[/dim]")

    # Save detected config
    config.session.cuda_version = cuda_version
    config.session.has_sudo = needs_sudo
    config.save()

    # Step 2: System packages
    console.print("\n[bold]Step 2: System packages[/bold]")
    console.print("  [dim]Will install: git curl tmux rsync build-essential python3-dev[/dim]")

    if confirm("  Install system packages?", default=True):
        # Use DEBIAN_FRONTEND to avoid interactive prompts
        apt_cmd = (
            f"DEBIAN_FRONTEND=noninteractive {sudo_prefix}apt-get update && "
            f"DEBIAN_FRONTEND=noninteractive {sudo_prefix}apt-get install -y git curl tmux rsync build-essential python3-dev"
        )
        console.print("  [dim]Running apt-get...[/dim]")
        result = remote.run(apt_cmd, timeout=120)
        if result.success:
            console.print("  [green]System packages installed[/green]")
        else:
            console.print(f"  [red]Failed: {result.stderr}[/red]")
            if not confirm("  Continue anyway?", default=False):
                return

    # Step 3: Install uv
    console.print("\n[bold]Step 3: Install uv[/bold]")

    # Check if uv already installed
    result = remote.run("which uv || ~/.local/bin/uv --version")
    if result.success:
        console.print("  [green]uv already installed[/green]")
    else:
        if confirm("  Install uv?", default=True):
            uv_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
            console.print("  [dim]Installing uv...[/dim]")
            result = remote.run(uv_cmd, timeout=60)
            if result.success:
                console.print("  [green]uv installed[/green]")
            else:
                console.print(f"  [red]Failed: {result.stderr}[/red]")

    # Get repo path for use in subsequent steps
    repo_path = config.session.repo_path.replace("~", "$HOME")

    # Ensure GitHub SSH uses IPv4 (IPv6/NAT64 can hang under load)
    remote.run('mkdir -p ~/.ssh && grep -q "Host github.com" ~/.ssh/config 2>/dev/null '
               '|| printf "Host github.com\\n    AddressFamily inet\\n" >> ~/.ssh/config')

    # Step 4: Clone repo (must happen before venv/packages)
    console.print("\n[bold]Step 4: Clone repository[/bold]")

    # Check if repo exists
    result = remote.run(f"test -d {repo_path} && echo exists")
    if "exists" in result.stdout:
        console.print("  [green]Repository already cloned[/green]")
        if confirm("  Pull latest changes?", default=True):
            result = remote.run_with_agent(
                f"cd {repo_path} && git pull origin main",
                timeout=60,
            )
            if result.success:
                console.print("  [green]Updated[/green]")
            else:
                console.print(f"  [yellow]Pull issue: {result.stderr}[/yellow]")
    else:
        if confirm("  Clone nanorun repository?", default=True):
            from .project_config import get_repo_url
            repo_url = get_repo_url() or "git@github.com:varunneal/nanorun-private.git"
            # Use GIT_SSH_COMMAND to auto-accept GitHub's host key
            clone_cmd = (
                f"GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=accept-new' "
                f"git clone {repo_url} nanorun"
            )
            console.print("  [dim]Cloning...[/dim]")
            result = remote.run_with_agent(clone_cmd, timeout=60)
            if result.success:
                console.print("  [green]Repository cloned[/green]")
            else:
                console.print(f"  [red]Failed: {result.stderr}[/red]")

    # Step 5: Create venv
    console.print("\n[bold]Step 5: Create Python environment[/bold]")

    # Source uv and create venv in repo directory
    venv_cmd = f"export PATH=$HOME/.local/bin:$PATH && cd {repo_path} && uv venv --python 3.12"
    if confirm("  Create Python 3.12 venv?", default=True):
        console.print("  [dim]Creating venv in repo directory...[/dim]")
        result = remote.run(venv_cmd, timeout=60)
        if result.success:
            console.print("  [green]Venv created[/green]")
        else:
            console.print(f"  [yellow]Note: {result.stderr}[/yellow]")

    # Step 6: Install torch
    console.print("\n[bold]Step 6: Install PyTorch[/bold]")
    torch_cmd = get_torch_install_cmd(cuda_version)
    console.print(f"  [dim]{torch_cmd}[/dim]")

    if confirm("  Install PyTorch nightly?", default=True):
        full_cmd = f"export PATH=$HOME/.local/bin:$PATH && cd {repo_path} && source .venv/bin/activate && {torch_cmd}"
        console.print("  [dim]Installing PyTorch (this may take a few minutes)...[/dim]")
        result = remote.run(full_cmd, timeout=300)
        if result.success:
            console.print("  [green]PyTorch installed[/green]")
        else:
            console.print(f"  [red]Failed: {result.stderr}[/red]")

    # Step 7: Install flash_attn_3
    console.print("\n[bold]Step 7: Install flash_attn_3[/bold]")
    flash_cmd = get_flash_attn_install_cmd(cuda_version)
    console.print(f"  [dim]{flash_cmd}[/dim]")

    if confirm("  Install flash_attn_3?", default=True):
        full_cmd = f"export PATH=$HOME/.local/bin:$PATH && cd {repo_path} && source .venv/bin/activate && {flash_cmd}"
        console.print("  [dim]Installing flash_attn_3...[/dim]")
        result = remote.run(full_cmd, timeout=300)
        if result.success:
            console.print("  [green]flash_attn_3 installed[/green]")
        else:
            console.print(f"  [red]Failed: {result.stderr}[/red]")

    # Step 8: Install other deps
    console.print("\n[bold]Step 8: Install dependencies[/bold]")
    deps = "huggingface-hub websockets tqdm numpy kernels setuptools datasets tiktoken"
    console.print(f"  [dim]Will install: {deps}[/dim]")

    if confirm("  Install dependencies?", default=True):
        deps_cmd = f"export PATH=$HOME/.local/bin:$PATH && cd {repo_path} && source .venv/bin/activate && uv pip install {deps}"
        console.print("  [dim]Installing...[/dim]")
        result = remote.run(deps_cmd, timeout=120)
        if result.success:
            console.print("  [green]Dependencies installed[/green]")
        else:
            console.print(f"  [red]Failed: {result.stderr}[/red]")

    # Step 9: HuggingFace auth
    console.print("\n[bold]Step 9: HuggingFace authentication[/bold]")
    console.print("  [dim]Provisions HF token on remote for bucket access[/dim]")

    if confirm("  Set up HuggingFace auth on remote?", default=True):
        from .hub import get_local_token
        hf_token = get_local_token()
        if hf_token:
            # Write token to remote file, then login from it (avoids token in process list)
            hf_cmd = (
                f"mkdir -p ~/.cache/huggingface && "
                f"cat > ~/.cache/huggingface/token && "
                f"export PATH=$HOME/.local/bin:$PATH && cd {repo_path} && source .venv/bin/activate && "
                f"hf auth login --token $(cat ~/.cache/huggingface/token) --add-to-git-credential"
            )
            console.print("  [dim]Logging in to HuggingFace on remote...[/dim]")
            # Use exec_command to pipe token via stdin
            try:
                client = remote._get_client()
                stdin, stdout, stderr = client.exec_command(hf_cmd, timeout=30)
                stdin.write(hf_token)
                stdin.channel.shutdown_write()
                exit_status = stdout.channel.recv_exit_status()
                if exit_status == 0:
                    console.print("  [green]HuggingFace auth configured[/green]")
                else:
                    err = stderr.read().decode('utf-8', errors='replace')
                    console.print(f"  [yellow]HF login issue: {err}[/yellow]")
                    console.print("  [dim]You can run 'hf auth login' on the remote manually[/dim]")
            except Exception as e:
                console.print(f"  [yellow]HF login issue: {e}[/yellow]")
                console.print("  [dim]You can run 'hf auth login' on the remote manually[/dim]")
        else:
            console.print("  [yellow]No local HF token found at ~/.cache/huggingface/token[/yellow]")
            console.print("  [dim]Run 'hf auth login' locally first, then re-run setup[/dim]")

    # Step 10: Download speedrun training data
    console.print("\n[bold]Step 10: Download speedrun training data[/bold]")
    console.print("  [dim]Will run data/cached_fineweb10B.py to download 24 FineWeb-10B shards (~2.4B tokens)[/dim]")

    if confirm("  Download speedrun data?", default=True):
        data_cmd = (
            f"export PATH=$HOME/.local/bin:$PATH && cd {repo_path} && "
            f"source .venv/bin/activate && python data/cached_fineweb10B.py 24"
        )
        console.print("  [dim]Downloading data (this may take a while)...[/dim]")
        result = remote.run(data_cmd, timeout=600)  # 10 min timeout for large downloads
        if result.success:
            console.print("  [green]Speedrun data downloaded[/green]")
        else:
            console.print(f"  [yellow]Data download issue: {result.stderr}[/yellow]")
            console.print("  [dim]You can run this manually later if needed[/dim]")

    # # Step 11: Download slowrun training data
    # console.print("\n[bold]Step 11: Download slowrun training data[/bold]")
    # console.print("  [dim]Will run data/prepare_slowrun_data.py to tokenize 100M train + 10M val tokens[/dim]")
    #
    # if confirm("  Download slowrun data?", default=True):
    #     data_cmd = (
    #         f"export PATH=$HOME/.local/bin:$PATH && cd {repo_path} && "
    #         f"source .venv/bin/activate && python data/prepare_slowrun_data.py"
    #     )
    #     console.print("  [dim]Downloading and tokenizing data (this may take a while)...[/dim]")
    #     result = remote.run(data_cmd, timeout=600)
    #     if result.success:
    #         console.print("  [green]Slowrun data prepared[/green]")
    #     else:
    #         console.print(f"  [yellow]Data preparation issue: {result.stderr}[/yellow]")
    #         console.print("  [dim]You can run this manually later if needed[/dim]")

    # # Step 12: Download parameter golf data
    # console.print("\n[bold]Step 12: Download parameter golf data[/bold]")
    # console.print("  [dim]Downloads FineWeb sp1024 + sp8192 tokenized shards for parameter golf competition[/dim]")
    #
    # if confirm("  Download parameter golf data?", default=False):
    #     # sp1024 from official repo
    #     sp1024_cmd = (
    #         f"export PATH=$HOME/.local/bin:$PATH && cd {repo_path} && "
    #         f"source .venv/bin/activate && "
    #         f"uv pip install -q sentencepiece huggingface-hub && "
    #         f"python experiments/parameter-golf/cached_challenge_fineweb.py --variant sp1024 --train-shards 80"
    #     )
    #     console.print("  [dim]Downloading sp1024 (80 shards, ~8B tokens)...[/dim]")
    #     result = remote.run(sp1024_cmd, timeout=1800)
    #     if result.success:
    #         console.print("  [green]sp1024 data downloaded[/green]")
    #     else:
    #         console.print(f"  [yellow]sp1024 download issue: {result.stderr[:200]}[/yellow]")
    #
    #     # sp8192 from sproos/parameter-golf-tokenizers
    #     sp8192_cmd = (
    #         f"export PATH=$HOME/.local/bin:$PATH && cd {repo_path} && "
    #         f"source .venv/bin/activate && "
    #         f"python -c \""
    #         f"from huggingface_hub import snapshot_download; "
    #         f"snapshot_download('sproos/parameter-golf-tokenizers', "
    #         f"local_dir='experiments/parameter-golf/', "
    #         f"allow_patterns=['datasets/fineweb10B_sp8192/*', 'tokenizers/fineweb_8192_bpe.*']); "
    #         f"print('Done')\""
    #     )
    #     console.print("  [dim]Downloading sp8192 from sproos/parameter-golf-tokenizers...[/dim]")
    #     result = remote.run(sp8192_cmd, timeout=1800)
    #     if result.success:
    #         console.print("  [green]sp8192 data downloaded[/green]")
    #     else:
    #         console.print(f"  [yellow]sp8192 download issue: {result.stderr[:200]}[/yellow]")

    # Step 13: Start remote daemon
    console.print("\n[bold]Step 13: Start remote daemon[/bold]")
    with DaemonClient(remote) as daemon:
        if daemon.is_daemon_running():
            console.print("  [green]Remote daemon already running[/green]")
        else:
            if daemon.restart_daemon():
                console.print("  [green]Remote daemon started[/green]")
            else:
                console.print("  [yellow]Failed to start remote daemon[/yellow]")
                console.print("  [dim]Run 'nanorun daemon restart' manually[/dim]")

    console.print("\n[bold green]Setup complete![/bold green]")


def verify_setup(remote: RemoteSession) -> bool:
    """Verify that the remote machine is properly set up."""
    config = Config.load()
    repo_path = config.session.repo_path if config.session else "~/nanorun"

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
