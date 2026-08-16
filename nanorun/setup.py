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


_GPU_MATCH_ORDER = [
    # (substring_to_match, returned_type) — order matters: longer/more-specific first.
    # Substrings are matched against the uppercased `nvidia-smi --query-gpu=name`
    # product string (the NVIDIA driver's supported-products name, e.g.
    # "NVIDIA RTX 6000 Ada Generation", "Tesla V100-SXM2-32GB").
    # Blackwell datacenter — Grace parts first ("NVIDIA GB200" contains "B200")
    ("GB10", "DGX_SPARK"), ("DGX SPARK", "DGX_SPARK"),
    ("GB300", "GB300"), ("GB200", "GB200"),
    ("B300", "B300"), ("B200", "B200"), ("B100", "BLACKWELL"),
    # Blackwell workstation ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition") —
    # must precede the generic BLACKWELL entry, which they all contain
    ("RTX PRO 6000", "RTX_PRO_6000"), ("RTX PRO 5000", "RTX_PRO_5000"),
    ("RTX PRO 4500", "RTX_PRO_4500"), ("RTX PRO 4000", "RTX_PRO_4000"),
    ("RTX PRO 2000", "RTX_PRO_2000"),
    ("BLACKWELL", "BLACKWELL"),
    # GeForce Blackwell ("NVIDIA GeForce RTX 5090", "... RTX 5090 D")
    ("RTX 5090", "RTX_5090"), ("RTX 5080", "RTX_5080"),
    ("RTX 5070", "RTX_5070"), ("RTX 5060", "RTX_5060"),
    # Hopper — H200 before H20 ("NVIDIA H200" contains "H20")
    ("GH200", "GH200"), ("H200", "H200"), ("H100", "H100"),
    ("H800", "H800"), ("H20", "H20"),
    # Ada Lovelace datacenter — L40S before L40 before L4; L20 before L2
    ("L40S", "L40S"), ("L40", "L40"), ("L20", "L20"), ("L4", "L4"), ("L2", "L2"),
    # Ada Lovelace workstation ("NVIDIA RTX 6000 Ada Generation") — SFF before plain 4000
    ("RTX 6000 ADA", "RTX_6000_ADA"), ("RTX 5880 ADA", "RTX_5880_ADA"),
    ("RTX 5000 ADA", "RTX_5000_ADA"), ("RTX 4500 ADA", "RTX_4500_ADA"),
    ("RTX 4000 SFF ADA", "RTX_4000_SFF_ADA"), ("RTX 4000 ADA", "RTX_4000_ADA"),
    ("RTX 2000 ADA", "RTX_2000_ADA"),
    # GeForce Ada ("NVIDIA GeForce RTX 4090", "... RTX 4090 D")
    ("RTX 4090", "RTX_4090"), ("RTX 4080", "RTX_4080"),
    ("RTX 4070", "RTX_4070"), ("RTX 4060", "RTX_4060"),
    # Ampere workstation ("NVIDIA RTX A6000") — the whole RTX A-series must precede
    # the bare datacenter entries below: "NVIDIA RTX A4000" contains "A40",
    # "NVIDIA RTX A3000" contains "A30", "NVIDIA RTX A1000" contains "A100".
    # Within the block, A4000 precedes A400 for the same reason.
    ("RTX A6000", "A6000"), ("RTX A5500", "A5500"), ("RTX A5000", "A5000"),
    ("RTX A4500", "A4500"), ("RTX A4000", "A4000"), ("RTX A3000", "A3000"),
    ("RTX A2000", "A2000"), ("RTX A1000", "A1000"), ("RTX A400", "A400"),
    # Ampere datacenter — A100 before A10 ("NVIDIA A100" contains "A10")
    ("A100", "A100"), ("A800", "A800"), ("A40", "A40"), ("A30", "A30"), ("A16", "A16"),
    ("A10G", "A10G"), ("A10", "A10"), ("A2", "A2"),
    # GeForce Ampere ("NVIDIA GeForce RTX 3090", "... RTX 3090 Ti")
    ("RTX 3090", "RTX_3090"), ("RTX 3080", "RTX_3080"),
    ("RTX 3070", "RTX_3070"), ("RTX 3060", "RTX_3060"),
    # Turing workstation ("Quadro RTX 6000" — distinct from the Ada RTX 6000 above)
    ("QUADRO RTX 8000", "QUADRO_RTX_8000"), ("QUADRO RTX 6000", "QUADRO_RTX_6000"),
    # Volta / Turing / Pascal datacenter — the "TESLA " prefix is required to avoid
    # matching "Quadro GV100"/"Quadro GP100" and "NVIDIA T400 4GB"
    ("TESLA V100", "V100"), ("TESLA T4", "T4"),
    ("TESLA P100", "P100"), ("TESLA P40", "P40"),
]


def detect_gpu_type(remote: RemoteSession) -> str:
    """Detect GPU type from nvidia-smi output, falling back to MPS on macOS.

    An nvidia-smi product name that matches no entry in `_GPU_MATCH_ORDER` returns
    "UNKNOWN" and warns with the reported name — previously such GPUs were silently
    reported as H100. Pass `--gpu-type` to `nanorun session start` to override.
    """
    result = remote.run("nvidia-smi --query-gpu=name --format=csv,noheader | head -1")
    reported = result.stdout.strip() if result.success else ""
    if reported:
        name = reported.upper()
        for substr, gpu_type in _GPU_MATCH_ORDER:
            if substr in name:
                return gpu_type
    mps_check = remote.run("python3 -c \"import torch; print(torch.backends.mps.is_available())\"")
    if mps_check.success and mps_check.stdout.strip() == "True":
        return "MPS"
    if reported:
        console.print(
            f"[yellow]Warning: unrecognized GPU {reported!r} — recording type as UNKNOWN. "
            "Use --gpu-type to override.[/yellow]"
        )
        return "UNKNOWN"
    console.print(
        "[yellow]Warning: could not detect GPU type (no nvidia-smi output) — "
        "assuming H100. Use --gpu-type to override.[/yellow]"
    )
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


# ─── Bootstrap git auth ──────────────────────────────────────────────────────


def _gather_bootstrap_git_auth() -> tuple[Optional[dict], list[str]]:
    """Collect this machine's GitHub SSH key + git identity for bootstrapping.

    The machine's own local session must push its nanorun/local/* branch, so it
    needs standing GitHub auth (agent forwarding only lives as long as our SSH
    connection). Same trust model as the HF token setup already ships.

    Returns (git_auth dict or None, warnings). Keys: key_name, key_b64, pub_b64,
    user_name, user_email.
    """
    import base64
    import subprocess
    from pathlib import Path

    warnings: list[str] = []

    def _git_config(key: str) -> str:
        r = subprocess.run(
            ["git", "config", "--get", key], capture_output=True, text=True,
        )
        return r.stdout.strip() if r.returncode == 0 else ""

    user_name = _git_config("user.name")
    user_email = _git_config("user.email")
    if not user_name or not user_email:
        warnings.append("local git user.name/user.email not set; commit identity not provisioned")

    # The key ssh would actually offer to github.com
    key_path = None
    r = subprocess.run(
        ["ssh", "-G", "github.com"], capture_output=True, text=True,
    )
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            if line.startswith("identityfile "):
                candidate = Path(line.split(None, 1)[1]).expanduser()
                if candidate.is_file():
                    key_path = candidate
                    break
    if not key_path:
        warnings.append("no SSH key for github.com found on this machine; git auth not provisioned")
        return None, warnings

    pub_path = key_path.with_suffix(key_path.suffix + ".pub")
    git_auth = {
        "key_name": key_path.name,
        "key_b64": base64.b64encode(key_path.read_bytes()).decode(),
        "pub_b64": (
            base64.b64encode(pub_path.read_bytes()).decode()
            if pub_path.is_file() else None
        ),
        "user_name": user_name,
        "user_email": user_email,
    }
    return git_auth, warnings


# Run artifacts a machine that executes its own sessions must never commit.
# Training scripts write logs/{run_id}.txt into the repo root, and the machine
# commits from that same worktree (`nanorun sync --all` runs `git add -A`
# there), so without this the logs ride along into its branch.
LOCAL_GIT_EXCLUDES = ("/logs/",)


def _local_excludes_cmd(repo_path: str) -> str:
    """Idempotent shell adding LOCAL_GIT_EXCLUDES to a repo's .git/info/exclude.

    Kept out of the tracked .gitignore: this is a property of a machine that
    runs experiments, not of the repository everyone shares.
    """
    prepare = (
        f'cd "{repo_path}" && GIT_DIR=$(git rev-parse --git-dir) && '
        f'mkdir -p "$GIT_DIR/info" && EX="$GIT_DIR/info/exclude" && touch "$EX"'
    )
    checks = [
        f"grep -qxF '{pattern}' \"$EX\" || echo '{pattern}' >> \"$EX\""
        for pattern in LOCAL_GIT_EXCLUDES
    ]
    return "\n".join(["set -e", prepare, *checks])


def _bootstrap_seed_session(session, repo_path: str) -> dict:
    """The machine-side local session identity pre-assigned by this device.

    Written to {repo}/.nanorun/sessions/local.json on the machine so that a
    plain `nanorun session start --local` there adopts this workspace_id — the
    same namespace this device's bootstrap session follows via the hub.
    """
    return {
        "name": "local",
        "host": "localhost",
        "user": session.user,
        "port": 0,
        "repo_path": repo_path,
        "tmux_session": "nanorun-local",
        "session_type": "local",
        "gpu_type": session.gpu_type,
        "gpu_count": session.gpu_count,
        "started_at": session.started_at,
        "workspace_id": session.workspace_id,
        "git_branch": f"nanorun/local/{session.workspace_id}",
    }


def probe_machine_local_identity(
    remote: RemoteSession, repo_path: str
) -> tuple[str, Optional[dict]]:
    """Read the machine's local-session identity, if it owns one.

    Returns (home-expanded repo_path, seed dict | None). The seed is whatever
    {repo}/.nanorun/sessions/local.json holds — a previous bootstrap's seed or
    the live config of a local session the machine already ran. A file that
    exists but does not parse reports as an empty dict, so callers treat the
    machine as occupied rather than silently clobbering state they can't read.
    """
    import json

    probe = (
        'H=$(getent passwd $(whoami) 2>/dev/null | cut -d: -f6); '
        '{ [ -n "$H" ] && [ "$H" != "/" ]; } || H="$HOME"; '
        'echo "NANORUN_HOME:$H"; '
        f"cat {repo_path}/.nanorun/sessions/local.json 2>/dev/null"
    )
    result = remote.run(probe, timeout=15)
    home = ""
    seed_text = ""
    lines = result.stdout.splitlines() if result.success else []
    for i, line in enumerate(lines):
        if line.startswith("NANORUN_HOME:"):
            home = line.split(":", 1)[1].strip()
            seed_text = "\n".join(lines[i + 1:]).strip()
            break
    expanded = repo_path
    if repo_path.startswith("~") and home and home != "/":
        expanded = repo_path.replace("~", home, 1)
    if not seed_text:
        return expanded, None
    try:
        seed = json.loads(seed_text)
    except ValueError:
        return expanded, {}
    return expanded, seed if isinstance(seed, dict) else {}


def replace_machine_local_identity(
    remote: RemoteSession,
    session,
    repo_path: str,
    old_workspace_id: str,
    repo_url: Optional[str] = None,
) -> bool:
    """Rebuild the machine around `session`'s fresh local-session identity.

    A retired identity leaves far more than a stale seed behind: the worktree
    sits on `nanorun/local/{old}` with that workspace's commits (and whatever a
    coding agent left uncommitted), `.nanorun/` holds its queue, state and logs,
    and `~/.local/bin/nanorun` is an editable install bound to that tree. So the
    machine is rebuilt rather than patched:

      1. kill every tmux session (daemon, experiments, coding agents) and the
         detached watcher — anything left alive keeps publishing under the
         retired namespace and recreates the directories being moved;
      2. move the whole repository aside to `{repo_path}@{old_workspace_id}` —
         nothing is deleted, and its committed work is already on the old branch;
      3. clone fresh at the original path, carrying the venv and downloaded
         dataset directories over from the archive so the follow-up
         `session setup` stays a fast no-op instead of a multi-GB re-download;
      4. write the new seed, so the machine's `session start --local` adopts it;
      5. reinstall the nanorun CLI, whose editable install still points at the
         tree that was just replaced.

    Steps 1-4 are required; a failure there returns False with the archive left
    intact for recovery. The CLI reinstall is best effort — `session setup`
    installs it again.
    """
    import base64
    import json

    suffix = old_workspace_id or "previous"

    # ── 1-2. Stop everything, move the old worktree aside ──────────────────
    # The watcher runs detached (`python -m nanorun.watcher`, start_new_session),
    # so tmux kill-server does not reach it. Left alive it keeps following the
    # retired hub namespace, and its PID file rides into the archive — the
    # machine would then report "no watcher" and happily start a second one.
    archive_cmd = "\n".join([
        "set -e",
        "tmux kill-server 2>/dev/null || true",
        f'pid=$(cat "{repo_path}/.nanorun/watcher/watcher.pid" 2>/dev/null) '
        '&& [ -n "$pid" ] && kill "$pid" 2>/dev/null || true',
        # `[.]` keeps the pattern from matching the shell running this script.
        'pkill -f "nanorun[.]watcher" 2>/dev/null || true',
        "sleep 1",  # let HUP'd daemon/experiment/watcher processes finish dying
        f'ARCHIVE="{repo_path}@{suffix}"; i=1',
        f'while [ -e "$ARCHIVE" ]; do ARCHIVE="{repo_path}@{suffix}.$i"; i=$((i+1)); done',
        f'if [ -e "{repo_path}" ]; then mv "{repo_path}" "$ARCHIVE"; fi',
        'echo "NANORUN_ARCHIVE:$ARCHIVE"',
    ])
    result = remote.run(archive_cmd, timeout=60)
    archive = ""
    for line in result.stdout.splitlines() if result.success else []:
        if line.startswith("NANORUN_ARCHIVE:"):
            archive = line.split(":", 1)[1].strip()
    if not result.success or not archive:
        console.print("  [red]could not archive the machine's old repository[/red]")
        if result.stderr:
            console.print(f"  [dim]{result.stderr.strip()[:200]}[/dim]")
        return False
    console.print(f"  [dim]old worktree archived to {archive}[/dim]")

    # ── 3. Fresh clone at the original path ────────────────────────────────
    if not repo_url:
        from .project_config import get_repo_url
        repo_url = get_repo_url() or "git@github.com:varunneal/nanorun-private.git"
    clone_cmd = (
        f"GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=accept-new' "
        f"git clone {repo_url} {repo_path}"
    )
    result = remote.run_with_agent(clone_cmd, timeout=180)
    if not result.success:
        console.print("  [red]fresh clone FAILED[/red]")
        if result.stderr:
            console.print(f"  [dim]{result.stderr.strip()[:200]}[/dim]")
        console.print(
            f"  [yellow]the machine's previous worktree is intact at {archive} — "
            f"move it back to {repo_path} to undo.[/yellow]"
        )
        return False
    console.print("  [green]repository re-cloned[/green]")

    # ── 3b-4. Carry the expensive artifacts over, then seed the identity ───
    # Only untracked directories move: the venv, and dataset downloads under
    # data/ (the tracked contents of data/ are all .py files).
    seed = _bootstrap_seed_session(session, repo_path)
    seed_b64 = base64.b64encode(json.dumps(seed, indent=2).encode()).decode()
    carry_cmd = "\n".join([
        "set -e",
        f'ARCHIVE="{archive}"',
        f'if [ -d "$ARCHIVE/.venv" ] && [ ! -e "{repo_path}/.venv" ]; then',
        f'  mv "$ARCHIVE/.venv" "{repo_path}/.venv"',
        "fi",
        f'if [ -d "$ARCHIVE/data" ]; then mkdir -p "{repo_path}/data"; fi',
        'for d in "$ARCHIVE"/data/*/; do',
        '  [ -d "$d" ] || continue',
        '  name=$(basename "$d")',
        f'  [ -e "{repo_path}/data/$name" ] || mv "$d" "{repo_path}/data/$name"',
        "done",
        f"mkdir -p {repo_path}/.nanorun/sessions",
        f"echo '{seed_b64}' | base64 -d > {repo_path}/.nanorun/sessions/local.json",
    ])
    result = remote.run(carry_cmd, timeout=120)
    if not result.success:
        console.print("  [red]could not seed the machine's new local session[/red]")
        if result.stderr:
            console.print(f"  [dim]{result.stderr.strip()[:200]}[/dim]")
        return False
    console.print("  [green]venv + datasets carried over, new identity seeded[/green]")

    # The fresh clone has no .git/info/exclude, and the machine commits from
    # this worktree. Re-apply now rather than waiting for `session setup`, so a
    # run started before setup can't drag logs/ into the new branch.
    remote.run(_local_excludes_cmd(repo_path), timeout=30)

    # ── 5. Rebind the CLI to the new tree (best effort) ────────────────────
    cli_cmd = (
        'UV=$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv"); '
        f'"$UV" tool install -e {repo_path} --force && '
        '{ "$UV" tool update-shell >/dev/null 2>&1 || true; }'
    )
    result = remote.run(cli_cmd, timeout=300)
    if result.success:
        console.print("  [green]nanorun CLI reinstalled[/green]")
    else:
        console.print(
            "  [yellow]nanorun CLI reinstall FAILED — session setup will retry[/yellow]"
        )
    return True


def _gather_agent_auth() -> tuple[dict, list[str]]:
    """Collect Claude Code + Codex credentials from this machine (best effort).

    Claude Code keeps OAuth credentials in the macOS keychain (service
    "Claude Code-credentials") or in ~/.claude/.credentials.json on Linux.
    Codex keeps them in ~/.codex/auth.json. Missing credentials downgrade to a
    warning — the tools are still installed, just not signed in.
    """
    import base64
    import subprocess
    from pathlib import Path

    warnings: list[str] = []
    auth: dict = {}

    creds = None
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            creds = r.stdout.strip()
    except OSError:
        pass  # not macOS
    if not creds:
        creds_file = Path.home() / ".claude" / ".credentials.json"
        if creds_file.is_file():
            creds = creds_file.read_text()
    if creds:
        auth["claude_creds_b64"] = base64.b64encode(creds.encode()).decode()
    else:
        warnings.append(
            "Claude Code credentials not found locally; run 'claude login' on the machine"
        )

    codex_auth = Path.home() / ".codex" / "auth.json"
    if codex_auth.is_file():
        auth["codex_auth_b64"] = base64.b64encode(codex_auth.read_bytes()).decode()
    else:
        warnings.append(
            "Codex auth.json not found locally; run 'codex login' on the machine"
        )

    return auth, warnings


# Shell wrappers dropped on bootstrapped machines. `claude` / `codex` launch
# inside their own persistent tmux session (so an SSH drop doesn't kill the
# agent) and skip permission prompts — these are ephemeral rented boxes.
# Sourced from ~/.bashrc; rewritten on every setup so updates propagate.
# NOTE: plain string, not an f-string — braces here must stay literal.
AGENT_TMUX_WRAPPERS = r"""# nanorun: run coding agents inside a dedicated tmux session.
# Root boxes: both agents refuse their permission-skip flag outright when euid is
# 0, so mark the environment as sandboxed. These are ephemeral, single-tenant
# rented machines, which is the same premise the flags themselves rest on.
_nanorun_agent_prefix() {
    if [ "$(id -u)" -eq 0 ]; then printf 'IS_SANDBOX=1 '; fi
}
_nanorun_agent_tmux() {
    local name="$1"; shift
    local bin="$HOME/.local/bin/$name"
    [ -x "$bin" ] || bin="$name"
    local envpfx
    envpfx="$(_nanorun_agent_prefix)"
    # Already multiplexed, or not a terminal (scripts, nanorun exec): run direct.
    if [ -n "$TMUX" ] || [ ! -t 1 ]; then
        env $envpfx "$bin" "$@"
        return
    fi
    # Headless/one-shot modes print and exit; a tmux session would eat the output.
    case " $* " in
        *" -p "*|*" --print "*|*" exec "*)
            env $envpfx "$bin" "$@"
            return
            ;;
    esac
    if tmux has-session -t "=$name" 2>/dev/null; then
        tmux attach-session -t "=$name"
        return
    fi
    # Hold the pane open when the agent exits non-zero. Without this a failure at
    # startup tears the session down before anything renders, and the only thing
    # left on screen is tmux's bare "[exited]".
    local cmd
    cmd="$envpfx$(printf '%q ' "$bin" "$@")"
    cmd="$cmd; rc=\$?; [ \$rc -eq 0 ] || { printf '\n[nanorun] %s exited with status %s\n' $name \$rc; printf 'Press Enter to close...'; read -r _; }"
    tmux new-session -s "$name" "$cmd"
}
claude() { _nanorun_agent_tmux claude --dangerously-skip-permissions "$@"; }
codex() { _nanorun_agent_tmux codex --dangerously-bypass-approvals-and-sandbox --no-alt-screen "$@"; }
"""


# ─── Setup script generation ─────────────────────────────────────────────────


def _generate_setup_script(
    repo_path: str,
    home_dir: str,
    cuda_version: str,
    sudo_prefix: str,
    hf_token: Optional[str],
    install_cli: bool = False,
    git_auth: Optional[dict] = None,
    agent_auth: Optional[dict] = None,
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
  # Also write to $HF_HOME if set and different (RunPod sets HF_HOME=/workspace/...)
  if [ -n "$HF_HOME" ] && [ "$HF_HOME" != "{home_dir}/.cache/huggingface" ]; then
    mkdir -p "$HF_HOME"
    echo '{token_b64}' | base64 -d > "$HF_HOME/token"
  fi
  echo "STATUS:hf_auth:OK:token written"
) &
PID_HF=$!
"""
    else:
        hf_block = """
echo "STATUS:hf_auth:FAIL:no local HF token"
PID_HF=""
"""

    # nanorun CLI block (bootstrap sessions only). Installed as a uv tool into an
    # isolated venv (~/.local/bin/nanorun) so it never touches the training .venv.
    # Editable install: later git pulls update the CLI code automatically.
    if install_cli:
        cli_block = """
# ── nanorun CLI (parallel; needs uv + cloned repo, independent of .venv) ──
(
  OUT=$(uv tool install -e "$REPO" --force 2>&1)
  if [ $? -eq 0 ]; then
    uv tool update-shell >/dev/null 2>&1 || true
    echo "STATUS:nanorun_cli:OK:installed to ~/.local/bin/nanorun"
  else
    echo "STATUS:nanorun_cli:FAIL:$(echo "$OUT" | tail -3 | tr '\\n' ' ')"
  fi
) &
PID_CLI=$!
"""
    else:
        cli_block = """
PID_CLI=""
"""

    # Git auth block (bootstrap sessions only): standing GitHub access so the
    # machine's own local session can push its nanorun/local/* branch.
    if git_auth:
        key_name = git_auth["key_name"]
        name_sh = git_auth["user_name"].replace("'", "'\\''")
        email_sh = git_auth["user_email"].replace("'", "'\\''")
        pub_write = ""
        if git_auth.get("pub_b64"):
            pub_write = (
                f"echo '{git_auth['pub_b64']}' | base64 -d > "
                f"$HOME_DIR/.ssh/{key_name}.pub && chmod 644 $HOME_DIR/.ssh/{key_name}.pub"
            )
        identity_cmds = ""
        if git_auth["user_name"] and git_auth["user_email"]:
            identity_cmds = (
                f"  git config --global user.name '{name_sh}'\n"
                f"  git config --global user.email '{email_sh}'\n"
            )
        git_block = f"""
# ── git auth (parallel; SSH key + identity for the machine's own sessions) ──
(
  mkdir -p $HOME_DIR/.ssh && chmod 700 $HOME_DIR/.ssh
  if [ ! -f "$HOME_DIR/.ssh/{key_name}" ]; then
    echo '{git_auth['key_b64']}' | base64 -d > $HOME_DIR/.ssh/{key_name}
    chmod 600 $HOME_DIR/.ssh/{key_name}
  fi
  {pub_write}
  if ! grep -qs "IdentityFile ~/.ssh/{key_name}" $HOME_DIR/.ssh/config; then
    printf "Host github.com\\n    IdentityFile ~/.ssh/{key_name}\\n    IdentitiesOnly yes\\n    StrictHostKeyChecking accept-new\\n" >> $HOME_DIR/.ssh/config
  fi
{identity_cmds}  echo "STATUS:git_auth:OK:{key_name} installed"
) &
PID_GIT=$!
"""
    else:
        git_block = """
PID_GIT=""
"""

    # Coding agents block (bootstrap sessions only, best effort): install Claude
    # Code and Codex into ~/.local/bin and drop in this machine's credentials so
    # both are signed in. agent_auth is a (possibly empty) dict in bootstrap mode.
    if agent_auth is not None:
        claude_cred_write = ""
        if agent_auth.get("claude_creds_b64"):
            claude_cred_write = f"""mkdir -p $HOME_DIR/.claude
  echo '{agent_auth["claude_creds_b64"]}' | base64 -d > $HOME_DIR/.claude/.credentials.json
  chmod 600 $HOME_DIR/.claude/.credentials.json
  [ -f $HOME_DIR/.claude.json ] || echo '{{"hasCompletedOnboarding": true}}' > $HOME_DIR/.claude.json"""
        codex_cred_write = ""
        if agent_auth.get("codex_auth_b64"):
            codex_cred_write = f"""mkdir -p $HOME_DIR/.codex
  echo '{agent_auth["codex_auth_b64"]}' | base64 -d > $HOME_DIR/.codex/auth.json
  chmod 600 $HOME_DIR/.codex/auth.json"""
        agents_block = f"""
# ── Claude Code (parallel; native installer, no sudo) ──
(
  {claude_cred_write}
  if [ -x $HOME_DIR/.local/bin/claude ]; then
    echo "STATUS:claude_code:OK:already installed"
  else
    OUT=$(curl -fsSL https://claude.ai/install.sh | bash 2>&1)
    if [ -x $HOME_DIR/.local/bin/claude ]; then
      echo "STATUS:claude_code:OK:installed"
    else
      echo "STATUS:claude_code:FAIL:$(echo "$OUT" | tail -2 | tr '\\n' ' ')"
    fi
  fi
) &
PID_CLAUDE=$!

# ── Codex CLI (parallel; official installer also updates existing installs) ──
(
  {codex_cred_write}
  mkdir -p $HOME_DIR/.local/bin
  if OUT=$(curl -fsSL https://chatgpt.com/codex/install.sh 2>&1 | sh 2>&1) && [ -x $HOME_DIR/.local/bin/codex ]; then
    echo "STATUS:codex:OK:installed/updated"
  else
    echo "STATUS:codex:FAIL:$(echo "$OUT" | tail -2 | tr '\\n' ' ')"
  fi
) &
PID_CODEX=$!

# ── Shell wrappers: `claude`/`codex` open in their own tmux session ──
WRAP="$HOME_DIR/.nanorun_agent_wrappers.sh"
cat > "$WRAP" <<'NANORUN_AGENT_WRAPPERS_EOF'
{AGENT_TMUX_WRAPPERS}NANORUN_AGENT_WRAPPERS_EOF
for RC in "$HOME_DIR/.bashrc" "$HOME_DIR/.zshrc"; do
  [ "$RC" = "$HOME_DIR/.bashrc" ] || [ -f "$RC" ] || continue
  touch "$RC"
  grep -q nanorun_agent_wrappers "$RC" 2>/dev/null || \\
    printf '\\n[ -f ~/.nanorun_agent_wrappers.sh ] && . ~/.nanorun_agent_wrappers.sh\\n' >> "$RC"
done
echo "STATUS:agent_wrappers:OK:claude/codex wrapped in tmux, permission prompts skipped"
"""
    else:
        agents_block = """
PID_CLAUDE=""
PID_CODEX=""
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

# Reclaim root-owned $HOME/.config (common cloud-image wart; breaks uv's receipt write)
[ -d $HOME_DIR/.config ] && [ ! -w $HOME_DIR/.config ] && {sudo_prefix}chown -R $(whoami) $HOME_DIR/.config

# ── apt (background) ──
(
  if dpkg -l git curl tmux rsync build-essential python3-dev sqlite3 2>/dev/null | grep -c '^ii' | grep -q '^7$'; then
    echo "STATUS:apt:OK:already present"
  else
    OUT=$(DEBIAN_FRONTEND=noninteractive {sudo_prefix}apt-get update -qq -o Acquire::Languages=none 2>&1 && \\
    DEBIAN_FRONTEND=noninteractive {sudo_prefix}apt-get install -y -qq --no-install-recommends \\
      git curl tmux rsync build-essential python3-dev sqlite3 2>&1)
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
{cli_block}
{git_block}
{agents_block}

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
[ -n "$PID_CLI" ] && wait $PID_CLI
[ -n "$PID_GIT" ] && wait $PID_GIT
[ -n "$PID_CLAUDE" ] && wait $PID_CLAUDE
[ -n "$PID_CODEX" ] && wait $PID_CODEX

echo "STATUS:DONE"
"""
    return script


# ─── Setup implementation ─────────────────────────────────────────────────────


def _remote_watcher_running(remote: RemoteSession, repo_path: str) -> bool:
    """Whether the machine already has a live watcher (its PID file names a live process)."""
    pid_files = (
        f"{repo_path}/.nanorun/watcher/watcher.pid",
        f"{repo_path}/.nanorun/local_daemon/daemon.pid",  # pre-rename watchers
    )
    check = " || ".join(
        f'{{ pid=$(cat {p} 2>/dev/null) && kill -0 "$pid" 2>/dev/null; }}'
        for p in pid_files
    )
    r = remote.run(f"( {check} ) && echo RUNNING", timeout=15)
    return r.success and "RUNNING" in r.stdout


class SetupFailure:
    """A non-fatal failure that gets reported at the end."""
    def __init__(self, step: str, detail: str):
        self.step = step
        self.detail = detail


def run_setup(remote: RemoteSession, auto_yes: bool = False, bootstrap: bool = False) -> None:
    """Run fast, non-interactive setup on remote machine.

    Ships a single bash script to the remote that handles all parallelism natively.
    Only 3 SSH round-trips: detect environment, run setup script, start daemon.

    In bootstrap mode the nanorun CLI is also installed on the machine and no
    daemon is started — the machine will run its own local session instead.
    """
    t0 = time.time()
    session = remote.config
    failures: list[SetupFailure] = []

    console.print(Panel.fit(
        "[bold cyan]nanorun setup[/bold cyan]\n"
        + ("Bootstrap provisioning (no daemon, installs nanorun CLI)."
           if bootstrap else "Single-script fast provisioning."),
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

    session.cuda_version = cuda_version
    session.has_sudo = needs_sudo
    Config.save_session(session)

    # Resolve repo path from detected HOME
    repo_path = session.repo_path
    if repo_path.startswith("~"):
        detected_home = detect_info.get("HOME", "").strip()
        if detected_home and detected_home != "/":
            repo_path = repo_path.replace("~", detected_home, 1)
        elif detect_info.get("UID") == "0":
            repo_path = repo_path.replace("~", "/root", 1)
    if repo_path != session.repo_path:
        console.print(f"  [yellow]HOME misconfigured, using: {repo_path}[/yellow]")
        session.repo_path = repo_path
        Config.save_session(session)

    home_dir = str(PurePosixPath(repo_path).parent)

    # Get repo URL
    from .project_config import get_repo_url
    repo_url = get_repo_url() or "git@github.com:varunneal/nanorun-private.git"

    # Get HF token
    from .hub import get_local_token
    hf_token = get_local_token()

    # Bootstrap machines run their own sessions, so they need standing git auth
    # and get signed-in coding agents (Claude Code + Codex, best effort)
    git_auth = None
    agent_auth = None
    if bootstrap:
        git_auth, git_warnings = _gather_bootstrap_git_auth()
        for warning in git_warnings:
            failures.append(SetupFailure("git_auth", warning))
        agent_auth, agent_warnings = _gather_agent_auth()
        for warning in agent_warnings:
            failures.append(SetupFailure("agents", warning))

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

    # ── Machine-local git excludes (bootstrap only) ───────────────────────────
    # The machine commits from its own worktree, so keep its run logs out of the
    # index. Non-fatal: a machine without the exclude still runs experiments.
    if bootstrap:
        r = remote.run(_local_excludes_cmd(repo_path), timeout=15)
        if r.success:
            console.print("  [green]git excludes: logs/ ignored on the machine[/green]")
        else:
            failures.append(SetupFailure("git_excludes", r.stderr[:200]))
            console.print("  [yellow]git excludes: FAILED[/yellow]")

    # ── Seed the machine's local session identity (bootstrap only) ────────────
    # Pre-assign the workspace_id so the machine's `session start --local`
    # adopts it and this device follows the same hub namespace by default.
    if bootstrap:
        import base64 as _b64
        import json as _json

        if not session.workspace_id:
            from .sync import _new_local_workspace_id
            session.workspace_id = _new_local_workspace_id(session)
            Config.save_session(session)

        seed_path = f"{repo_path}/.nanorun/sessions/local.json"
        existing = remote.run(f"cat {seed_path} 2>/dev/null", timeout=15)
        existing_id = None
        if existing.success and existing.stdout.strip():
            try:
                existing_id = _json.loads(existing.stdout).get("workspace_id")
            except ValueError:
                pass
        if existing_id:
            if existing_id != session.workspace_id:
                # The machine already owns an identity — follow it, don't fork it.
                session.workspace_id = existing_id
                Config.save_session(session)
                console.print(
                    f"  [yellow]adopted existing local session namespace: {existing_id}[/yellow]"
                )
            else:
                console.print(f"  [green]local session already seeded:[/green] {existing_id}")
        else:
            seed = _bootstrap_seed_session(session, repo_path)
            seed_b64 = _b64.b64encode(_json.dumps(seed, indent=2).encode()).decode()
            r = remote.run(
                f"mkdir -p {repo_path}/.nanorun/sessions && "
                f"echo '{seed_b64}' | base64 -d > {seed_path}",
                timeout=15,
            )
            if r.success:
                console.print(
                    f"  [green]local session seeded:[/green] namespace {session.workspace_id}"
                )
            else:
                failures.append(SetupFailure("seed", r.stderr[:200]))
                console.print("  [red]local session seed FAILED[/red]")

    # ── Run setup script (1 SSH call, all parallelism inside) ─────────────────
    console.print("\n[bold]Running setup (parallel)...[/bold]")

    script = _generate_setup_script(
        repo_path=repo_path,
        home_dir=home_dir,
        cuda_version=cuda_version,
        sudo_prefix=sudo_prefix,
        hf_token=hf_token,
        install_cli=bootstrap,
        git_auth=git_auth,
        agent_auth=agent_auth,
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

    # ── Start daemon (1 SSH call) — skipped for bootstrap sessions ────────────
    if bootstrap:
        console.print("\n[dim]Bootstrap session — not starting a daemon.[/dim]")
    else:
        console.print("\n[bold]Starting daemon...[/bold]")
        with DaemonClient(remote) as daemon:
            if daemon.is_daemon_running():
                console.print("  [green]daemon: already running[/green]")
            else:
                if daemon.restart_daemon():
                    console.print("  [green]daemon: started[/green]")
                else:
                    failures.append(SetupFailure("daemon", "Failed to start daemon"))
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

    if bootstrap:
        next_steps = [
            f"ssh -A {session.user}@{session.host}",
            f"cd {repo_path}",
            "nanorun session start --local",
        ]
        if not _remote_watcher_running(remote, repo_path):
            next_steps.append("nanorun watcher start --background")
        console.print(Panel.fit(
            "\n".join(next_steps)
            + "\n\n[dim]`nanorun` was installed to ~/.local/bin, which the machine's\n"
              "shell only picks up on login. A shell you already had open there\n"
              "will say 'command not found' — reconnect, or `source ~/.bashrc`.[/dim]",
            title="Next steps (on the machine itself)",
            border_style="cyan",
        ))


def verify_setup(remote: RemoteSession) -> bool:
    """Verify that the remote machine is properly set up."""
    session = remote.config
    configured_path = session.repo_path if session else "~/nanorun"
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

    if getattr(session, "bootstrap", False):
        checks.append((
            "CLI",
            "test -x $HOME/.local/bin/nanorun && echo 'installed'",
            "nanorun CLI",
        ))
        checks.append((
            "GitSSH",
            "ssh -o BatchMode=yes -o ConnectTimeout=10 -T git@github.com 2>&1 "
            "| grep -o 'successfully authenticated'",
            "GitHub SSH auth",
        ))
        checks.append((
            "ClaudeCode",
            "test -x $HOME/.local/bin/claude && echo 'installed'",
            "Claude Code",
        ))
        checks.append((
            "Codex",
            "test -x $HOME/.local/bin/codex && echo 'installed'",
            "Codex CLI",
        ))

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
