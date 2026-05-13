## nanorun

This is a comprehensive platform for training, speedrunning, and researching LLMs. Handles machine setup, experiment queuing, log collection, experiment lineages. You (or agent) write training scripts, view results from CLI or via the local dashboard (`nanorun dashboard`). Replacement for W&B.

## Setup

### Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- A GitHub account with SSH keys configured (`ssh -T git@github.com` should work)
- A [HuggingFace](https://huggingface.co) account with a write token

### SSH keys

nanorun uses SSH agent forwarding (`ssh -A`) to clone your fork onto remote machines. Your local SSH agent must have a key that can access your GitHub fork:

```bash
# Check your SSH agent has keys loaded
ssh-add -l

# If empty, add your key
ssh-add ~/.ssh/id_ed25519    # or whatever key you use for GitHub

# Verify GitHub access
ssh -T git@github.com
```

### Install

```bash
# Fork this repo on GitHub, then clone your fork
git clone git@github.com:YOUR_USERNAME/nanorun.git
cd nanorun
uv sync
uv tool install -e .
nanorun --help
```

### Configure `nanorun.toml`

Edit `nanorun.toml` at the repo root. You need to set your repo URL and choose a storage backend for logs/weights:

```toml
[repo]
url = "git@github.com:YOUR_USERNAME/nanorun.git"

[hub]
bucket_id = "YOUR_USERNAME/nanorun"
```

- **repo.url** — Your fork's SSH clone URL. Used by `nanorun session setup` to clone the repo on remote machines.

**Storage backends** — nanorun syncs logs and model weights through one of:

**HuggingFace Buckets** (default) — Create a bucket at [huggingface.co/new-bucket](https://huggingface.co/new-bucket), then set `bucket_id` to match. Login locally so nanorun can authenticate:
```bash
uv run huggingface-cli login
```
This writes your token to `~/.cache/huggingface/token`. During `nanorun session setup`, this token is automatically provisioned on the remote machine.

**S3-compatible storage** (AWS, R2, MinIO, GCS) — set `backend = "s3"` in `nanorun.toml` and configure credentials via standard AWS env vars or `~/.aws/credentials`.

**Local filesystem** — set `backend = "local"` and `path = "/mnt/shared/nanorun-logs"` (or any local/shared path). Zero network overhead. Good for machines with shared NFS/mounted storage.

## Quickstart

The recommended path: open Claude Code (or any coding agent) in this repo and let it drive.

1. Complete the setup above (fork, install, configure `nanorun.toml`, HF auth)
2. Grab the IP/SSH command of any GPU machine you have access to
3. Tell the agent to set it up: "connect to root@<ip> and set up the machine"
4. Ask it to queue an experiment: "queue records/record74"

From there, run `nanorun local start` in a terminal tab — it syncs results and serves the dashboard at localhost:8080.

### Manual quickstart

```bash
# Connect to your GPU machine
nanorun session start root@<ip> --name my-h100

# One-time setup (installs deps, clones repo, downloads data)
nanorun session setup

# Start the local daemon (syncs logs + metrics, serves dashboard at localhost:8080)
nanorun local start

# Create a track for your experiments
nanorun track create my-experiments experiments/myexp/

# Copy a baseline and edit it
cp experiments/records/record74.py experiments/myexp/train.py
# ... make your changes ...

# Sync and run
nanorun sync experiments/myexp/train.py
nanorun job add experiments/myexp/train.py

# Monitor (dashboard already running at localhost:8080)
nanorun job status         # quick check from CLI
nanorun job logs --tail    # raw output
```

## Included Scripts

`experiments/records/`, `experiments/optimizer-records/`, and `experiments/medium-track/` contain record scripts from the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun. These are highly optimized GPT-2 (124M) training scripts — good starting points for experimentation.

- **records/** — Curated list of Modded NanoGPT records (train to val_loss <3.28 in shortest wall-clock time on 8xH100)
- **medium-track/** — Medium-scale records (train to val_loss <2.92)
- **optimizer-records/** — Optimizer track records (fixed architecture, compete for fewest training steps <3.28)


## How it works

```
Your machine                       Remote GPU
───────────                        ──────────
nanorun sync ──► GitHub ──────────► remote daemon
                                        │
local daemon ◄────── HF Bucket ◄────────┘
  ├─ local dashboard
  └─ metrics parsed → SQLite
```

See `agents.md` for full documentation of all commands, architecture, and workflows.

## License

MIT
