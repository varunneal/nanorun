# nanorun

Experiment platform for training, speedrunning, and researching. Handles machine setup, experiment queuing, log collection, experiment lineages. You (or agent) write training scripts, view results from CLI or via the local dashboard (`nanorun dashboard`). Replacement for W&B.

## Setup

```bash
git clone git@github.com:varunneal/nanorun.git
cd nanorun
uv sync
uv tool install -e .
nanorun --help
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

### HuggingFace Bucket

nanorun uses a HuggingFace Bucket to sync logs and model weights between remote and local. You'll need:

1. A HuggingFace account with a write token at `~/.cache/huggingface/token`
2. Update the bucket ID in `nanorun/hub.py` (`BUCKET_ID = "your-username/your-bucket"`)

The bucket is created automatically on first push if it doesn't exist.

## Quickstart

The recommended path: open Claude Code (or any coding agent) in this repo and let it drive.

1. Grab the IP/SSH command of any GPU machine you have access to
2. Tell the agent to set it up: "connect to root@<ip> and set up the machine"
3. Ask it to queue an experiment: "queue records/record74"

From there, run `nanorun local start` in one terminal tab to automatically sync results. Run `nanorun dashboard` in another window for a localhost dashboard.

### Manual quickstart

```bash
# Connect to your GPU machine
nanorun session start root@<ip> --name my-h100

# One-time setup (installs deps, clones repo, downloads data)
nanorun session setup

# Start the local daemon (syncs logs + metrics in background)
nanorun local start

# Create a track for your experiments
nanorun track create my-experiments experiments/myexp/

# Copy a baseline and edit it
cp experiments/records/record74.py experiments/myexp/train.py
# ... make your changes ...

# Sync and run
nanorun sync experiments/myexp/train.py
nanorun job add experiments/myexp/train.py

# Monitor
nanorun dashboard          # web UI at localhost:8080
nanorun job status         # quick check
nanorun job logs --tail    # raw output
```

## Included Scripts

`experiments/records/` and `experiments/optimizer-track-records/` contain record scripts from the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun. These are highly optimized GPT-2 (124M) training scripts — good starting points for experimentation.

- **records/** — Speedrun records (train to val_loss <3.28 in shortest wall-clock time on 8xH100)
- **optimizer-track-records/** — Optimizer track (fixed architecture, compete on optimizer design)

## How it works

```
Your machine                          Remote GPU machine
───────────                           ──────────────────
nanorun CLI ──git push──► GitHub ──git pull──► ~/nanorun/
local daemon                          remote daemon
  │                                     │
  ├─ pulls logs from HF Bucket ◄───────┤─ pushes logs to HF Bucket
  ├─ parses metrics → SQLite            ├─ runs experiments in tmux
  └─ crash notifications                └─ manages queue
```

See `agents.md` for full documentation of all commands, architecture, and workflows.

## License

MIT
