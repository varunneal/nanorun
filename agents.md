# nanorun

General-purpose experiment and training platform for GPU machines (H100, H200, GH200). Designed for speedruns, optimizer research, architecture experiments, and other ML training workloads.

## Overview

nanorun is a **monorepo** containing both the CLI tool and all experiment code. It automates the experiment cycle: SSH into remote GPU machines, sync code, run experiments, and monitor results.

## Repository Structure

```
nanorun/
├── nanorun/                # CLI tool source
│   ├── cli.py              # CLI entry point, command definitions
│   ├── config.py           # Multi-session config, track management
│   ├── remote_control.py   # RemoteSession (SSH), DaemonClient (RPC wrapper)
│   ├── rpc_client.py       # WebSocket RPC client + SSH tunnel management
│   ├── rpc_types.py        # Shared RPC protocol types (Request, Response, Event)
│   ├── hub.py              # HuggingFace Buckets integration (logs, weights)
│   ├── setup.py            # Machine setup (apt, uv, torch, data, HF auth)
│   ├── sync.py             # Git sync (commit, push, pull on remote)
│   ├── runner.py           # Experiment execution, torchrun commands
│   ├── queue.py            # Job queue management
│   ├── tracker.py          # SQLite experiment/metrics storage
│   ├── remote_daemon.py    # Remote daemon with integrated RPC server
│   ├── local_daemon.py     # Local daemon (event-driven, multi-session)
│   ├── lineage.py          # Script parentage and code diffs
│   ├── utils.py            # Shared utilities
│   └── dashboard/          # Web UI (FastAPI)
├── experiments/            # Training scripts organized by track
│   ├── records/            # Speedrun record scripts (good starting points)
│   └── optimizer-track-records/  # Optimizer track baselines
├── data/                   # Data loading utilities
│   └── cached_fineweb10B.py  # Downloads FineWeb-10B shards
├── .nanorun/               # Local config (gitignored)
│   ├── active_session      # Name of currently active session
│   ├── experiments.db      # SQLite with experiments + metrics
│   ├── diffs/              # Code diffs indexed by child hash
│   ├── logs/               # Synced log files from HF Bucket (flat, keyed by run_id)
│   ├── tunnels/            # SSH tunnel lock files (per session)
│   ├── local_daemon/       # Local daemon state
│   │   ├── daemon.pid
│   │   └── daemon.log
│   └── sessions/           # Per-session config and state
│       └── {name}/
│           ├── session.json    # SessionConfig (host, user, GPU type, etc.)
│           ├── state.json      # Daemon tracking state
│           ├── queue_cache.json # Cached queue from remote
│           ├── crashes.json    # Recent crash notifications
│           └── events.log      # Per-session event log
└── pyproject.toml
```

On the **remote machine**, this repo is cloned to `~/nanorun/`.

## Installation

```bash
git clone git@github.com:<your-username>/nanorun.git
cd nanorun
uv sync
uv tool install -e .
nanorun --help
```

## Core Commands

### Session management

| Command | Description |
|---------|-------------|
| `nanorun session start user@<ip> [--name NAME]` | Connect to remote machine (auto-generates name if not specified) |
| `nanorun session list` | List all sessions with connected/disconnected status |
| `nanorun session switch <name>` | Change active session |
| `nanorun session status [--session NAME]` | Check connection status |
| `nanorun session setup [--session NAME]` | Interactive machine setup (apt, uv, torch, clone repo, HF auth) |
| `nanorun session setup --verify` | Verify all dependencies work |
| `nanorun session attach [--session NAME]` | Attach to remote tmux session |
| `nanorun session cleanup` | Remove disconnected sessions |

### Tracks

| Command | Description |
|---------|-------------|
| `nanorun track create <name> <dir>` | Create experiment track |
| `nanorun track list` | List all tracks |
| `nanorun track delete <name>` | Delete a track |
| `nanorun track info <name>` | Show track details |

### Sync and jobs

All job/sync commands accept `--session NAME` to target a specific session (defaults to active session). If no active session is set but exactly one session is connected, that session is used automatically.

| Command | Description |
|---------|-------------|
| `nanorun sync <file> [<file2> ...]` | Sync specific file(s) to remote (default, encouraged) |
| `nanorun sync --all -m "msg"` | Sync all changes (git add -A) |
| `nanorun sync --no-verify` | Skip Python syntax check |
| `nanorun job add <script> --env K=V` | Add experiment to queue (use multiple `--env` for multiple vars) |
| `nanorun job add <script> --first` | Add experiment to front of queue |
| `nanorun job sweep <script> --env K=V1,V2` | Add parameter sweep (use multiple `--env` for multiple vars) |
| `nanorun job status` | Show current job + queue status |
| `nanorun job status --daemon` | Also show remote daemon status |
| `nanorun job logs --tail` | Watch raw tmux output |
| `nanorun job queue --flat` | Show queued experiments (plain text, no truncation) |
| `nanorun job clear` | Clear all queued experiments |
| `nanorun job remove <index>` | Remove item from queue |
| `nanorun job cancel` | Cancel current experiment, pause queue |
| `nanorun job resume` | Start next queued experiment |
| `nanorun job ps` | Show GPU processes on remote |

### Local daemon and dashboard

| Command | Description |
|---------|-------------|
| `nanorun local start` | Start local daemon (manages all sessions) |
| `nanorun local stop` | Stop local daemon |
| `nanorun local status` | Show local daemon status (all sessions) |
| `nanorun local restart` | Restart local daemon |
| `nanorun dashboard` | Start web UI at localhost:8080 |

### Dashboard API

When the dashboard is running (`nanorun dashboard`), these endpoints are available at `localhost:8080`:

| Endpoint | Description |
|----------|-------------|
| `/api/experiments?track=X&status=Y&limit=N` | List experiments (grouped by code_hash by default) |
| `/api/experiments/running` | Currently running experiments with latest metrics |
| `/api/experiment/{exp_id}` | Full detail: loss curve, metrics, crash log |
| `/api/experiment/{exp_id}/metrics` | All metrics for an experiment |
| `/api/logs/{run_id}` | Raw training log text |
| `/api/crash/{exp_id}` | Crash/output log |
| `/api/queue` | Queue state: running + queued items across all sessions |

### Hub (Log/Weight Storage)

The hub supports two backends: HuggingFace Buckets (default) or S3-compatible storage. Configured in `nanorun.toml`. The public API is identical regardless of backend.

| Command | Description |
|---------|-------------|
| `nanorun hub status` | Show bucket info and auth status |
| `nanorun hub logs <experiment_id>` | Fetch and display log from hub |
| `nanorun hub weights <experiment_id>` | List model weights in hub |
| `nanorun hub weights <experiment_id> --download` | Download weights locally |

## Sync Flow

```
Local (your machine)  --git push-->  GitHub  --git pull-->  Remote (GPU machine, ~/nanorun/)
```

`nanorun sync` requires either specific files or `--all`.

**Single-file sync (default, encouraged):**
```bash
nanorun sync experiments/muon/train.py              # sync one file
nanorun sync experiments/muon/train.py utils.py     # sync multiple files
nanorun sync experiments/muon/train.py -m "fix LR"  # with custom message
```
1. `git add -- <files> && git commit` locally (auto-message: `sync <files>`)
2. `git push` to origin
3. SSH to remote and `git pull` in `~/nanorun/`

**Sync all (explicit opt-in):**
```bash
nanorun sync --all                  # sync everything
nanorun sync --all -m "big refactor"
```
1. `git add -A && git commit` locally
2. `git push` to origin
3. SSH to remote and `git pull` in `~/nanorun/`

Running `nanorun sync` with no arguments and no `--all` flag will error and prompt you to specify files.

The daemon is restarted only if `nanorun/remote_daemon.py` was among the changed files. The restart is safe even with experiments running - experiments run in their own tmux windows and daemon state is persisted to disk.

### Metrics and log sync

Logs and metrics flow through HuggingFace Buckets:
1. Remote daemon periodically pushes all logs from `~/nanorun/logs/` to HF Bucket
2. Local daemon pulls `.txt` log files from HF Bucket (rsync-like diffing via `sync_bucket`)
3. Local daemon parses metrics from downloaded logs into SQLite

Model weights are pushed to the hub by the remote daemon when the training script saves checkpoints to `logs/{run_id}/state_step{step:06d}.pt`. Most training scripts do NOT save weights by default — this only happens if the script explicitly includes checkpoint saving. Weights are NOT auto-synced to local. Use `nanorun hub weights` to list/download on demand.

## Key Modules

- **config.py**: Multi-session config stored in `.nanorun/sessions/{name}/session.json`, active session pointer, track management
- **remote_control.py**: `RemoteSession` (SSH via Paramiko), `DaemonClient` (wraps RPC calls to remote daemon)
- **rpc_client.py**: WebSocket RPC client with SSH tunnel management. Shared tunnel per session via lock files.
- **rpc_types.py**: Shared protocol types — `Request`, `Response`, `EventMessage`, `Method` enum, `Event` enum
- **hub.py**: HuggingFace Buckets integration — log sync up/down, weight upload/download, auth
- **setup.py**: Auto-detects CUDA version from `nvidia-smi`, builds correct torch URL, downloads training data, provisions HF token
- **runner.py**: Builds `torchrun` commands, sends to remote daemon via RPC
- **queue.py**: Job queue management (delegates to remote daemon via RPC)
- **tracker.py**: SQLite experiment/metrics storage, parsing log output
- **remote_daemon.py**: Remote daemon with integrated WebSocket RPC server, hub sync, queue management
- **local_daemon.py**: Event-driven local daemon managing multiple sessions, hub-based metrics sync
- **dashboard/app.py**: FastAPI routes for experiment list, loss curves, queue status
- **lineage.py**: Parses parent declarations, generates and stores code diffs

## GPU Type Support

Supports H100, H200, GH200. Auto-detected from `nvidia-smi` at session start, stored in session config. Override with `--gpu-type` flag.

## Daemon Architecture

nanorun uses a two-daemon architecture with WebSocket RPC communication:

1. **Remote Daemon** (`remote_daemon.py`) - runs on the GPU machine in tmux window `nanorun:daemon`
   - Integrated WebSocket RPC server on `localhost:9321`
   - Manages experiment queue persistently
   - Executes experiments in tmux windows
   - Computes code hashes on remote (guarantees correctness)
   - Writes mapping files for recovery (`~/nanorun/.daemon/mappings/`)
   - Auto-starts next experiment when one finishes
   - Pushes logs and model weights to HuggingFace Bucket
   - Emits events: `EXPERIMENT_STARTED`, `EXPERIMENT_FINISHED`, `EXPERIMENT_FAILED`, `QUEUE_CHANGED`

2. **Local Daemon** (`local_daemon.py`) - runs on your machine
   - Manages **multiple sessions simultaneously** (one WebSocket connection per session)
   - Event-driven: listens to remote daemon events via RPC (not polling)
   - Pulls logs from HF Bucket (rsync-like sync, every ~3s when tracking)
   - Parses metrics from downloaded logs into SQLite
   - Provides crash notifications (macOS desktop alerts)
   - Marks sessions as disconnected on SSH tunnel failure (no reconnect spam)

### Communication

```
CLI commands ──► RPC (WebSocket over SSH tunnel) ──► Remote Daemon
                                                         │
Local Daemon ──► RPC events (same tunnel) ◄──────────────┘
                                                         │
                 HF Bucket ◄── logs + weights ───────────┘
                     │
Local Daemon ──► pull .txt logs ──► parse metrics ──► SQLite
```

- **RPC over SSH**: SSH port forwarding tunnels `localhost:N → remote:9321`. One tunnel per session, shared between CLI and daemon via lock files in `.nanorun/tunnels/`.
- **Hub**: Logs and weights stored in HuggingFace Bucket. Paths: `logs/{session}/{run_id}.txt`, `weights/{session}/{experiment_id}/state_step*.pt`

### Remote Directory Structure

```
~/nanorun/.daemon/
├── daemon.pid          # PID file with flock (prevents multiple daemons)
├── daemon.log          # Persistent remote daemon stdout/stderr (tee'd from tmux; survives restarts)
├── state.json          # {"status": "running", "current_experiment_id": 42, "session_name": "..."}
├── queue.txt           # Queued experiments (JSON lines)
├── crash_logs/         # Final tmux output for failed experiments
└── mappings/
    ├── 42.json         # {experiment_id, run_id, script, code_hash, log_file, ...}
    └── 43.json
```

### Multi-Session

- Multiple sessions can be active simultaneously, each with its own remote machine
- Local daemon tracks all connected sessions in parallel
- Each session has independent queue, experiments, and daemon state
- Session names are auto-generated (`session-1`, `session-2`) or user-specified
- If no active session is set but exactly one is connected, commands use it automatically
- Sessions persist on disk; disconnected sessions stop retrying and require explicit reconnection

### Recovery

If your local machine dies while experiments are running:

1. Remote daemon continues running experiments and pushing logs to hub
2. When you're back, start local daemon: `nanorun local start`
   - Automatically detects running/completed experiments via RPC
   - Pulls experiment mappings and updates local DB
   - Syncs logs from HF Bucket

## Log Files

Training scripts write logs to `~/nanorun/logs/{run_id}.txt` on the remote machine. The remote daemon pushes these to HuggingFace Bucket, and the local daemon pulls them.

### Writing to the log file

Use `print0` (or the script's `log` helper) for anything that should land in the log file — metrics, diagnostics. Plain `print()` only hits the tmux pane and is lost from the hub/SQLite/dashboard. Use `print0(..., console=True)` to write both.

### Sync Chain

1. Training script writes to `~/nanorun/logs/{run_id}.txt` on remote
2. Remote daemon syncs all of `~/nanorun/logs/` to HF Bucket (every ~5s)
3. Local daemon pulls `.txt` files from HF Bucket via `sync_bucket` (every ~3s when tracking)
4. Local daemon parses metrics from downloaded logs into SQLite

### Local Storage

Logs are stored in `.nanorun/logs/` (gitignored, flat directory keyed by run_id UUID).

### Manual Access

```bash
# View synced logs locally
ls .nanorun/logs/

# Read a specific log
grep "val_loss" .nanorun/logs/{run_id}.txt

# Fetch a log via the dashboard API
curl -s http://localhost:8080/api/logs/{run_id} | grep "val_loss"

# Fetch a log from hub directly
nanorun hub logs <experiment_id>

# Query the experiments database
sqlite3 .nanorun/experiments.db "SELECT id, name, remote_run_id FROM experiments ORDER BY id DESC LIMIT 5;"
```

## Script Lineage

When creating a new training script, **always declare its parent** in the frontmatter. This builds a lineage tree that tracks how experiments evolve.

### Frontmatter Format

Add a docstring at the very top of your script:

```python
"""
parent: experiments/records/record52.py
"""

import torch
# ... rest of script
```

### Rules

1. **Copy an existing script as your starting point** — these scripts are 1000+ lines of highly optimized code; rewriting from scratch will drop critical details. Use `cp parent.py new_script.py`, then edit the copy.
2. **Always declare a parent** when deriving from an existing script.
3. **Root nodes** (no parent) are scripts that don't derive from anything.
4. **Path is relative** to the repo root (`~/nanorun/` on remote).
5. The parent declaration must be in the **first docstring** of the file.

### What Happens

When you run an experiment:
1. nanorun computes a SHA256 hash of your script's entire file contents, using the first 12 characters (`code_hash`). Note: whitespace changes affect the hash.
2. If a parent is declared, it computes the parent's hash (`parent_hash`)
3. A unified diff is generated and stored in `.nanorun/diffs/{code_hash}.diff`
4. Both hashes are recorded in the experiments database

### Example Workflow

```bash
# Create a track for your experiment
nanorun track create muon-experiment experiments/muon/ -d "Muon optimizer experiments"

# Copy a record as your baseline
cp experiments/records/record52.py experiments/muon/higher_lr_warmup.py

# Edit higher_lr_warmup.py - add frontmatter at top:
"""
parent: experiments/records/record52.py
"""

# Make your changes...

# Add to queue (will prompt to sync)
nanorun job add experiments/muon/higher_lr_warmup.py
```

## Triton Kernels Versioning

When a training script uses external Triton kernels, declare them in the frontmatter so nanorun tracks them alongside the script.

### Frontmatter Format

```python
"""
parent: experiments/records/record60.py
kernels: experiments/records/record61_kernels.py
"""

from triton_kernels import XXT, ba_plus_cAA  # resolved via symlink
```

### What Happens

When you run an experiment with a `kernels:` declaration:
1. The daemon creates a symlink `triton_kernels.py` in the script's directory pointing to the kernels file
2. The `code_hash` is computed from **both** files: `SHA256(script + '\n---KERNELS---\n' + kernels)[:12]`
3. The diff (in `.nanorun/diffs/`) includes changes to both script and kernels
4. The `kernels_path` is stored in the experiment record

### Rules

1. **Path is relative** to the repo root (e.g., `experiments/records/record61_kernels.py`)
2. The kernels file can be in a **different directory** than the script
3. Scripts **without** `kernels:` work as before (script-only hash)
4. You can use both `parent:` and `kernels:` in the same frontmatter

## Script Notes

You can attach notes to any training script using a sidecar markdown file. Notes are displayed in the dashboard and are version-controlled alongside the script.

### File Format

For a script at `experiments/records/record52.py`, create a notes file at:
```
experiments/records/record52.notes.md
```

The notes file is plain markdown. No special format required.

## Tracks (Experiment Workstreams)

Tracks let you organize parallel lines of experimentation. Track is **auto-inferred from script path** — if a directory contains `.track.json`, experiments in that directory are automatically tagged.

```bash
# Create tracks for different experiment themes
nanorun track create lr-sweep experiments/lr/ -d "Learning rate experiments"
nanorun track create attention-mods experiments/attn/ -d "Attention modifications"

# Add experiments (track auto-inferred from path)
nanorun job add experiments/lr/train_lr.py --env LR=0.03
# Output: "Track: lr-sweep"

# List all tracks
nanorun track list
```

Tracks are just metadata (`.track.json` files) — they map to directories where you keep your experimental scripts.

## Job Queue

Experiments run one at a time. All experiments are added to a queue and processed sequentially.

```bash
# Add an experiment to the queue:
$ nanorun job add experiments/records/train.py --env LR=0.03
Added to queue (position 1)
Nothing running - starting experiment...

# If something is already running, it just queues:
$ nanorun job add experiments/records/train2.py
Added to queue (position 2)

# Add a parameter sweep (all configs get queued):
$ nanorun job sweep experiments/lr/train.py --env LR=0.01,0.02,0.03
Adding 3 configurations to queue
Nothing running - starting first experiment...

# Multi-variable sweep creates a grid (3x2 = 6 configs):
$ nanorun job sweep experiments/lr/train.py --env LR=0.01,0.02,0.03 --env WD=0.1,0.2
Adding 6 configurations to queue

# View queue:
$ nanorun job queue
Queue Status: active

  #  Script                              Env       Track    GPUs
  1  experiments/records/train.py        LR=0.03   records  1
  2  experiments/lr/train.py             LR=0.02   lr       1

# Cancel current experiment (pauses queue):
$ nanorun job cancel

# Start next queued experiment:
$ nanorun job resume
```

## Important Notes

**Auto-start**: When you use `job add` or `job sweep`, if nothing is currently running, the experiment starts immediately.

**Code version at run time, not queue time**: The daemon runs whatever code is on the remote at the time the experiment starts, NOT at the time it was queued. If you `nanorun sync` after queueing, all pending experiments will run the updated code.

**Sync check**: Both `job add` and `job sweep` check if the specific script being added has unsynced changes. If the script has local changes that haven't been pushed, you'll be prompted to sync it first.

**Environment variables**: The `--env K=V` flag passes actual environment variables to the training script (prepended to the torchrun command). Access them in your script via `os.environ['K']`. Use a separate `--env` flag for each variable (e.g., `--env LR=0.03 --env WD=0.1`).

## Typical Workflow

1. Start a session: `nanorun session start user@<ip> --name my-h100`
2. Setup machine (first time): `nanorun session setup` (installs deps, clones repo, provisions HF auth)
3. Start local daemon: `nanorun local start`
4. Create a track: `nanorun track create my-experiment experiments/myexp/`
5. Copy a baseline script into your track directory
6. Edit your script, then add to queue: `nanorun job add experiments/myexp/train.py`
7. Monitor: `nanorun dashboard` or `nanorun job status`
8. Debug: `nanorun job logs --tail` (raw tmux output)
9. Check weights: `nanorun hub weights <experiment_id>`

## General Notes

- SSH uses `-A` flag to forward your GitHub keys for git clone on remote
- Experiments run in tmux so they persist if you disconnect
- Setup auto-detects CUDA 12.6/12.8/13.0 and picks correct torch nightly
- Setup runs `data/cached_fineweb10B.py` to download training data shards
- Setup provisions HuggingFace token on remote (reads local `~/.cache/huggingface/token`)
- Dashboard auto-refreshes every 5 seconds
- Queue is managed by remote daemon via RPC — local queue files are caches
- The `.nanorun/` directory is gitignored (local-only config and data)
- Use `--session NAME` on any job/sync command to target a specific session

## Querying the Database

Experiment data is stored in SQLite at `.nanorun/experiments.db`. You can query it directly:

```bash
sqlite3 .nanorun/experiments.db

# List all experiments
SELECT id, name, track, status, started_at FROM experiments ORDER BY id DESC LIMIT 10;

# Get metrics for a specific experiment
SELECT step, val_loss, train_time_ms, step_avg_ms FROM metrics WHERE experiment_id = 42 ORDER BY step;

# Find best val_loss per track
SELECT e.track, e.name, MIN(m.val_loss) as best_loss, m.train_time_ms
FROM experiments e
JOIN metrics m ON e.id = m.experiment_id
GROUP BY e.track
ORDER BY best_loss;
```

### Schema

**experiments** table:
- `id`, `name`, `track`, `script`, `code_hash`, `parent_hash`, `git_commit`
- `status` (running, completed, failed, cancelled, queued)
- `env_vars` (JSON), `gpus`, `gpu_type` (H100/H200/GH200), `run_number`, `tmux_window`, `remote_run_id`
- `session_name` (TEXT), `kernels_path` (TEXT), `crash_log` (TEXT)
- `started_at`, `finished_at`

**metrics** table:
- `experiment_id`, `step`, `total_steps`
- `val_loss`, `train_time_ms`, `step_avg_ms`
- `is_final_step`, `recorded_at`

## Included Experiment Scripts

The `experiments/records/` and `experiments/optimizer-track-records/` directories contain record scripts from the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun competition. These are good starting points for experimentation:

- **`experiments/records/`** — Speedrun records: train GPT-2 (124M) on FineWeb-10B to val_loss <3.28 in shortest wall-clock time on 8xH100. These are highly optimized, production-quality training scripts.
- **`experiments/optimizer-track-records/`** — Optimizer track records: fixed architecture, compete on optimizer design. Good baselines for optimizer research.

To start experimenting, copy one of these scripts into your own track and declare it as a parent in the frontmatter.
