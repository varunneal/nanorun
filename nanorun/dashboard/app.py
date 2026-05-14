"""FastAPI dashboard application."""

import time
import webbrowser
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import discover_tracks, Config
from ..queue import get_queue_state
from ..local_daemon import safe_json_load, get_queue_cache_file
from ..tracker import (
    get_experiments,
    get_experiment,
    get_running_experiments,
    get_metrics,
    get_latest_metric,
    get_final_metric,
    get_db,
)

app = FastAPI(title="nanorun Dashboard")

# Setup templates and static files
DASHBOARD_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=DASHBOARD_DIR / "templates")
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR / "static"), name="static")

# Cache buster: changes on each server start
BOOT_VERSION = str(int(time.time()))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request, "v": BOOT_VERSION})


@app.get("/api/themes")
async def list_themes():
    """List available themes from the themes directory."""
    themes_dir = DASHBOARD_DIR / "static" / "themes"
    return [p.stem for p in sorted(themes_dir.glob("*.css"))]


def _batch_latest_metrics(experiment_ids: List[int]) -> dict:
    """Fetch latest metric for each experiment in a single query."""
    if not experiment_ids:
        return {}
    conn = get_db()
    placeholders = ",".join("?" for _ in experiment_ids)
    rows = conn.execute(
        f"""SELECT m.* FROM metrics m
            INNER JOIN (
                SELECT experiment_id, MAX(step) as max_step
                FROM metrics
                WHERE experiment_id IN ({placeholders})
                GROUP BY experiment_id
            ) latest ON m.experiment_id = latest.experiment_id AND m.step = latest.max_step""",
        experiment_ids,
    ).fetchall()
    conn.close()
    return {
        row["experiment_id"]: {
            "step": row["step"],
            "total_steps": row["total_steps"],
            "val_loss": row["val_loss"],
            "train_time_ms": row["train_time_ms"],
        }
        for row in rows
    }


@app.get("/api/experiments")
async def list_experiments(track: Optional[str] = None, status: Optional[str] = None, search: Optional[str] = None, limit: int = 100, aggregate: bool = True):
    """List experiments, optionally aggregated by code_hash."""
    # Aggregation collapses sweeps (50+ runs each), so fetch more raw rows than display limit
    experiments = get_experiments(track=track, status=status, search=search, limit=2000)

    # Batch-fetch latest metrics in one query instead of N+1
    all_ids = [exp.id for exp in experiments]
    latest_metrics = _batch_latest_metrics(all_ids)

    if not aggregate:
        # Return flat list (old behavior)
        results = []
        for exp in experiments:
            m = latest_metrics.get(exp.id)
            results.append({
                "id": exp.id,
                "name": exp.name,
                "track": exp.track,
                "script": exp.script,
                "code_hash": exp.code_hash,
                "status": exp.status,
                "gpus": exp.gpus,
                "gpu_type": exp.gpu_type,
                "env_vars": exp.env_vars,
                "started_at": exp.started_at.isoformat() if exp.started_at else None,
                "finished_at": exp.finished_at.isoformat() if exp.finished_at else None,
                "current_step": m["step"] if m else None,
                "total_steps": m["total_steps"] if m else None,
                "val_loss": m["val_loss"] if m else None,
                "train_time_ms": m["train_time_ms"] if m else None,
            })
        return {"experiments": results[:limit]}

    # Aggregate by (code_hash, track, gpus, gpu_type)
    from collections import defaultdict
    import json

    groups = defaultdict(list)
    for exp in experiments:
        # Use (code_hash, track, gpus, gpu_type) as key, or unique ID if no hash
        gpu_type = getattr(exp, 'gpu_type', 'H100') or 'H100'
        if exp.code_hash:
            key = (exp.code_hash, exp.track or "", exp.gpus, gpu_type)
        else:
            key = (f"_no_hash_{exp.id}", exp.track or "", exp.gpus, gpu_type)
        groups[key].append(exp)

    results = []
    for code_hash, group_exps in groups.items():
        # Sort by started_at desc to get most recent first
        group_exps.sort(key=lambda e: e.started_at.isoformat() if e.started_at else "", reverse=True)
        primary = group_exps[0]  # Most recent experiment

        # Collect metrics for all experiments in group
        val_losses = []
        train_times = []
        env_var_sets = set()
        experiment_ids = []
        statuses = []

        for exp in group_exps:
            experiment_ids.append(exp.id)
            statuses.append(exp.status)
            env_var_sets.add(json.dumps(exp.env_vars, sort_keys=True))
            # Only include completed experiments in averages (not running ones)
            if exp.status != "running":
                m = latest_metrics.get(exp.id)
                if m:
                    if m["val_loss"] is not None:
                        val_losses.append(m["val_loss"])
                    if m["train_time_ms"] is not None:
                        train_times.append(m["train_time_ms"])

        # Determine if this is a sweep (same code, different env vars)
        is_sweep = len(env_var_sets) > 1

        # Compute aggregates
        n_runs = len(group_exps)
        mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else None
        mean_train_time = sum(train_times) / len(train_times) if train_times else None

        # Aggregate status: running if any running, else completed if any completed, else first status
        if "running" in statuses:
            agg_status = "running"
        elif "completed" in statuses:
            agg_status = "completed"
        else:
            agg_status = statuses[0] if statuses else "unknown"

        # Get current step from most recent experiment with metrics
        current_step = None
        total_steps = None
        for exp in group_exps:
            m = latest_metrics.get(exp.id)
            if m:
                current_step = m["step"]
                total_steps = m["total_steps"]
                break

        results.append({
            "id": primary.id,  # Primary experiment ID
            "experiment_ids": experiment_ids,  # All experiment IDs in group
            "name": primary.name,
            "track": primary.track,
            "script": primary.script,
            "code_hash": primary.code_hash,
            "status": agg_status,
            "gpus": primary.gpus,
            "gpu_type": primary.gpu_type,
            "env_vars": primary.env_vars,
            "started_at": primary.started_at.isoformat() if primary.started_at else None,
            # Aggregated metrics
            "n_runs": n_runs,
            "is_sweep": is_sweep,
            "current_step": current_step,
            "total_steps": total_steps,
            "val_loss": mean_val_loss,
            "train_time_ms": mean_train_time,
            "val_losses": val_losses,  # Individual values for details
            "train_times": train_times,
        })

    # Sort by most recent started_at
    results.sort(key=lambda r: r["started_at"] or "", reverse=True)
    return {"experiments": results[:limit]}


@app.get("/api/experiments/running")
async def list_running_experiments():
    """List only running experiments with their latest metrics."""
    experiments = get_running_experiments()

    results = []
    for exp in experiments:
        latest = get_latest_metric(exp.id)
        results.append({
            "id": exp.id,
            "name": exp.name,
            "track": exp.track,
            "status": exp.status,
            "current_step": latest.step if latest else None,
            "total_steps": latest.total_steps if latest else None,
            "val_loss": latest.val_loss if latest else None,
            "train_time_ms": latest.train_time_ms if latest else None,
        })

    return {"experiments": results}


@app.get("/api/queue")
async def get_queue_status():
    """Get queue status across all sessions.

    Returns running experiments and queued items from all sessions,
    each tagged with session_name. Queued items use per-session ordinals.
    """
    from ..config import Config

    sessions = Config.list_sessions()
    running_list = []
    queued_list = []

    # Get running experiments from DB (all sessions)
    running_exps = get_running_experiments()
    for exp in running_exps:
        latest = get_latest_metric(exp.id)
        running_list.append({
            "id": exp.id,
            "name": exp.name,
            "script": exp.script,
            "track": exp.track,
            "tmux_window": exp.tmux_window,
            "gpus": exp.gpus,
            "gpu_type": exp.gpu_type,
            "env_vars": exp.env_vars,
            "session_name": exp.session_name,
            "current_step": latest.step if latest else None,
            "total_steps": latest.total_steps if latest else None,
            "val_loss": latest.val_loss if latest else None,
        })

    # Get queued items from all session caches
    for session in sessions:
        cache_data = safe_json_load(get_queue_cache_file(session.name), default={})
        queue_items = cache_data.get("queue", []) if cache_data else []
        for idx, item in enumerate(queue_items):
            queued_list.append({
                "id": item.get("experiment_id"),
                "script": item.get("script", ""),
                "env_vars": item.get("env_vars", {}),
                "track": item.get("track"),
                "gpus": item.get("gpus", 1),
                "gpu_type": item.get("gpu_type", "H100"),
                "name": item.get("name"),
                "session_name": session.name,
                "session_index": idx + 1,
            })

    # Get queue state
    state = get_queue_state()

    return {
        "running": running_list[0] if len(running_list) == 1 else None,
        "running_list": running_list,
        "queued": queued_list,
        "state": state,
    }


@app.get("/api/sessions")
async def get_sessions():
    """Get session statuses + hub syncer state, sorted disconnected-first."""
    from ..local_daemon import SessionState

    sessions = Config.list_sessions()
    daemon = getattr(app.state, "daemon", None)
    hub = daemon.hub_syncer if daemon else None
    result = []
    for sc in sessions:
        state = SessionState.load(sc.name)
        result.append({
            "name": sc.name,
            "host": f"{sc.user}@{sc.host}:{sc.port}",
            "gpu_type": sc.gpu_type,
            "gpu_count": sc.gpu_count,
            "status": state.status,
            "last_error": state.last_error,
            "metrics_synced": state.metrics_synced,
            "tracking_experiment_id": state.tracking_experiment_id,
        })
    result.sort(key=lambda s: (0 if s["status"] == "disconnected" else 1, s["name"]))
    return {
        "sessions": result,
        "hub": {
            "status": hub.status if hub else "unknown",
            "last_error": hub.last_error if hub else None,
            "last_sync_at": hub.last_sync_at if hub else None,
        },
    }


@app.post("/api/sessions/{name}/reconnect")
async def reconnect_session(name: str):
    """Trigger a reconnect attempt for a disconnected session."""
    daemon = app.state.daemon
    if not daemon:
        return JSONResponse({"error": "Daemon not available"}, status_code=503)
    ok = daemon.reconnect_session(name)
    return {"success": ok, "message": "Reconnecting..." if ok else "Session not found"}


@app.post("/api/hub/reconnect")
async def reconnect_hub():
    """Trigger a reconnect attempt for the hub syncer."""
    daemon = app.state.daemon
    if not daemon:
        return JSONResponse({"error": "Daemon not available"}, status_code=503)
    ok = daemon.reconnect_hub()
    return {"success": ok, "message": "Hub reconnecting..." if ok else "Hub syncer already running"}


@app.delete("/api/sessions/{name}")
async def delete_session(name: str):
    """Remove a session (only if disconnected)."""
    import shutil
    from ..local_daemon import SessionState

    state = SessionState.load(name)
    if state.status == "connected":
        return JSONResponse(
            {"error": "Cannot remove a connected session. Disconnect first."},
            status_code=400,
        )
    Config.delete_session(name)
    state_dir = Config.get_session_state_dir(name)
    if state_dir.exists():
        shutil.rmtree(state_dir, ignore_errors=True)
    daemon = getattr(app.state, "daemon", None)
    if daemon and hasattr(daemon, "remove_session"):
        daemon.remove_session(name)
    return {"success": True, "message": f"Session '{name}' removed"}


@app.post("/api/sessions/{name}/daemon-restart")
async def restart_remote_daemon(name: str):
    """Restart the remote daemon for a session (stop + start)."""
    import threading
    from ..remote_control import get_daemon_client, DaemonError

    def _do_restart():
        client = get_daemon_client(name)
        if client:
            with client:
                try:
                    client.restart_daemon()
                except DaemonError:
                    pass

    threading.Thread(target=_do_restart, daemon=True).start()
    return {"success": True, "message": "Daemon restart initiated"}


@app.get("/api/sessions/{name}/daemon-status")
async def get_session_daemon_status(name: str):
    """Get remote daemon status (experiment, queue, GPU) for a connected session."""
    from ..queue import get_daemon_status

    status = get_daemon_status(session_name=name)
    if not status:
        return JSONResponse({"error": "Could not reach daemon"}, status_code=503)
    return status


@app.get("/api/experiment/{exp_id}")
async def get_experiment_detail(exp_id: int):
    """Get detailed data for a single experiment including loss curve."""
    exp = get_experiment(exp_id)
    if not exp:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)

    # Get all metrics for loss curve
    metrics = get_metrics(exp_id)
    loss_curve = [
        {
            "step": m.step,
            "val_loss": m.val_loss,
            "train_time_ms": m.train_time_ms,
            "step_avg_ms": m.step_avg_ms,
        }
        for m in metrics
        if m.val_loss is not None
    ]

    # Get final metric
    final = get_final_metric(exp_id)
    latest = get_latest_metric(exp_id)

    return {
        "id": exp.id,
        "name": exp.name,
        "track": exp.track,
        "script": exp.script,
        "code_hash": exp.code_hash,
        "remote_run_id": exp.remote_run_id,
        "status": exp.status,
        "gpus": exp.gpus,
        "gpu_type": exp.gpu_type,
        "env_vars": exp.env_vars,
        "git_commit": exp.git_commit,
        "tmux_window": exp.tmux_window,
        "crash_log": exp.crash_log,
        "started_at": exp.started_at.isoformat() if exp.started_at else None,
        "finished_at": exp.finished_at.isoformat() if exp.finished_at else None,
        # Metrics summary
        "current_step": latest.step if latest else None,
        "total_steps": latest.total_steps if latest else None,
        "final_val_loss": final.val_loss if final else (latest.val_loss if latest else None),
        "final_train_time_ms": final.train_time_ms if final else (latest.train_time_ms if latest else None),
        # Full loss curve for plotting
        "loss_curve": loss_curve,
        "metrics_count": len(metrics),
    }


@app.get("/api/experiment/{exp_id}/metrics")
async def get_experiment_metrics(exp_id: int):
    """Get all metrics for an experiment (for detailed analysis)."""
    exp = get_experiment(exp_id)
    if not exp:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)

    metrics = get_metrics(exp_id)
    return {
        "experiment_id": exp_id,
        "metrics": [
            {
                "step": m.step,
                "total_steps": m.total_steps,
                "val_loss": m.val_loss,
                "train_time_ms": m.train_time_ms,
                "step_avg_ms": m.step_avg_ms,
                "is_final_step": m.is_final_step,
                "recorded_at": m.recorded_at.isoformat() if m.recorded_at else None,
            }
            for m in metrics
        ],
    }


@app.delete("/api/experiment/{exp_id}")
async def delete_experiment_endpoint(exp_id: int):
    """Delete an experiment and all its metrics."""
    from ..tracker import delete_experiment

    exp = get_experiment(exp_id)
    if not exp:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)

    try:
        delete_experiment(exp_id)
        return {"success": True, "message": f"Deleted experiment {exp_id}"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/tracks")
async def list_tracks():
    """List all experiment tracks, sorted by most recent experiment."""
    tracks = discover_tracks()

    # Get most recent experiment time per track
    conn = get_db()
    rows = conn.execute(
        """SELECT track, MAX(started_at) as latest
           FROM experiments
           WHERE (deleted IS NULL OR deleted = 0) AND track IS NOT NULL
           GROUP BY track"""
    ).fetchall()
    conn.close()
    recency = {row["track"]: row["latest"] or "" for row in rows}

    track_list = [
        {
            "name": t.name,
            "directory": t.directory,
            "description": t.description,
        }
        for t in tracks
    ]
    track_list.sort(key=lambda t: recency.get(t["name"], ""), reverse=True)

    return {"tracks": track_list}


@app.get("/api/logs/{run_id}")
async def get_log_file(run_id: str):
    """Get the log file content for a remote run."""
    logs_dir = Config.get_config_dir() / "logs"
    # Logs are stored per-session: logs/{session}/{run_id}.txt
    # Also check flat dir for pre-migration logs
    log_file = None
    flat = logs_dir / f"{run_id}.txt"
    if flat.exists():
        log_file = flat
    else:
        for session_dir in logs_dir.iterdir():
            if not session_dir.is_dir():
                continue
            candidate = session_dir / f"{run_id}.txt"
            if candidate.exists():
                log_file = candidate
                break

    if not log_file:
        return JSONResponse({"error": f"Log file not found: {run_id}.txt"}, status_code=404)

    try:
        content = log_file.read_text()
        return PlainTextResponse(content)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/diff/{code_hash}")
async def get_diff_file(code_hash: str):
    """Get the diff file content for a code hash."""
    diffs_dir = Config.get_config_dir() / "diffs"
    diff_file = diffs_dir / f"{code_hash}.diff"

    if not diff_file.exists():
        return JSONResponse({"error": f"Diff not found for code hash: {code_hash}"}, status_code=404)

    try:
        content = diff_file.read_text()
        return PlainTextResponse(content)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/notes/{script_path:path}")
async def get_script_notes(script_path: str):
    """Get the notes file content for a script.

    Notes are stored as sidecar files: {script_name}.notes.md
    For example: experiments/records/train_gpt_record52.py -> train_gpt_record52.notes.md
    """
    # Get the repo root
    repo_root = Path(__file__).parent.parent.parent
    script_file = repo_root / script_path

    if not script_file.exists():
        return JSONResponse({"error": f"Script not found: {script_path}"}, status_code=404)

    # Build notes file path: same directory, {stem}.notes.md
    notes_file = script_file.parent / f"{script_file.stem}.notes.md"

    if not notes_file.exists():
        return Response(status_code=204)

    try:
        content = notes_file.read_text()
        return PlainTextResponse(content)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/crash/{exp_id}")
async def get_crash_log(exp_id: int):
    """Get the crash log for an experiment."""
    exp = get_experiment(exp_id)
    if not exp:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)

    if not exp.crash_log:
        return JSONResponse({"error": "No crash log for this experiment"}, status_code=404)

    return PlainTextResponse(exp.crash_log)


@app.get("/api/env-defaults/{script_path:path}")
async def get_env_defaults(script_path: str):
    """Parse env var defaults from a script's os.environ.get() calls.

    Looks for patterns like:
        os.environ.get("KEY", "default_value")
        os.environ.get('KEY', 'default_value')
        os.environ.get("KEY", default_value)  (unquoted int/float)
    """
    import re

    repo_root = Path(__file__).parent.parent.parent
    script_file = repo_root / script_path

    if not script_file.exists():
        return JSONResponse({"error": f"Script not found: {script_path}"}, status_code=404)

    try:
        content = script_file.read_text()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # Match os.environ.get("KEY", "default") or os.environ.get("KEY", default)
    pattern = r"""os\.environ\.get\(\s*['"](\w+)['"]\s*,\s*(['"]?)(.+?)\2\s*\)"""
    defaults = {}
    for match in re.finditer(pattern, content):
        key = match.group(1)
        value = match.group(3).strip()
        # Skip non-hyperparameter env vars (torch internals, paths, ranks)
        if key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR',
                    'MASTER_PORT', 'DATA_PATH', 'PYTORCH_ALLOC_CONF'):
            continue
        # Deduplicate: first occurrence wins (top-level default)
        if key not in defaults:
            defaults[key] = value

    return {"defaults": defaults, "script": script_path}


@app.post("/api/reveal/{exp_id}")
async def reveal_in_finder(exp_id: int):
    """Reveal the experiment's script file in Finder."""
    import subprocess

    exp = get_experiment(exp_id)
    if not exp:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)

    if not exp.script:
        return JSONResponse({"error": "No script path for this experiment"}, status_code=404)

    # Get the repo root (where nanorun-platform is)
    repo_root = Path(__file__).parent.parent.parent
    script_path = repo_root / exp.script

    if not script_path.exists():
        return JSONResponse({"error": f"Script file not found: {exp.script}"}, status_code=404)

    try:
        subprocess.run(["open", "-R", str(script_path)], check=True)
        return {"success": True, "path": str(script_path)}
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": f"Failed to reveal file: {e}"}, status_code=500)


def run_dashboard(port: int = 8080, open_browser: bool = True):
    """Run the dashboard server."""
    import uvicorn

    print(f"Starting dashboard at http://localhost:{port}")

    if open_browser:
        webbrowser.open(f"http://localhost:{port}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
