"""FastAPI dashboard application."""

import time
import webbrowser
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import discover_tracks, Config
from ..queue import get_queue_state
from ..local_daemon import DASHBOARD_HOST, safe_json_load, get_queue_cache_file
from ..tracker import (
    get_experiments,
    get_experiment,
    get_running_experiments,
    get_metrics,
    get_latest_metric,
    get_final_metric,
    get_loss_metrics,
    get_db,
    get_crash_log as get_crash_log_content,
)

app = FastAPI(title="nanorun Dashboard")

# Setup templates and static files
DASHBOARD_DIR = Path(__file__).parent
REPO_ROOT = DASHBOARD_DIR.parent.parent.resolve()
templates = Jinja2Templates(directory=DASHBOARD_DIR / "templates")
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR / "static"), name="static")

# Cache buster: changes on each server start
BOOT_VERSION = str(int(time.time()))


def _resolve_within(root: Path, untrusted_path: str) -> Optional[Path]:
    """Resolve a user-controlled path only when it remains inside ``root``."""
    resolved_root = root.resolve()
    candidate = (resolved_root / untrusted_path).resolve()
    return candidate if candidate.is_relative_to(resolved_root) else None


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
    """Fetch latest metric for each experiment in a single query.

    Returns the current step (MAX step) plus the latest value of each loss
    series, which may come from an earlier step than the max: val_loss only
    lands on eval steps, and train_loss only on steps the script logged one.

    ``loss``/``loss_metric`` collapse the two series into "the number this run
    is actually judged on" — validation when the run reports it, training loss
    when validation is all the run ever had.
    """
    if not experiment_ids:
        return {}
    conn = get_db()
    placeholders = ",".join("?" for _ in experiment_ids)
    # Get latest step (for progress), latest eval step (val_loss/train_time),
    # and latest train-loss step — each series tracked independently.
    rows = conn.execute(
        f"""SELECT
                latest.experiment_id,
                latest.max_step as step,
                latest.total_steps,
                eval.val_loss,
                eval.train_time_ms,
                tl.train_loss,
                COALESCE(eval.step_avg_ms, latest.step_avg_ms) AS step_avg_ms
            FROM (
                SELECT m.experiment_id, m.total_steps, m.step_avg_ms, sub.max_step
                FROM metrics m
                INNER JOIN (
                    SELECT experiment_id, MAX(step) as max_step
                    FROM metrics
                    WHERE experiment_id IN ({placeholders})
                    GROUP BY experiment_id
                ) sub ON m.experiment_id = sub.experiment_id AND m.step = sub.max_step
            ) latest
            LEFT JOIN (
                SELECT m2.experiment_id, m2.val_loss, m2.train_time_ms, m2.step_avg_ms
                FROM metrics m2
                INNER JOIN (
                    SELECT experiment_id, MAX(step) as max_eval_step
                    FROM metrics
                    WHERE experiment_id IN ({placeholders}) AND val_loss IS NOT NULL
                    GROUP BY experiment_id
                ) esub ON m2.experiment_id = esub.experiment_id AND m2.step = esub.max_eval_step
            ) eval ON latest.experiment_id = eval.experiment_id
            LEFT JOIN (
                SELECT m3.experiment_id, m3.train_loss
                FROM metrics m3
                INNER JOIN (
                    SELECT experiment_id, MAX(step) as max_train_step
                    FROM metrics
                    WHERE experiment_id IN ({placeholders}) AND train_loss IS NOT NULL
                    GROUP BY experiment_id
                ) tsub ON m3.experiment_id = tsub.experiment_id AND m3.step = tsub.max_train_step
            ) tl ON latest.experiment_id = tl.experiment_id""",
        experiment_ids + experiment_ids + experiment_ids,
    ).fetchall()
    return {
        row["experiment_id"]: {
            "step": row["step"],
            "total_steps": row["total_steps"],
            "val_loss": row["val_loss"],
            "train_time_ms": row["train_time_ms"],
            "train_loss": row["train_loss"],
            "step_avg_ms": row["step_avg_ms"],
            "loss": row["val_loss"] if row["val_loss"] is not None else row["train_loss"],
            "loss_metric": "val_loss" if row["val_loss"] is not None
                           else ("train_loss" if row["train_loss"] is not None else None),
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
                "train_loss": m["train_loss"] if m else None,
                "loss": m["loss"] if m else None,
                "loss_metric": m["loss_metric"] if m else None,
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

        # Collect metrics for all experiments in group. Runs are grouped by
        # code_hash, so every member is the same script and therefore reports
        # the same loss series — averaging `loss` across the group is safe.
        val_losses = []
        train_times = []
        loss_metrics = set()
        env_var_sets = set()
        experiment_ids = []
        statuses = []

        for exp in group_exps:
            experiment_ids.append(exp.id)
            statuses.append(exp.status)
            env_var_sets.add(json.dumps(exp.env_vars, sort_keys=True))
            m = latest_metrics.get(exp.id)
            if m:
                if m["loss"] is not None:
                    val_losses.append(m["loss"])
                    loss_metrics.add(m["loss_metric"])
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
            "loss": mean_val_loss,
            # Single metric for the group, or None if members disagree (only
            # possible if a script changed which series it logs without a code
            # change — treat as unknown rather than silently mixing them).
            "loss_metric": next(iter(loss_metrics)) if len(loss_metrics) == 1 else None,
            "train_time_ms": mean_train_time,
            "val_losses": val_losses,  # Individual values for details
            "losses": val_losses,
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
            "train_loss": latest.train_loss if latest else None,
            "loss": latest.loss if latest else None,
            "loss_metric": latest.loss_metric if latest else None,
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
            "train_loss": latest.train_loss if latest else None,
            "loss": latest.loss if latest else None,
            "loss_metric": latest.loss_metric if latest else None,
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
        if sc.session_type == "iris":
            host = "iris controller"
            status = "iris"
        elif sc.session_type == "local":
            host = "this device"
            status = state.status
        elif getattr(sc, "bootstrap", False):
            # Provision-only: never tracked, so "disconnected" would be noise.
            host = f"{sc.user}@{sc.host}:{sc.port}"
            status = "bootstrap"
        else:
            host = f"{sc.user}@{sc.host}:{sc.port}"
            status = state.status
        result.append({
            "name": sc.name,
            "session_type": sc.session_type,
            "bootstrap": getattr(sc, "bootstrap", False),
            "host": host,
            "gpu_type": sc.gpu_type,
            "gpu_count": sc.gpu_count,
            "status": status,
            "git_branch": sc.git_branch if sc.session_type == "local" else None,
            "hub_namespace": (
                sc.hub_namespace if sc.session_type == "local" else None
            ),
            "sync_paused": getattr(sc, "sync_paused", False),
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

    session_config = Config.load_session(name)
    if not session_config:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    if session_config.session_type == "local":
        from ..remote_control import local_session_removal_blocker

        blocker = local_session_removal_blocker(session_config)
        if blocker:
            return JSONResponse(
                {"error": f"Cannot remove local session: {blocker}"},
                status_code=400,
            )

    state = SessionState.load(name)
    if state.status == "connected":
        return JSONResponse(
            {"error": "Cannot remove a connected session. Disconnect first."},
            status_code=400,
        )
    # Removing local connection metadata does not prove remote work stopped.
    removed, _ = Config.delete_session(name)
    if not removed:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    state_dir = Config.get_sessions_dir() / name
    if state_dir.exists():
        shutil.rmtree(state_dir, ignore_errors=True)
    daemon = getattr(app.state, "daemon", None)
    if daemon and hasattr(daemon, "remove_session"):
        daemon.remove_session(name)
    msg = f"Session '{name}' removed"
    return {"success": True, "message": msg}


@app.post("/api/sessions/{name}/sync-pause")
async def set_session_sync_pause(name: str, paused: bool = True):
    """Pause or resume the local daemon's background scanning for a session.

    Persists the per-session `sync_paused` flag. The HubSyncer skips paused
    sessions, and SSH SessionTrackers idle (dropping their connection) until
    resumed. On-demand commands still work. Takes effect within one sync cycle.
    """
    if not Config.set_session_paused(name, paused):
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return {
        "success": True,
        "paused": paused,
        "message": "Sync paused" if paused else "Sync resumed",
    }


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


@app.get("/api/metrics/version")
async def get_metrics_version(experiment_ids: Optional[str] = None):
    """Lightweight endpoint returning metric counts for change detection.

    If experiment_ids is provided (comma-separated), returns per-experiment counts.
    Otherwise returns global total.
    """
    conn = get_db()
    if experiment_ids:
        ids = [int(x) for x in experiment_ids.split(",") if x.strip()]
        if not ids:
            return {"version": 0, "counts": {}}
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT experiment_id, COUNT(*) as cnt FROM metrics WHERE experiment_id IN ({placeholders}) GROUP BY experiment_id",
            ids,
        ).fetchall()
        counts = {str(row["experiment_id"]): row["cnt"] for row in rows}
        version = sum(counts.values())
        return {"version": version, "counts": counts}
    else:
        row = conn.execute("SELECT COUNT(*) as cnt FROM metrics").fetchone()
        return {"version": row["cnt"]}


@app.get("/api/experiment/{exp_id}")
async def get_experiment_detail(exp_id: int):
    """Get detailed data for a single experiment including loss curve."""
    exp = get_experiment(exp_id)
    if not exp:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)

    # Get all metrics for loss curve. The curve follows the run's primary
    # series only — a run that logs train_loss every step and val_loss every
    # 125th would otherwise produce a curve alternating between two different
    # quantities. `val_loss` stays on each point for backward compatibility.
    metrics = get_metrics(exp_id)
    loss_metric = get_loss_metrics([exp_id]).get(exp_id)
    loss_curve = [
        {
            "step": m.step,
            "loss": getattr(m, loss_metric),
            "val_loss": m.val_loss,
            "train_loss": m.train_loss,
            "train_time_ms": m.train_time_ms,
            "step_avg_ms": m.step_avg_ms,
        }
        for m in metrics
        if loss_metric and getattr(m, loss_metric) is not None
    ]

    # Get final metric
    final = get_final_metric(exp_id)
    latest = get_latest_metric(exp_id)

    # For loss/train_time: prefer final step, then latest point on the curve
    if final and final.loss is not None:
        summary_val_loss = final.loss
        summary_train_time = final.train_time_ms
    elif loss_curve:
        summary_val_loss = loss_curve[-1]["loss"]
        summary_train_time = loss_curve[-1]["train_time_ms"]
    else:
        summary_val_loss = latest.loss if latest else None
        summary_train_time = latest.train_time_ms if latest else None

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
        "parent_hash": exp.parent_hash,
        "kernels_path": exp.kernels_path,
        "dependencies": exp.dependencies,
        "tmux_window": exp.tmux_window,
        "crash_log": get_crash_log_content(exp.id),
        "session_name": exp.session_name,
        "started_at": exp.started_at.isoformat() if exp.started_at else None,
        "finished_at": exp.finished_at.isoformat() if exp.finished_at else None,
        # Metrics summary
        "current_step": latest.step if latest else None,
        "total_steps": latest.total_steps if latest else None,
        "final_val_loss": summary_val_loss,
        "final_loss": summary_val_loss,
        "loss_metric": loss_metric,
        "final_train_time_ms": summary_train_time,
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
        "loss_metric": get_loss_metrics([exp_id]).get(exp_id),
        "metrics": [
            {
                "step": m.step,
                "total_steps": m.total_steps,
                "val_loss": m.val_loss,
                "train_loss": m.train_loss,
                "loss": m.loss,
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


@app.get("/api/logs/{run_id:path}")
async def get_log_file(run_id: str):
    """Get the log file content for a remote run."""
    from ..hub import _iris_job_id_to_filename

    logs_dir = Config.get_config_dir() / "logs"
    # Logs are stored per-session: logs/{session}/{run_id}.txt
    # Also check flat dir for pre-migration logs
    # Try direct name, then legacy iris job ID encoding for old experiments
    candidates = [run_id, _iris_job_id_to_filename(run_id)]

    if _resolve_within(logs_dir, f"{run_id}.txt") is None:
        return JSONResponse({"error": "Invalid log path"}, status_code=400)

    log_file = None
    for rid in candidates:
        flat = _resolve_within(logs_dir, f"{rid}.txt")
        if flat and flat.is_file():
            log_file = flat
            break
        for session_dir in logs_dir.iterdir():
            if not session_dir.is_dir():
                continue
            relative_path = Path(session_dir.name) / f"{rid}.txt"
            candidate = _resolve_within(logs_dir, str(relative_path))
            if candidate and candidate.is_file():
                log_file = candidate
                break
        if log_file:
            break

    if not log_file:
        return JSONResponse({"error": f"Log file not found: {run_id}"}, status_code=404)

    return FileResponse(str(log_file), media_type="text/plain")


@app.get("/api/diff/{code_hash}")
async def get_diff_file(code_hash: str):
    """Get the diff file content for a code hash."""
    diffs_dir = Config.get_config_dir() / "diffs"
    diff_file = _resolve_within(diffs_dir, f"{code_hash}.diff")

    if diff_file is None:
        return JSONResponse({"error": "Invalid diff path"}, status_code=400)

    if not diff_file.is_file():
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
    script_file = _resolve_within(REPO_ROOT, script_path)

    if script_file is None:
        return JSONResponse({"error": "Invalid script path"}, status_code=400)

    if not script_file.is_file():
        return JSONResponse({"error": f"Script not found: {script_path}"}, status_code=404)

    # Build notes file path: same directory, {stem}.notes.md
    notes_relative = (
        script_file.parent.relative_to(REPO_ROOT.resolve())
        / f"{script_file.stem}.notes.md"
    )
    notes_file = _resolve_within(REPO_ROOT, str(notes_relative))

    if notes_file is None:
        return JSONResponse({"error": "Invalid notes path"}, status_code=400)

    if not notes_file.is_file():
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

    content = get_crash_log_content(exp_id)
    if not content:
        return JSONResponse({"error": "No crash log for this experiment"}, status_code=404)

    return PlainTextResponse(content)


@app.get("/api/env-defaults/{script_path:path}")
async def get_env_defaults(script_path: str):
    """Parse env var defaults from a script's os.environ.get() calls.

    Looks for patterns like:
        os.environ.get("KEY", "default_value")
        os.environ.get('KEY', 'default_value')
        os.environ.get("KEY", default_value)  (unquoted int/float)
    """
    import re

    script_file = _resolve_within(REPO_ROOT, script_path)

    if script_file is None:
        return JSONResponse({"error": "Invalid script path"}, status_code=400)

    if not script_file.is_file():
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

    script_path = _resolve_within(REPO_ROOT, exp.script)

    if script_path is None:
        return JSONResponse({"error": "Invalid script path"}, status_code=400)

    if not script_path.is_file():
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

    uvicorn.run(app, host=DASHBOARD_HOST, port=port, log_level="warning")
