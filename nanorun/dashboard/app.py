"""FastAPI dashboard application."""

import asyncio
import json
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import discover_tracks, Config
from ..queue import get_queue_state
from ..watcher import DASHBOARD_HOST, safe_json_load, get_queue_cache_file
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
    get_dashboard_event_bounds,
    get_dashboard_events,
    append_dashboard_event,
)

app = FastAPI(title="nanorun Dashboard")

# Setup templates and static files
DASHBOARD_DIR = Path(__file__).parent
REPO_ROOT = DASHBOARD_DIR.parent.parent.resolve()
templates = Jinja2Templates(directory=DASHBOARD_DIR / "templates")
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR / "static"), name="static")

# Cache buster: changes on each server start
BOOT_VERSION = str(int(time.time()))
SSE_HEARTBEAT_SECONDS = 15.0
SSE_POLL_SECONDS = 0.1
SSE_REPLAY_BATCH = 100


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


def _format_sse(event: dict) -> str:
    """Serialize one durable dashboard event as a standard SSE record."""
    return (
        f"id: {event['id']}\n"
        f"event: {event['event_type']}\n"
        f"data: {json.dumps(event['payload'], separators=(',', ':'))}\n\n"
    )


def _requested_event_id(request: Request, after: Optional[int]) -> int:
    """Resolve explicit and native EventSource replay cursors safely."""
    candidates = [after if after is not None else 0]
    header = request.headers.get("last-event-id")
    if header:
        try:
            candidates.append(int(header))
        except ValueError:
            pass
    return max(0, *candidates)


@app.get("/api/events")
async def dashboard_events(request: Request, after: Optional[int] = None):
    """Replay retained projection changes, then stream newly committed ones."""
    requested_after = _requested_event_id(request, after)

    async def replay_stream():
        cursor = requested_after
        last_output = time.monotonic()
        while True:
            if await request.is_disconnected():
                return

            oldest, newest = await asyncio.to_thread(get_dashboard_event_bounds)
            if oldest and cursor < oldest - 1:
                reset = {
                    "id": newest,
                    "event_type": "dashboard.reset",
                    "payload": {
                        "reason": "replay_unavailable",
                        "oldest_event_id": oldest,
                        "last_event_id": newest,
                        "revision": newest,
                    },
                }
                yield _format_sse(reset)
                return

            events = await asyncio.to_thread(
                get_dashboard_events, cursor, SSE_REPLAY_BATCH,
            )
            if events:
                for event in events:
                    cursor = event["id"]
                    yield _format_sse(event)
                last_output = time.monotonic()
                continue

            now = time.monotonic()
            if now - last_output >= SSE_HEARTBEAT_SECONDS:
                # Comments keep the connection alive without advancing replay ID.
                yield ": heartbeat\n\n"
                last_output = now
            await asyncio.sleep(SSE_POLL_SECONDS)

    async def stream():
        app.state.sse_connected_clients = (
            getattr(app.state, "sse_connected_clients", 0) + 1
        )
        try:
            async for record in replay_stream():
                yield record
        finally:
            app.state.sse_connected_clients = max(
                0, getattr(app.state, "sse_connected_clients", 1) - 1,
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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


def _flat_experiment_summaries(
    track: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    revision: int = 0,
) -> list[dict]:
    """Return the normalized experiment state used by snapshots and patches."""
    experiments = get_experiments(
        track=track, status=status, search=search, limit=2000,
    )
    latest_metrics = _batch_latest_metrics([exp.id for exp in experiments])
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
            "session_name": exp.session_name,
            "remote_run_id": exp.remote_run_id,
            "started_at": exp.started_at.isoformat() if exp.started_at else None,
            "finished_at": exp.finished_at.isoformat() if exp.finished_at else None,
            "current_step": m["step"] if m else None,
            "total_steps": m["total_steps"] if m else None,
            "val_loss": m["val_loss"] if m else None,
            "train_loss": m["train_loss"] if m else None,
            "loss": m["loss"] if m else None,
            "loss_metric": m["loss_metric"] if m else None,
            "train_time_ms": m["train_time_ms"] if m else None,
            "revision": revision,
            "group": {
                "code_hash": exp.code_hash,
                "track": exp.track or "",
                "gpus": exp.gpus,
                "gpu_type": exp.gpu_type or "H100",
            },
        })
    return results


def _aggregate_experiment_summaries(
    summaries: list[dict],
    limit: int = 100,
) -> list[dict]:
    """Aggregate normalized rows with the dashboard's stable group identity."""
    from collections import defaultdict

    groups = defaultdict(list)
    for exp in summaries:
        group = exp["group"]
        hash_key = group["code_hash"] or f"_no_hash_{exp['id']}"
        key = (hash_key, group["track"], group["gpus"], group["gpu_type"])
        groups[key].append(exp)

    results = []
    for group_exps in groups.values():
        group_exps.sort(key=lambda e: e["started_at"] or "", reverse=True)
        primary = group_exps[0]
        losses = [e["loss"] for e in group_exps if e["loss"] is not None]
        train_times = [
            e["train_time_ms"] for e in group_exps
            if e["train_time_ms"] is not None
        ]
        loss_metrics = {
            e["loss_metric"] for e in group_exps if e["loss"] is not None
        }
        statuses = [e["status"] for e in group_exps]
        if "running" in statuses:
            aggregate_status = "running"
        elif "completed" in statuses:
            aggregate_status = "completed"
        else:
            aggregate_status = statuses[0] if statuses else "unknown"
        with_metrics = next(
            (e for e in group_exps if e["current_step"] is not None), None,
        )
        results.append({
            "id": primary["id"],
            "experiment_ids": [e["id"] for e in group_exps],
            "name": primary["name"],
            "track": primary["track"],
            "script": primary["script"],
            "code_hash": primary["code_hash"],
            "status": aggregate_status,
            "gpus": primary["gpus"],
            "gpu_type": primary["gpu_type"],
            "env_vars": primary["env_vars"],
            "started_at": primary["started_at"],
            "n_runs": len(group_exps),
            "is_sweep": len({
                json.dumps(e["env_vars"], sort_keys=True) for e in group_exps
            }) > 1,
            "current_step": with_metrics["current_step"] if with_metrics else None,
            "total_steps": with_metrics["total_steps"] if with_metrics else None,
            "val_loss": sum(losses) / len(losses) if losses else None,
            "loss": sum(losses) / len(losses) if losses else None,
            "loss_metric": next(iter(loss_metrics)) if len(loss_metrics) == 1 else None,
            "train_time_ms": (
                sum(train_times) / len(train_times) if train_times else None
            ),
            "val_losses": losses,
            "losses": losses,
            "train_times": train_times,
            "group": primary["group"],
            "revision": max(e.get("revision", 0) for e in group_exps),
        })
    results.sort(key=lambda result: result["started_at"] or "", reverse=True)
    return results[:max(1, min(limit, 2000))]


@app.get("/api/experiments")
async def list_experiments(
    track: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    aggregate: bool = True,
):
    """List experiments, optionally aggregated by code hash and hardware."""
    summaries = _flat_experiment_summaries(track, status, search)
    if not aggregate:
        return {"experiments": summaries[:max(1, min(limit, 2000))]}
    return {"experiments": _aggregate_experiment_summaries(summaries, limit)}


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

    return _get_queue_status()


def _get_queue_status(session_name: Optional[str] = None) -> dict:
    """Build queue projection globally or for exactly one session."""
    sessions = Config.list_sessions()
    if session_name is not None:
        sessions = [session for session in sessions if session.name == session_name]
    running_list = []
    queued_list = []

    running_exps = get_running_experiments(session_name=session_name)
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


@app.get("/api/queue/{session_name}")
async def get_session_queue_status(session_name: str):
    """Return only one session's running and queued projection."""
    if Config.load_session(session_name) is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return _get_queue_status(session_name)


def _session_summary(sc, watcher=None) -> dict:
    """Build the patchable dashboard representation for one session."""
    from ..watcher import SessionState

    state = SessionState.load(sc.name)
    if sc.session_type == "iris":
        host = "iris controller"
        status = "iris"
    elif sc.session_type == "local":
        host = "this device"
        status = state.status
    elif getattr(sc, "bootstrap", False):
        host = f"{sc.user}@{sc.host}:{sc.port}"
        status = "bootstrap"
    else:
        host = f"{sc.user}@{sc.host}:{sc.port}"
        status = state.status
    return {
        "name": sc.name,
        "session_type": sc.session_type,
        "bootstrap": getattr(sc, "bootstrap", False),
        "host": host,
        "gpu_type": sc.gpu_type,
        "gpu_count": sc.gpu_count,
        "status": status,
        "git_branch": sc.git_branch if sc.session_type == "local" else None,
        "hub_namespace": (
            sc.hub_namespace
            if sc.session_type == "local" or getattr(sc, "bootstrap", False)
            else None
        ),
        "sync_paused": getattr(sc, "sync_paused", False),
        "last_error": state.last_error,
        "metrics_synced": state.metrics_synced,
        "tracking_experiment_id": state.tracking_experiment_id,
    }


@app.get("/api/sessions")
async def get_sessions():
    """Get session statuses + hub syncer state, sorted disconnected-first."""
    sessions = Config.list_sessions()
    watcher = getattr(app.state, "watcher", None)
    hub = watcher.hub_syncer if watcher else None
    result = [_session_summary(sc, watcher) for sc in sessions]
    result.sort(key=lambda s: (0 if s["status"] == "disconnected" else 1, s["name"]))
    return {
        "sessions": result,
        "hub": {
            "status": hub.status if hub else "unknown",
            "last_error": hub.last_error if hub else None,
            "last_sync_at": hub.last_sync_at if hub else None,
        },
    }


@app.get("/api/sessions/{name}")
async def get_session(name: str):
    """Return one session chip/popover projection for a targeted patch."""
    sc = Config.load_session(name)
    if not sc:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return {"session": _session_summary(sc, getattr(app.state, "watcher", None))}


@app.get("/api/dashboard/snapshot")
async def get_dashboard_snapshot():
    """Return normalized initial state plus a race-closing replay cursor.

    The cursor is read first. A change committed later is therefore either
    already visible in the following reads or replayed after this cursor (and a
    harmless duplicate is possible), but it cannot fall into a connection gap.
    """
    oldest_event_id, last_event_id = get_dashboard_event_bounds()
    summaries = _flat_experiment_summaries(revision=last_event_id)
    queue = _get_queue_status()
    sessions = await get_sessions()
    return {
        "experiment_summaries": summaries,
        "experiments": _aggregate_experiment_summaries(summaries),
        "queue": queue,
        "sessions": sessions["sessions"],
        "hub": sessions["hub"],
        "oldest_event_id": oldest_event_id,
        "last_event_id": last_event_id,
    }


@app.post("/api/sessions/{name}/reconnect")
async def reconnect_session(name: str):
    """Trigger a reconnect attempt for a disconnected session."""
    watcher = getattr(app.state, "watcher", None)
    if not watcher:
        return JSONResponse({"error": "Watcher not available"}, status_code=503)
    ok = watcher.reconnect_session(name)
    return {"success": ok, "message": "Reconnecting..." if ok else "Session not found"}


@app.post("/api/hub/reconnect")
async def reconnect_hub():
    """Trigger a reconnect attempt for the hub syncer."""
    watcher = getattr(app.state, "watcher", None)
    if not watcher:
        return JSONResponse({"error": "Watcher not available"}, status_code=503)
    ok = watcher.reconnect_hub()
    return {"success": ok, "message": "Hub reconnecting..." if ok else "Hub syncer already running"}


@app.delete("/api/sessions/{name}")
async def delete_session(name: str):
    """Remove a session and retire its in-flight dashboard state.

    Removing a machine from the dashboard is explicit administrative intent,
    unlike a transient disconnect. Treat its queued and running experiments as
    cancelled before deleting the connection metadata and queue cache.
    """
    import shutil
    from ..tracker import terminate_session_experiments
    from ..watcher import SessionState

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

    running_ids, queued_ids = terminate_session_experiments(
        name,
        note=f"Session '{name}' removed from dashboard; machine treated as terminated.",
        session_id=session_config.session_id,
    )
    removed, _ = Config.delete_session(name)
    if not removed:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    state_dir = Config.get_sessions_dir() / name
    if state_dir.exists():
        shutil.rmtree(state_dir, ignore_errors=True)
    watcher = getattr(app.state, "watcher", None)
    if watcher and hasattr(watcher, "remove_session"):
        watcher.remove_session(name)
    append_dashboard_event(
        "session.changed", name,
        {"session_name": name, "deleted": True},
    )
    cancelled_count = len(running_ids) + len(queued_ids)
    msg = f"Session '{name}' removed"
    if cancelled_count:
        msg += f"; cancelled {cancelled_count} in-flight experiment(s)"
    return {
        "success": True,
        "message": msg,
        "cancelled_running": len(running_ids),
        "cancelled_queued": len(queued_ids),
    }


@app.post("/api/sessions/{name}/sync-pause")
async def set_session_sync_pause(name: str, paused: bool = True):
    """Pause or resume the watcher's background scanning for a session.

    Persists the per-session `sync_paused` flag. The HubSyncer skips paused
    sessions, and SSH SessionTrackers idle (dropping their connection) until
    resumed. On-demand commands still work. Takes effect within one sync cycle.
    """
    if not Config.set_session_paused(name, paused):
        return JSONResponse({"error": "Session not found"}, status_code=404)
    append_dashboard_event(
        "session.changed", name,
        {"session_name": name, "sync_paused": paused},
    )
    return {
        "success": True,
        "paused": paused,
        "message": "Sync paused" if paused else "Sync resumed",
    }


_BOOTSTRAP_DAEMON_MESSAGE = (
    "This is a bootstrap (provision-only) session: its execution daemon is "
    "owned by the machine's own local session and is followed via the hub."
)


def _bootstrap_daemon_block(name: str) -> Optional[JSONResponse]:
    """Refuse daemon operations on bootstrap sessions (provision-only)."""
    sc = Config.load_session(name)
    if sc and getattr(sc, "bootstrap", False):
        return JSONResponse({"error": _BOOTSTRAP_DAEMON_MESSAGE}, status_code=400)
    return None


@app.post("/api/sessions/{name}/daemon-restart")
async def restart_daemon(name: str):
    """Restart the daemon for a session (stop + start)."""
    import threading
    from ..remote_control import get_daemon_client, DaemonError

    blocked = _bootstrap_daemon_block(name)
    if blocked:
        return blocked

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
    """Get daemon status (experiment, queue, GPU) for a connected session."""
    from ..queue import get_daemon_status

    blocked = _bootstrap_daemon_block(name)
    if blocked:
        return blocked
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


def _bounded_loss_curve(
    experiment_id: int,
    metric_name: str,
    max_points: int,
) -> list[dict]:
    """Return a min/max-preserving SQL-bounded loss series."""
    if metric_name not in {"val_loss", "train_loss"}:
        raise ValueError(f"Unsupported loss metric: {metric_name}")
    max_points = max(4, min(max_points, 1200))
    conn = get_db()
    stats = conn.execute(
        f"SELECT COUNT(*) AS count, MIN(step) AS min_step, MAX(step) AS max_step "
        f"FROM metrics WHERE experiment_id = ? AND {metric_name} IS NOT NULL",
        (experiment_id,),
    ).fetchone()
    if not stats["count"]:
        return []

    columns = (
        f"step, {metric_name} AS loss, val_loss, train_loss, "
        "train_time_ms, step_avg_ms"
    )
    if stats["count"] <= max_points:
        rows = conn.execute(
            f"SELECT {columns} FROM metrics WHERE experiment_id = ? "
            f"AND {metric_name} IS NOT NULL ORDER BY step",
            (experiment_id,),
        ).fetchall()
    else:
        bucket_count = max(1, (max_points - 2) // 2)
        rows = conn.execute(
            f"""
            WITH series AS (
                SELECT {columns}
                FROM metrics
                WHERE experiment_id = ? AND {metric_name} IS NOT NULL
            ), bounds AS (
                SELECT MIN(step) AS min_step, MAX(step) AS max_step FROM series
            ), bucketed AS (
                SELECT series.*,
                    CAST(
                        (series.step - bounds.min_step - 1) * ? /
                        MAX(1, bounds.max_step - bounds.min_step - 1)
                        AS INTEGER
                    ) AS bucket
                FROM series CROSS JOIN bounds
                WHERE series.step > bounds.min_step
                  AND series.step < bounds.max_step
            ), ranked AS (
                SELECT bucketed.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY bucket ORDER BY loss ASC, step ASC
                    ) AS min_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY bucket ORDER BY loss DESC, step ASC
                    ) AS max_rank
                FROM bucketed
            )
            SELECT series.* FROM series CROSS JOIN bounds
            WHERE series.step IN (bounds.min_step, bounds.max_step)
            UNION
            SELECT step, loss, val_loss, train_loss, train_time_ms, step_avg_ms
            FROM ranked WHERE min_rank = 1 OR max_rank = 1
            ORDER BY step
            """,
            (experiment_id, bucket_count),
        ).fetchall()

    return [
        {
            "step": row["step"],
            "loss": row["loss"],
            "val_loss": row["val_loss"],
            "train_loss": row["train_loss"],
            "train_time_ms": row["train_time_ms"],
            "step_avg_ms": row["step_avg_ms"],
        }
        for row in rows[:max_points]
    ]


def _bounded_experiment_detail(
    exp_id: int,
    max_points: int,
    requested_loss_metric: Optional[str] = None,
) -> Optional[dict]:
    """Build chart-ready detail without materializing unbounded histories."""
    exp = get_experiment(exp_id)
    if not exp:
        return None
    if requested_loss_metric not in {None, "val_loss", "train_loss"}:
        raise ValueError("loss_metric must be val_loss or train_loss")

    conn = get_db()
    counts = conn.execute(
        "SELECT COUNT(*) AS metric_count, "
        "SUM(val_loss IS NOT NULL) AS val_count, "
        "SUM(train_loss IS NOT NULL) AS train_count "
        "FROM metrics WHERE experiment_id = ?",
        (exp_id,),
    ).fetchone()
    available_loss_metrics = []
    if counts["val_count"]:
        available_loss_metrics.append("val_loss")
    if counts["train_count"]:
        available_loss_metrics.append("train_loss")
    primary_loss_metric = (
        "val_loss" if counts["val_count"]
        else ("train_loss" if counts["train_count"] else None)
    )
    requested = (
        [requested_loss_metric] if requested_loss_metric
        else available_loss_metrics
    )
    loss_curves = {
        metric_name: _bounded_loss_curve(exp_id, metric_name, max_points)
        for metric_name in requested
        if metric_name in available_loss_metrics
    }

    final = get_final_metric(exp_id)
    latest = get_latest_metric(exp_id)
    primary_curve = loss_curves.get(primary_loss_metric)
    if primary_curve is None and primary_loss_metric:
        # A targeted alternate-series refresh still needs a stable run summary.
        primary_curve = _bounded_loss_curve(
            exp_id, primary_loss_metric, min(max_points, 16),
        )
    if final and final.loss is not None:
        summary_loss = final.loss
        summary_train_time = final.train_time_ms
    elif primary_curve:
        summary_loss = primary_curve[-1]["loss"]
        summary_train_time = primary_curve[-1]["train_time_ms"]
    else:
        summary_loss = latest.loss if latest else None
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
        "session_name": exp.session_name,
        "started_at": exp.started_at.isoformat() if exp.started_at else None,
        "finished_at": exp.finished_at.isoformat() if exp.finished_at else None,
        "current_step": latest.step if latest else None,
        "total_steps": latest.total_steps if latest else None,
        "final_val_loss": summary_loss,
        "final_loss": summary_loss,
        "loss_metric": primary_loss_metric,
        "requested_loss_metric": requested_loss_metric,
        "available_loss_metrics": available_loss_metrics,
        "final_train_time_ms": summary_train_time,
        "loss_curves": loss_curves,
        "metrics_count": counts["metric_count"],
        "curve_max_points": max(4, min(max_points, 1200)),
    }


@app.get("/api/experiment/{exp_id}/chart")
async def get_experiment_chart(
    exp_id: int,
    max_points: int = 1200,
    loss_metric: Optional[str] = None,
):
    """Get bounded chart/detail data for one run."""
    try:
        detail = _bounded_experiment_detail(exp_id, max_points, loss_metric)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if detail is None:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)
    return detail


@app.get("/api/experiment/{exp_id}")
async def get_experiment_detail(exp_id: int):
    """Get detailed data for a single experiment including loss curve."""
    exp = get_experiment(exp_id)
    if not exp:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)

    # Keep the loss series separate: a run may log train_loss every step and
    # val_loss only on eval steps. `loss_curve` remains the primary series for
    # backward compatibility, while `loss_curves` lets the dashboard switch
    # between the two without ever connecting unlike quantities.
    metrics = get_metrics(exp_id)
    loss_metric = get_loss_metrics([exp_id]).get(exp_id)
    loss_curves = {
        metric_name: [
            {
                "step": m.step,
                "loss": getattr(m, metric_name),
                "val_loss": m.val_loss,
                "train_loss": m.train_loss,
                "train_time_ms": m.train_time_ms,
                "step_avg_ms": m.step_avg_ms,
            }
            for m in metrics
            if getattr(m, metric_name) is not None
        ]
        for metric_name in ("val_loss", "train_loss")
    }
    available_loss_metrics = [
        metric_name for metric_name, curve in loss_curves.items() if curve
    ]
    loss_curve = loss_curves.get(loss_metric, [])

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
        "available_loss_metrics": available_loss_metrics,
        "final_train_time_ms": summary_train_time,
        # Full loss curve for plotting
        "loss_curve": loss_curve,
        "loss_curves": loss_curves,
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
