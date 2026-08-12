"""FastAPI dashboard application."""

import asyncio
import json
import time
import webbrowser
from pathlib import Path
from typing import Optional

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
from starlette.middleware.gzip import GZipMiddleware

from ..config import discover_tracks, Config
from ..queue import get_queue_state
from ..watcher import DASHBOARD_HOST, safe_json_load, get_queue_cache_file
from ..tracker import (
    get_experiment,
    get_running_experiments,
    get_db,
    get_crash_log as get_crash_log_content,
    get_dashboard_event_bounds,
    get_dashboard_events,
    append_dashboard_event,
    read_experiment_summaries,
)
from .experiment_query import (
    QueryValidationError,
    encode_ndjson,
    execute_experiment_query,
    validate_query_request,
)

app = FastAPI(title="nanorun Dashboard")
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

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


@app.post("/api/experiments/query")
async def query_experiments(request: Request):
    """Run named experiment reads and stream revision-aware NDJSON frames."""
    try:
        body = await request.json()
        normalized = validate_query_request(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    except QueryValidationError as exc:
        # Validation is complete before StreamingResponse is constructed, so a
        # malformed request never turns into a partial 200 response.
        return JSONResponse({"error": str(exc)}, status_code=422)

    frames = await asyncio.to_thread(execute_experiment_query, normalized)
    return StreamingResponse(
        encode_ndjson(frames),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


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
    running_summaries = read_experiment_summaries([exp.id for exp in running_exps])
    for exp in running_exps:
        summary = running_summaries.get(exp.id, {})
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
            "current_step": summary.get("current_step"),
            "total_steps": summary.get("total_steps"),
            "val_loss": summary.get("val_loss"),
            "train_loss": summary.get("train_loss"),
            "loss": summary.get("loss"),
            "loss_metric": summary.get("loss_metric"),
            "revision": summary.get("revision", exp.revision),
            "metrics_revision": summary.get("metrics_revision", exp.metrics_revision),
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
    """Return non-experiment state plus a race-closing replay cursor.

    The cursor is read first. A change committed later is therefore either
    already visible in the following reads or replayed after this cursor (and a
    harmless duplicate is possible), but it cannot fall into a connection gap.
    """
    oldest_event_id, last_event_id = get_dashboard_event_bounds()
    queue = _get_queue_status()
    sessions = await get_sessions()
    return {
        "queue": queue,
        "sessions": sessions["sessions"],
        "hub": sessions["hub"],
        "tracks": _get_tracks_state()["tracks"],
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
    return _get_tracks_state()


def _get_tracks_state() -> dict:
    """Return track metadata for the snapshot and the standalone resource."""
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
