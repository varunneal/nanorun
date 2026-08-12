"""Experiment tracking with SQLite storage."""

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple

from .config import Config


DASHBOARD_EVENT_MAX_ROWS = 10_000
DASHBOARD_EVENT_MAX_AGE_DAYS = 7


def get_db_path() -> Path:
    """Get path to experiments database."""
    config_dir = Config.get_config_dir()
    return config_dir / "experiments.db"


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and run migrations.

    Called once per connection (connections are now persistent per-thread, so
    this runs rarely). Every statement is idempotent — CREATE ... IF NOT EXISTS,
    ALTER only when a column is missing — so running it on an already-migrated
    DB is a cheap no-op.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            track TEXT,
            script TEXT NOT NULL,
            code_hash TEXT,
            parent_hash TEXT,
            git_commit TEXT,
            env_vars TEXT,
            gpus INTEGER DEFAULT 1,
            gpu_type TEXT DEFAULT 'H100',
            run_number INTEGER,
            tmux_window TEXT,
            remote_run_id TEXT,
            status TEXT DEFAULT 'running',
            revision INTEGER NOT NULL DEFAULT 0,
            metrics_revision INTEGER NOT NULL DEFAULT 0,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL REFERENCES experiments(id),
            step INTEGER NOT NULL,
            total_steps INTEGER,
            val_loss REAL,
            train_loss REAL,
            train_time_ms INTEGER,
            step_avg_ms REAL,
            is_final_step BOOLEAN DEFAULT 0,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(experiment_id, step)
        );

        -- crash_log lives in its own table so the experiments table stays small
        -- (crash blobs are ~30KB each and were bloating every list query).
        CREATE TABLE IF NOT EXISTS crash_logs (
            experiment_id INTEGER PRIMARY KEY REFERENCES experiments(id),
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dashboard_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            entity_id TEXT,
            payload TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON metrics(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
        CREATE INDEX IF NOT EXISTS idx_experiments_track ON experiments(track);
        CREATE INDEX IF NOT EXISTS idx_dashboard_events_created_at
            ON dashboard_events(created_at);
        CREATE INDEX IF NOT EXISTS idx_dashboard_events_entity
            ON dashboard_events(event_type, entity_id, id);
    """)
    conn.commit()

    # Migration: add new columns if they don't exist (for existing DBs).
    # NOTE: crash_log is intentionally NOT here — it's been extracted to the
    # crash_logs table (see below); re-adding it would just recreate the bloat.
    cursor = conn.execute("PRAGMA table_info(experiments)")
    columns = [row[1] for row in cursor.fetchall()]
    migrations = {
        "code_hash": "TEXT",
        "parent_hash": "TEXT",
        "remote_run_id": "TEXT",
        "deleted": "INTEGER DEFAULT 0",
        "gpu_type": "TEXT DEFAULT 'H100'",
        "kernels_path": "TEXT",
        "dependencies": "TEXT",
        "session_name": "TEXT",
        "session_id": "TEXT",
        "queue_command": "TEXT",
        "revision": "INTEGER NOT NULL DEFAULT 0",
        "metrics_revision": "INTEGER NOT NULL DEFAULT 0",
    }
    metric_migrations = {
        "train_loss": "REAL",
    }
    for col, col_type in migrations.items():
        if col not in columns:
            conn.execute(f"ALTER TABLE experiments ADD COLUMN {col} {col_type}")

    cursor = conn.execute("PRAGMA table_info(metrics)")
    metric_columns = [row[1] for row in cursor.fetchall()]
    for col, col_type in metric_migrations.items():
        if col not in metric_columns:
            conn.execute(f"ALTER TABLE metrics ADD COLUMN {col} {col_type}")

    # Indexes on migration-added columns go AFTER the ALTER loop above (the column
    # must exist first — putting these in the CREATE-TABLE executescript would fail
    # on an existing DB whose column hasn't been added yet). idx_experiments_run_id
    # backs the delivery-driven parse's run_id (log stem) -> experiment point lookup;
    # a plain index is perf-identical to UNIQUE for that lookup and safe on the prod
    # DB. idx_experiments_session_id backs incarnation-scoped reconciliation. Both
    # are idempotent (IF NOT EXISTS).
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_run_id ON experiments(remote_run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_session_id ON experiments(session_id)")
    # Dashboard query indexes.  The partial metric indexes make the correlated
    # latest-row lookups seek directly to the selected point instead of scanning
    # a run's history.  Existing databases start with zero revisions; no metric
    # history scan or backfill is needed.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_experiments_visible_order "
        "ON experiments(COALESCE(deleted, 0), started_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_experiments_group_order "
        "ON experiments(code_hash, track, gpus, gpu_type, deleted, started_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_experiment_step "
        "ON metrics(experiment_id, step DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_latest_val "
        "ON metrics(experiment_id, step DESC) WHERE val_loss IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_final_step "
        "ON metrics(experiment_id, step DESC) WHERE is_final_step = 1"
    )
    conn.commit()

    # NOTE: moving the legacy inlined crash_log data into crash_logs and dropping
    # the column is a HEAVY, one-time operation (full-table rewrite + VACUUM on a
    # ~250MB DB) that takes an exclusive write lock. We deliberately do NOT run it
    # here — that would fire opportunistically on the next connection and could
    # collide with live metric writes. Run it explicitly in a quiet window via
    # migrate_crash_logs_out() (see scripts/migrate_crash_logs.py). Until then,
    # get_crash_log() falls back to the legacy column so old logs stay visible.


# Persistent per-thread connections. Each thread (session tracker, hub syncer,
# dashboard worker, CLI process) keeps one long-lived connection rather than
# opening/closing one per call — this eliminates the WAL cycling + fsync churn
# that per-call connections caused (dozens of open/close per metric line).
_local = threading.local()


def get_db() -> sqlite3.Connection:
    """Get the calling thread's persistent database connection.

    The connection is cached in thread-local storage and reused for the life of
    the thread. WAL mode + one connection per thread is SQLite's recommended
    concurrency pattern (many readers, one writer, serialized by busy_timeout).
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.Error:
            # Connection was closed (e.g. by close_db, or a caller's conn.close());
            # fall through and recreate.
            try:
                conn.close()
            except Exception:
                pass
            _local.conn = None

    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=1000")
    _init_schema(conn)
    _local.conn = conn
    return conn


def close_db() -> None:
    """Close and drop the calling thread's cached connection.

    Called from watcher/thread shutdown paths for a clean final WAL checkpoint,
    and from tests to reset connection state between temp databases.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None


@dataclass
class Experiment:
    """An experiment record."""
    id: int
    name: str
    track: Optional[str]
    script: str
    code_hash: Optional[str]
    parent_hash: Optional[str]
    git_commit: Optional[str]
    env_vars: Dict[str, str]
    gpus: int
    gpu_type: str
    run_number: Optional[int]
    tmux_window: Optional[str]
    remote_run_id: Optional[str]
    status: str
    crash_log: Optional[str]
    started_at: datetime
    finished_at: Optional[datetime]
    kernels_path: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)
    session_name: Optional[str] = None
    session_id: Optional[str] = None  # {name}::{started_at} — per-incarnation scope key
    revision: int = 0
    metrics_revision: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Experiment":
        keys = row.keys()
        return cls(
            id=row["id"],
            name=row["name"],
            track=row["track"],
            script=row["script"],
            code_hash=row["code_hash"] if "code_hash" in keys else None,
            parent_hash=row["parent_hash"] if "parent_hash" in keys else None,
            git_commit=row["git_commit"],
            env_vars=json.loads(row["env_vars"]) if row["env_vars"] else {},
            gpus=row["gpus"],
            gpu_type=row["gpu_type"] if "gpu_type" in keys and row["gpu_type"] else "H100",
            run_number=row["run_number"],
            tmux_window=row["tmux_window"],
            remote_run_id=row["remote_run_id"] if "remote_run_id" in keys else None,
            status=row["status"],
            # crash_log now lives in the crash_logs table; fetch on demand via
            # get_crash_log(). Kept on the dataclass for backwards compatibility,
            # always None here so list queries never load crash blobs.
            crash_log=None,
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            kernels_path=row["kernels_path"] if "kernels_path" in keys else None,
            dependencies=(
                json.loads(row["dependencies"])
                if "dependencies" in keys and row["dependencies"]
                else {}
            ),
            session_name=row["session_name"] if "session_name" in keys else None,
            session_id=row["session_id"] if "session_id" in keys else None,
            revision=row["revision"] if "revision" in keys else 0,
            metrics_revision=(
                row["metrics_revision"] if "metrics_revision" in keys else 0
            ),
        )


@dataclass
class Metric:
    """A single metric checkpoint.

    ``val_loss`` and ``train_loss`` are independent series: a script may emit
    either, both, or (at a shared step) have them merged into one row by the
    upsert in ``record_metric``. Use ``loss``/``loss_metric`` when you want
    "whichever series this run is actually reporting".
    """
    step: int
    total_steps: Optional[int]
    val_loss: Optional[float]
    train_loss: Optional[float]
    train_time_ms: Optional[int]
    step_avg_ms: Optional[float]
    is_final_step: bool
    recorded_at: datetime

    @property
    def loss(self) -> Optional[float]:
        """Primary loss for this row — validation wins when both are present."""
        return self.val_loss if self.val_loss is not None else self.train_loss

    @property
    def loss_metric(self) -> Optional[str]:
        """Which series ``loss`` came from, or None if the row has neither."""
        if self.val_loss is not None:
            return "val_loss"
        return "train_loss" if self.train_loss is not None else None


# =============================================================================
# Database operations
# =============================================================================

def _dashboard_group(exp: sqlite3.Row) -> Dict[str, Any]:
    """Return the structured identity used by the dashboard's group cards."""
    gpu_type = exp["gpu_type"] or "H100"
    return {
        "code_hash": exp["code_hash"],
        "track": exp["track"] or "",
        "gpus": exp["gpus"],
        "gpu_type": gpu_type,
    }


def read_experiment_summaries(
    experiment_ids: List[int],
    *,
    conn: Optional[sqlite3.Connection] = None,
    include_deleted: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Read canonical dashboard summaries for any number of experiments.

    This is the one loss/progress definition used by query responses, queue
    state, and durable events.  It is deliberately one SQL statement: every
    selected metric row is found through an indexed ``(experiment_id, step)``
    lookup, so adding runs does not add statements or introduce a whole-metrics
    ``GROUP BY/MAX`` scan.
    """
    ids = list(dict.fromkeys(
        int(experiment_id) for experiment_id in experiment_ids
        if isinstance(experiment_id, int) and not isinstance(experiment_id, bool)
    ))
    if not ids:
        return {}
    db = conn or get_db()
    placeholders = ",".join("?" for _ in ids)
    deleted_clause = "" if include_deleted else "AND COALESCE(e.deleted, 0) = 0"
    rows = db.execute(
        f"""
        SELECT e.*,
               progress.step AS progress_step,
               progress.total_steps AS progress_total_steps,
               latest_val.step AS latest_val_step,
               latest_val.val_loss AS latest_val_loss,
               latest_val.train_time_ms AS latest_val_time,
               latest_val.step_avg_ms AS latest_val_step_avg,
               latest_train.step AS latest_train_step,
               latest_train.train_loss AS latest_train_loss,
               latest_train.train_time_ms AS latest_train_time,
               latest_train.step_avg_ms AS latest_train_step_avg,
               final_val.step AS final_val_step,
               final_val.val_loss AS final_val_loss,
               final_val.train_time_ms AS final_val_time,
               final_train.step AS final_train_step,
               final_train.train_loss AS final_train_loss,
               final_train.train_time_ms AS final_train_time
        FROM experiments e
        LEFT JOIN metrics progress ON progress.id = (
            SELECT p.id FROM metrics p
            WHERE p.experiment_id = e.id
            ORDER BY p.step DESC LIMIT 1
        )
        LEFT JOIN metrics latest_val ON latest_val.id = (
            SELECT v.id FROM metrics v
            WHERE v.experiment_id = e.id AND v.val_loss IS NOT NULL
            ORDER BY v.step DESC LIMIT 1
        )
        LEFT JOIN metrics latest_train ON latest_train.id = (
            SELECT t.id FROM metrics t
            WHERE t.experiment_id = e.id AND t.train_loss IS NOT NULL
            ORDER BY t.step DESC LIMIT 1
        )
        LEFT JOIN metrics final_val ON final_val.id = (
            SELECT fv.id FROM metrics fv
            WHERE fv.experiment_id = e.id AND fv.is_final_step = 1
              AND fv.val_loss IS NOT NULL
            ORDER BY fv.step DESC LIMIT 1
        )
        LEFT JOIN metrics final_train ON final_train.id = (
            SELECT ft.id FROM metrics ft
            WHERE ft.experiment_id = e.id AND ft.is_final_step = 1
              AND ft.train_loss IS NOT NULL
            ORDER BY ft.step DESC LIMIT 1
        )
        WHERE e.id IN ({placeholders}) {deleted_clause}
        """,
        ids,
    ).fetchall()

    result: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        has_val = row["latest_val_step"] is not None
        has_train = row["latest_train_step"] is not None
        primary = "val_loss" if has_val else ("train_loss" if has_train else None)
        if primary == "val_loss":
            latest_loss = row["latest_val_loss"]
            latest_loss_step = row["latest_val_step"]
            latest_time = row["latest_val_time"]
            latest_step_avg = row["latest_val_step_avg"]
            final_loss = row["final_val_loss"]
            final_loss_step = row["final_val_step"]
            final_time = row["final_val_time"]
        elif primary == "train_loss":
            latest_loss = row["latest_train_loss"]
            latest_loss_step = row["latest_train_step"]
            latest_time = row["latest_train_time"]
            latest_step_avg = row["latest_train_step_avg"]
            final_loss = row["final_train_loss"]
            final_loss_step = row["final_train_step"]
            final_time = row["final_train_time"]
        else:
            latest_loss = latest_loss_step = latest_time = latest_step_avg = None
            final_loss = final_loss_step = final_time = None

        available = []
        if has_val:
            available.append("val_loss")
        if has_train:
            available.append("train_loss")
        summary = {
            "id": row["id"],
            "name": row["name"],
            "track": row["track"],
            "script": row["script"],
            "code_hash": row["code_hash"],
            "parent_hash": row["parent_hash"],
            "git_commit": row["git_commit"],
            "status": row["status"],
            "gpus": row["gpus"],
            "gpu_type": row["gpu_type"] or "H100",
            "env_vars": json.loads(row["env_vars"]) if row["env_vars"] else {},
            "run_number": row["run_number"],
            "tmux_window": row["tmux_window"],
            "remote_run_id": row["remote_run_id"],
            "kernels_path": row["kernels_path"],
            "dependencies": (
                json.loads(row["dependencies"]) if row["dependencies"] else {}
            ),
            "session_name": row["session_name"],
            "session_id": row["session_id"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "revision": int(row["revision"] or 0),
            "metrics_revision": int(row["metrics_revision"] or 0),
            "current_step": row["progress_step"],
            "total_steps": row["progress_total_steps"],
            "val_loss": row["latest_val_loss"],
            "train_loss": row["latest_train_loss"],
            "latest_loss": latest_loss,
            "latest_loss_step": latest_loss_step,
            "loss": latest_loss,
            "loss_metric": primary,
            "available_loss_metrics": available,
            "train_time_ms": latest_time,
            "step_avg_ms": latest_step_avg,
            "final_loss": final_loss,
            "final_loss_step": final_loss_step,
            # Existing rendering uses this as the explicit latest fallback.
            "final_val_loss": latest_loss,
            "final_train_time_ms": final_time if final_loss is not None else latest_time,
            "group": _dashboard_group(row),
        }
        result[row["id"]] = summary
    return result


def _dashboard_experiment_summary(
    conn: sqlite3.Connection,
    experiment_id: int,
) -> Optional[Dict[str, Any]]:
    """Build the canonical bounded experiment patch carried by events."""
    return read_experiment_summaries([experiment_id], conn=conn).get(experiment_id)


def _compact_dashboard_events(conn: sqlite3.Connection) -> None:
    """Keep replay storage bounded without touching projection data."""
    conn.execute(
        "DELETE FROM dashboard_events WHERE id < COALESCE(("
        "SELECT id FROM dashboard_events ORDER BY id DESC LIMIT 1 OFFSET ?"
        "), 0)",
        (DASHBOARD_EVENT_MAX_ROWS - 1,),
    )
    conn.execute(
        "DELETE FROM dashboard_events "
        "WHERE created_at < datetime('now', ?)",
        (f"-{DASHBOARD_EVENT_MAX_AGE_DAYS} days",),
    )


def _append_dashboard_event(
    conn: sqlite3.Connection,
    event_type: str,
    entity_id: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
) -> int:
    """Append one event inside the caller's current SQLite transaction."""
    event_payload = dict(payload or {})
    event_payload.setdefault("committed_at", datetime.now(timezone.utc).isoformat())
    cursor = conn.execute(
        "INSERT INTO dashboard_events (event_type, entity_id, payload) "
        "VALUES (?, ?, ?)",
        (event_type, str(entity_id) if entity_id is not None else None,
         json.dumps(event_payload, separators=(",", ":"), sort_keys=True)),
    )
    event_id = cursor.lastrowid
    # The replay cursor orders the event log; experiment revision counters order
    # entity state.  Keeping them separate prevents an older query response from
    # winning merely because its HTTP request completed after a newer SSE event.
    event_payload["event_cursor"] = event_id
    conn.execute(
        "UPDATE dashboard_events SET payload = ? WHERE id = ?",
        (json.dumps(event_payload, separators=(",", ":"), sort_keys=True), event_id),
    )
    _compact_dashboard_events(conn)
    return event_id


def append_dashboard_event(
    event_type: str,
    entity_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> int:
    """Append and commit an event for a projection stored outside SQLite."""
    conn = get_db()
    event_id = _append_dashboard_event(conn, event_type, entity_id, payload)
    conn.commit()
    return event_id


def get_dashboard_event_bounds() -> Tuple[int, int]:
    """Return the oldest and newest retained event IDs (zero when empty)."""
    row = get_db().execute(
        "SELECT COALESCE(MIN(id), 0) AS oldest, "
        "COALESCE(MAX(id), 0) AS newest FROM dashboard_events"
    ).fetchone()
    return row["oldest"], row["newest"]


def get_dashboard_events(after: int, limit: int = 100) -> List[Dict[str, Any]]:
    """Read one bounded, ordered replay page."""
    bounded_limit = max(1, min(limit, 500))
    rows = get_db().execute(
        "SELECT id, event_type, entity_id, payload, created_at "
        "FROM dashboard_events WHERE id > ? ORDER BY id LIMIT ?",
        (after, bounded_limit),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "event_type": row["event_type"],
            "entity_id": row["entity_id"],
            "payload": json.loads(row["payload"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_dashboard_experiment_summary(experiment_id: int) -> Optional[Dict[str, Any]]:
    """Return a current event-shaped summary for one experiment."""
    return _dashboard_experiment_summary(get_db(), experiment_id)


def commit_metric_batch(experiment_ids: List[int]) -> None:
    """Commit a parse batch and emit at most one metric event per experiment."""
    conn = get_db()
    ids = list(dict.fromkeys(experiment_ids))
    if ids:
        placeholders = ",".join("?" for _ in ids)
        conn.execute(
            f"UPDATE experiments SET revision = revision + 1, "
            f"metrics_revision = metrics_revision + 1 "
            f"WHERE id IN ({placeholders})",
            ids,
        )
    summaries = read_experiment_summaries(ids, conn=conn)
    for experiment_id in ids:
        summary = summaries.get(experiment_id)
        if summary:
            _append_dashboard_event(
                conn,
                "metrics.changed",
                str(experiment_id),
                {
                    "experiment_id": experiment_id,
                    "group": summary["group"],
                    "latest_step": summary["current_step"],
                    "revision": summary["revision"],
                    "metrics_revision": summary["metrics_revision"],
                    "summary": summary,
                },
            )
    conn.commit()

def _build_queue_command(script: str, env_vars: Optional[Dict[str, str]]) -> str:
    """Build a reproducible nanorun command from script + env_vars."""
    parts = ["nanorun job add", script]
    if env_vars:
        for k, v in env_vars.items():
            if not k.startswith("_"):
                parts.append(f"--env {k}={v}")
    return " ".join(parts)


def create_experiment(
    name: str,
    script: str,
    track: Optional[str] = None,
    code_hash: Optional[str] = None,
    parent_hash: Optional[str] = None,
    git_commit: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    gpus: int = 1,
    gpu_type: str = "H100",
    run_number: Optional[int] = None,
    tmux_window: Optional[str] = None,
    dependencies: Optional[Dict[str, str]] = None,
    session_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> int:
    """Create a new experiment record. Returns the experiment ID."""
    conn = get_db()
    cursor = conn.execute(
        """
        INSERT INTO experiments (name, track, script, code_hash, parent_hash, git_commit, env_vars, gpus, gpu_type, run_number, tmux_window, dependencies, session_name, session_id, queue_command)
        VALUES (:name, :track, :script, :code_hash, :parent_hash, :git_commit, :env_vars, :gpus, :gpu_type, :run_number, :tmux_window, :dependencies, :session_name, :session_id, :queue_command)
        """,
        {
            "name": name,
            "track": track,
            "script": script,
            "code_hash": code_hash,
            "parent_hash": parent_hash,
            "git_commit": git_commit,
            "env_vars": json.dumps(env_vars or {}),
            "gpus": gpus,
            "gpu_type": gpu_type,
            "run_number": run_number,
            "tmux_window": tmux_window,
            "dependencies": json.dumps(dependencies or {}, sort_keys=True),
            "session_name": session_name,
            "session_id": session_id,
            "queue_command": _build_queue_command(script, env_vars),
        }
    )
    exp_id = cursor.lastrowid
    summary = _dashboard_experiment_summary(conn, exp_id)
    _append_dashboard_event(
        conn, "experiment.created", str(exp_id),
        {"experiment_id": exp_id, "group": summary["group"],
         "revision": summary["revision"],
         "metrics_revision": summary["metrics_revision"], "summary": summary},
    )
    conn.commit()
    return exp_id


def record_metric(
    experiment_id: int,
    step: int,
    total_steps: Optional[int] = None,
    val_loss: Optional[float] = None,
    train_loss: Optional[float] = None,
    train_time_ms: Optional[int] = None,
    step_avg_ms: Optional[float] = None,
    is_final_step: bool = False,
    commit: bool = True,
) -> bool:
    """Record a metric checkpoint. Returns True only if new data was written.

    Pass ``commit=False`` to batch many rows in one transaction — the caller then
    commits once at the end of the pass, turning N fsyncs into 1. ``cursor.rowcount``
    reports whether this row actually inserted/updated (0 on a no-op conflict;
    verified for this exact upsert).
    """
    conn = get_db()
    cursor = conn.execute(
        """
        INSERT INTO metrics
            (experiment_id, step, total_steps, val_loss, train_loss, train_time_ms, step_avg_ms, is_final_step)
        VALUES
            (:experiment_id, :step, :total_steps, :val_loss, :train_loss, :train_time_ms, :step_avg_ms, :is_final_step)
        ON CONFLICT(experiment_id, step) DO UPDATE SET
            total_steps = COALESCE(:total_steps, total_steps),
            val_loss = COALESCE(:val_loss, val_loss),
            train_loss = COALESCE(:train_loss, train_loss),
            train_time_ms = COALESCE(:train_time_ms, train_time_ms),
            step_avg_ms = COALESCE(:step_avg_ms, step_avg_ms),
            is_final_step = MAX(is_final_step, :is_final_step)
        WHERE total_steps IS NOT COALESCE(:total_steps, total_steps)
           OR val_loss IS NOT COALESCE(:val_loss, val_loss)
           OR train_loss IS NOT COALESCE(:train_loss, train_loss)
           OR train_time_ms IS NOT COALESCE(:train_time_ms, train_time_ms)
           OR step_avg_ms IS NOT COALESCE(:step_avg_ms, step_avg_ms)
           OR is_final_step < :is_final_step
        """,
        {
            "experiment_id": experiment_id,
            "step": step,
            "total_steps": total_steps,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "train_time_ms": train_time_ms,
            "step_avg_ms": step_avg_ms,
            "is_final_step": is_final_step,
        }
    )
    changed = cursor.rowcount > 0
    if commit:
        if changed:
            conn.execute(
                "UPDATE experiments SET revision = revision + 1, "
                "metrics_revision = metrics_revision + 1 WHERE id = ?",
                (experiment_id,),
            )
            summary = _dashboard_experiment_summary(conn, experiment_id)
            if summary:
                _append_dashboard_event(
                    conn,
                    "metrics.changed",
                    str(experiment_id),
                    {
                        "experiment_id": experiment_id,
                        "group": summary["group"],
                        "latest_step": summary["current_step"],
                        "revision": summary["revision"],
                        "metrics_revision": summary["metrics_revision"],
                        "summary": summary,
                    },
                )
        conn.commit()
    return changed


def update_experiment_status(experiment_id: int, status: str) -> bool:
    """Update experiment status and keep terminal timestamps consistent."""
    conn = get_db()
    if status in ("completed", "failed", "cancelled"):
        cursor = conn.execute(
            "UPDATE experiments SET status = ?, finished_at = ?, "
            "revision = revision + 1 "
            "WHERE id = ? AND (status IS NOT ? OR finished_at IS NULL)",
            (status, datetime.now(timezone.utc).isoformat(), experiment_id, status)
        )
    elif status == "running":
        # A direct remote observation may correct a stale terminal projection.
        cursor = conn.execute(
            "UPDATE experiments SET status = ?, started_at = COALESCE(started_at, ?), "
            "finished_at = NULL, revision = revision + 1 WHERE id = ? "
            "AND (status IS NOT ? OR finished_at IS NOT NULL)",
            (status, datetime.now(timezone.utc).isoformat(), experiment_id, status)
        )
    else:
        cursor = conn.execute(
            "UPDATE experiments SET status = ?, finished_at = NULL, "
            "revision = revision + 1 WHERE id = ? "
            "AND (status IS NOT ? OR finished_at IS NOT NULL)",
            (status, experiment_id, status)
        )
    changed = cursor.rowcount > 0
    if changed:
        summary = _dashboard_experiment_summary(conn, experiment_id)
        if summary:
            _append_dashboard_event(
                conn, "experiment.updated", str(experiment_id),
                {"experiment_id": experiment_id, "group": summary["group"],
                 "revision": summary["revision"],
                 "metrics_revision": summary["metrics_revision"], "summary": summary},
            )
    conn.commit()
    return changed


TERMINAL_EXPERIMENT_STATUSES = frozenset({"completed", "failed", "cancelled"})
OBSERVED_EXPERIMENT_STATUSES = TERMINAL_EXPERIMENT_STATUSES | {
    "queued", "running", "unknown",
}


def apply_authoritative_experiment_status(
    experiment_id: int,
    observed_status: str,
) -> bool:
    """Apply execution evidence to the local projection.

    A final metric is direct proof that the training contract completed, so a
    later exit-classification or stale mapping cannot downgrade it. Other local
    projections may be corrected by remote active state, terminal mappings,
    terminal events, or acknowledged control operations.
    """
    if observed_status not in OBSERVED_EXPERIMENT_STATUSES:
        return False
    experiment = get_experiment(experiment_id)
    if not experiment or experiment.status == observed_status:
        return False
    if (
        experiment.status == "completed"
        and observed_status != "completed"
        and get_final_metric(experiment_id) is not None
    ):
        return False
    update_experiment_status(experiment_id, observed_status)
    return True


def terminate_session_experiments(
    session_name: str,
    running_status: str = "cancelled",
    queued_status: str = "cancelled",
    note: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[List[int], List[int]]:
    """Move a session's in-flight experiments to a terminal state.

    This is an explicit administrative operation, not a connectivity-recovery
    primitive. Callers must already have authoritative evidence or direct user
    intent for the requested terminal transitions.

    Running rows go to ``running_status``, queued rows to ``queued_status``. When
    ``note`` is given it's stamped as the crash log on each running row that
    doesn't already have one, so the reason survives in the UI. Returns
    ``(running_ids, queued_ids)`` — the experiments that were transitioned — for
    the caller to log or emit events on.

    When ``session_id`` is given, scoping pins to that incarnation (with a NULL
    fallback so pre-upgrade rows still match by name) — this prevents killing a
    *different* machine incarnation that happened to reuse ``session_name``. When
    ``session_id`` is None, falls back to matching by name alone (legacy behavior).
    """
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Transitional scope clause: exact-incarnation match, OR (pre-upgrade rows with
    # NULL session_id) match by name. Bare-name fallback when caller has no id.
    if session_id is not None:
        scope_sql = "(session_id = ? OR (session_id IS NULL AND session_name = ?))"
        scope_params = (session_id, session_name)
    else:
        scope_sql = "session_name = ?"
        scope_params = (session_name,)

    running_ids = [
        row["id"] for row in conn.execute(
            f"SELECT id FROM experiments WHERE status = 'running' AND {scope_sql}",
            scope_params,
        ).fetchall()
    ]
    queued_ids = [
        row["id"] for row in conn.execute(
            f"SELECT id FROM experiments WHERE status = 'queued' AND {scope_sql}",
            scope_params,
        ).fetchall()
    ]

    if note:
        for exp_id in running_ids:
            if not get_crash_log(exp_id):
                set_crash_log(exp_id, note)

    if running_ids:
        conn.execute(
            f"UPDATE experiments SET status = ?, finished_at = COALESCE(finished_at, ?), "
            f"revision = revision + 1 "
            f"WHERE status = 'running' AND {scope_sql}",
            (running_status, now, *scope_params),
        )
    if queued_ids:
        conn.execute(
            f"UPDATE experiments SET status = ?, finished_at = COALESCE(finished_at, ?), "
            f"revision = revision + 1 "
            f"WHERE status = 'queued' AND {scope_sql}",
            (queued_status, now, *scope_params),
        )
    changed_ids = running_ids + queued_ids
    summaries = read_experiment_summaries(changed_ids, conn=conn)
    for exp_id in changed_ids:
        summary = summaries.get(exp_id)
        if summary:
            _append_dashboard_event(
                conn, "experiment.updated", str(exp_id),
                {"experiment_id": exp_id, "group": summary["group"],
                 "revision": summary["revision"],
                 "metrics_revision": summary["metrics_revision"], "summary": summary},
            )
    conn.commit()
    return running_ids, queued_ids


def update_experiment_metadata(
    experiment_id: int,
    code_hash: Optional[str] = None,
    tmux_window: Optional[str] = None,
    remote_run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    git_commit: Optional[str] = None,
    parent_hash: Optional[str] = None,
    kernels_path: Optional[str] = None,
    dependencies: Optional[Dict[str, str]] = None,
    session_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Update experiment metadata.

    Used to update fields that are set by the daemon after experiment creation.
    (crash_log is stored separately — use set_crash_log().)
    """
    conn = get_db()
    updates = []
    params = []

    if code_hash is not None:
        updates.append("code_hash = ?")
        params.append(code_hash)
    if tmux_window is not None:
        updates.append("tmux_window = ?")
        params.append(tmux_window)
    if remote_run_id is not None:
        updates.append("remote_run_id = ?")
        params.append(remote_run_id)
    if started_at is not None:
        updates.append("started_at = ?")
        params.append(started_at)
    if finished_at is not None:
        updates.append("finished_at = ?")
        params.append(finished_at)
    if git_commit is not None:
        updates.append("git_commit = ?")
        params.append(git_commit)
    if parent_hash is not None:
        updates.append("parent_hash = ?")
        params.append(parent_hash)
    if kernels_path is not None:
        updates.append("kernels_path = ?")
        params.append(kernels_path)
    if dependencies is not None:
        updates.append("dependencies = ?")
        params.append(json.dumps(dependencies, sort_keys=True))
    if session_name is not None:
        updates.append("session_name = ?")
        params.append(session_name)
    if session_id is not None:
        updates.append("session_id = ?")
        params.append(session_id)

    if updates:
        before = _dashboard_experiment_summary(conn, experiment_id)
        params.append(experiment_id)
        query = f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?"
        conn.execute(query, params)
        after = _dashboard_experiment_summary(conn, experiment_id)
        if after and after != before:
            conn.execute(
                "UPDATE experiments SET revision = revision + 1 WHERE id = ?",
                (experiment_id,),
            )
            after = _dashboard_experiment_summary(conn, experiment_id)
            _append_dashboard_event(
                conn, "experiment.updated", str(experiment_id),
                {"experiment_id": experiment_id, "group": after["group"],
                 "revision": after["revision"],
                 "metrics_revision": after["metrics_revision"], "summary": after},
            )
        conn.commit()


def set_crash_log(experiment_id: int, content: str) -> None:
    """Store (or replace) the crash/output log for an experiment."""
    if content is None:
        return
    conn = get_db()
    conn.execute(
        "INSERT INTO crash_logs (experiment_id, content) VALUES (?, ?) "
        "ON CONFLICT(experiment_id) DO UPDATE SET content = excluded.content",
        (experiment_id, content),
    )
    conn.commit()


def get_crash_log(experiment_id: int) -> Optional[str]:
    """Fetch the crash/output log for an experiment, or None if not stored.

    Reads the crash_logs table first. For DBs where the one-time extraction
    (migrate_crash_logs_out) hasn't run yet, falls back to the legacy inlined
    experiments.crash_log column so historical logs remain visible. The fallback
    self-heals once the column is dropped (the SELECT raises and we return None).
    """
    conn = get_db()
    row = conn.execute(
        "SELECT content FROM crash_logs WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    if row:
        return row["content"]
    try:
        legacy = conn.execute(
            "SELECT crash_log FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None  # column already dropped by migrate_crash_logs_out()
    return legacy["crash_log"] if legacy and legacy["crash_log"] else None


def migrate_crash_logs_out(vacuum: bool = True) -> Dict[str, object]:
    """Move legacy inlined crash_log blobs into crash_logs and drop the column.

    HEAVY one-time operation: rewrites the whole experiments table under an
    exclusive write lock, then (optionally) VACUUMs to reclaim file space. Run
    it only when the watcher is stopped / no experiments are writing. Idempotent
    and crash-safe: the backfill is INSERT OR IGNORE (never loses data) and the
    DROP is transactional (an interrupt rolls back cleanly), so re-running after
    an interruption finishes the job.

    Returns a stats dict: {done, migrated, dropped, already_migrated}.

    If the column is already gone (e.g. a prior migration ran without VACUUM),
    VACUUM still runs when requested so a bloated file gets reclaimed.
    """
    conn = get_db()
    columns = [row[1] for row in conn.execute("PRAGMA table_info(experiments)")]
    already_migrated = "crash_log" not in columns

    migrated = 0
    dropped = False
    if not already_migrated:
        cur = conn.execute(
            "INSERT OR IGNORE INTO crash_logs (experiment_id, content) "
            "SELECT id, crash_log FROM experiments WHERE crash_log IS NOT NULL"
        )
        migrated = cur.rowcount
        conn.commit()

        dropped = True
        try:
            conn.execute("ALTER TABLE experiments DROP COLUMN crash_log")
        except sqlite3.OperationalError:
            # SQLite < 3.35 has no DROP COLUMN — free the blobs in place instead so
            # a VACUUM can still reclaim the space (column stays but empty).
            conn.execute("UPDATE experiments SET crash_log = NULL WHERE crash_log IS NOT NULL")
            dropped = False
        conn.commit()

    if vacuum:
        # VACUUM cannot run inside a transaction; commits above leave us idle.
        # Runs even when already_migrated so a prior no-VACUUM drop is reclaimed.
        conn.execute("VACUUM")
        conn.commit()

    return {
        "done": True,
        "already_migrated": already_migrated,
        "migrated": migrated,
        "dropped": dropped,
    }


def get_experiment(experiment_id: int) -> Optional[Experiment]:
    """Get an experiment by ID."""
    conn = get_db()
    row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    return Experiment.from_row(row) if row else None


def delete_experiment(experiment_id: int) -> None:
    """Soft delete an experiment (sets deleted flag)."""
    conn = get_db()
    cursor = conn.execute(
        "UPDATE experiments SET deleted = 1, revision = revision + 1 "
        "WHERE id = ? AND COALESCE(deleted, 0) = 0",
        (experiment_id,),
    )
    if cursor.rowcount:
        deleted_summary = read_experiment_summaries(
            [experiment_id], conn=conn, include_deleted=True,
        ).get(experiment_id)
        _append_dashboard_event(
            conn, "experiment.deleted", str(experiment_id),
            {"experiment_id": experiment_id,
             "revision": deleted_summary["revision"] if deleted_summary else 0,
             "metrics_revision": (
                 deleted_summary["metrics_revision"] if deleted_summary else 0
             )},
        )
    conn.commit()


def get_running_experiments(
    session_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[Experiment]:
    """Get all running experiments, optionally filtered by session.

    Pass ``session_id`` to scope to a single incarnation (with a NULL fallback so
    pre-upgrade rows still match by name). Without it, filtering is by name alone.
    """
    conn = get_db()
    query = "SELECT * FROM experiments WHERE status = 'running' AND (deleted IS NULL OR deleted = 0)"
    params = []
    if session_id is not None:
        query += " AND (session_id = ? OR (session_id IS NULL AND session_name = ?))"
        params.extend([session_id, session_name])
    elif session_name:
        query += " AND session_name = ?"
        params.append(session_name)
    query += " ORDER BY started_at DESC"
    rows = conn.execute(query, params).fetchall()
    return [Experiment.from_row(row) for row in rows]


def get_all_experiment_ids() -> set:
    """Get set of all experiment IDs in local database."""
    conn = get_db()
    rows = conn.execute("SELECT id FROM experiments").fetchall()
    return {row["id"] for row in rows}


def create_experiment_from_mapping(
    experiment_id: int,
    name: str,
    script: str,
    status: str = "running",
    track: Optional[str] = None,
    code_hash: Optional[str] = None,
    remote_run_id: Optional[str] = None,
    tmux_window: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    gpus: int = 1,
    gpu_type: str = "H100",
    crash_log: Optional[str] = None,
    git_commit: Optional[str] = None,
    parent_hash: Optional[str] = None,
    kernels_path: Optional[str] = None,
    dependencies: Optional[Dict[str, str]] = None,
    session_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> int:
    """Idempotently create an experiment with an explicit remote-assigned ID.

    Unlike create_experiment(), this uses a specific ID rather than auto-increment.
    Used when the daemon first confirms a queued or running experiment.
    Replayed queue snapshots, mappings, and events are harmless.

    Returns the experiment ID.
    """
    conn = get_db()
    cursor = conn.execute(
        """
        INSERT INTO experiments (id, name, track, script, code_hash, parent_hash, git_commit, env_vars, gpus, gpu_type, tmux_window, remote_run_id, status, started_at, finished_at, kernels_path, dependencies, session_name, session_id, queue_command)
        VALUES (:id, :name, :track, :script, :code_hash, :parent_hash, :git_commit, :env_vars, :gpus, :gpu_type, :tmux_window, :remote_run_id, :status, :started_at, :finished_at, :kernels_path, :dependencies, :session_name, :session_id, :queue_command)
        ON CONFLICT(id) DO NOTHING
        """,
        {
            "id": experiment_id,
            "name": name,
            "track": track,
            "script": script,
            "code_hash": code_hash,
            "parent_hash": parent_hash,
            "git_commit": git_commit,
            "env_vars": json.dumps(env_vars or {}),
            "gpus": gpus,
            "gpu_type": gpu_type,
            "tmux_window": tmux_window,
            "remote_run_id": remote_run_id,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "kernels_path": kernels_path,
            "dependencies": json.dumps(dependencies or {}, sort_keys=True),
            "session_name": session_name,
            "session_id": session_id,
            "queue_command": _build_queue_command(script, env_vars),
        }
    )
    if cursor.rowcount:
        summary = _dashboard_experiment_summary(conn, experiment_id)
        _append_dashboard_event(
            conn, "experiment.created", str(experiment_id),
            {"experiment_id": experiment_id, "group": summary["group"],
             "revision": summary["revision"],
             "metrics_revision": summary["metrics_revision"], "summary": summary},
        )
    conn.commit()
    if crash_log is not None:
        set_crash_log(experiment_id, crash_log)
    return experiment_id


def get_experiments(
    track: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    session_name: Optional[str] = None,
) -> List[Experiment]:
    """Get experiments with optional filters (excludes soft-deleted).

    Args:
        track: Filter by track name
        status: Filter by status (running, completed, failed, etc.)
        search: Search in name, script, track, and code_hash fields
        limit: Maximum number of results to return
        session_name: Filter by session name
    """
    conn = get_db()
    query = "SELECT * FROM experiments WHERE (deleted IS NULL OR deleted = 0)"
    params = []

    if track:
        query += " AND track = ?"
        params.append(track)
    if status:
        query += " AND status = ?"
        params.append(status)
    if session_name:
        query += " AND session_name = ?"
        params.append(session_name)
    if search:
        # Search across name, script, track, and code_hash
        query += " AND (name LIKE ? OR script LIKE ? OR track LIKE ? OR code_hash LIKE ?)"
        search_pattern = f"%{search}%"
        params.extend([search_pattern, search_pattern, search_pattern, search_pattern])

    query += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [Experiment.from_row(row) for row in rows]


def get_metrics(experiment_id: int) -> List[Metric]:
    """Get all metrics for an experiment."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM metrics WHERE experiment_id = :id ORDER BY step",
        {"id": experiment_id}
    ).fetchall()
    return [
        Metric(
            step=row["step"],
            total_steps=row["total_steps"],
            val_loss=row["val_loss"],
            train_loss=row["train_loss"],
            train_time_ms=row["train_time_ms"],
            step_avg_ms=row["step_avg_ms"],
            is_final_step=bool(row["is_final_step"]),
            recorded_at=datetime.fromisoformat(row["recorded_at"]) if row["recorded_at"] else None,
        )
        for row in rows
    ]


def get_latest_metric(experiment_id: int) -> Optional[Metric]:
    """Get the most recent metric for an experiment."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM metrics WHERE experiment_id = :id ORDER BY step DESC LIMIT 1",
        {"id": experiment_id}
    ).fetchone()
    if not row:
        return None
    return Metric(
        step=row["step"],
        total_steps=row["total_steps"],
        val_loss=row["val_loss"],
        train_loss=row["train_loss"],
        train_time_ms=row["train_time_ms"],
        step_avg_ms=row["step_avg_ms"],
        is_final_step=bool(row["is_final_step"]),
        recorded_at=datetime.fromisoformat(row["recorded_at"]) if row["recorded_at"] else None,
    )


def get_final_metric(experiment_id: int) -> Optional[Metric]:
    """Get the final metric for an experiment (where is_final_step=True)."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM metrics WHERE experiment_id = :id AND is_final_step = 1 LIMIT 1",
        {"id": experiment_id}
    ).fetchone()
    if not row:
        return None
    return Metric(
        step=row["step"],
        total_steps=row["total_steps"],
        val_loss=row["val_loss"],
        train_loss=row["train_loss"],
        train_time_ms=row["train_time_ms"],
        step_avg_ms=row["step_avg_ms"],
        is_final_step=True,
        recorded_at=datetime.fromisoformat(row["recorded_at"]) if row["recorded_at"] else None,
    )


def get_loss_metrics(experiment_ids: List[int]) -> Dict[int, str]:
    """Infer each experiment's primary loss series from the rows it actually has.

    Returns {experiment_id: 'val_loss' | 'train_loss'}; experiments with no
    loss rows at all are absent from the mapping.

    Inference rather than declaration: a run that reports a validation loss is
    judged on it, and one that only ever reports training loss is judged on
    that. This needs no frontmatter field and works retroactively on runs that
    were recorded before train_loss was surfaced.
    """
    if not experiment_ids:
        return {}
    conn = get_db()
    placeholders = ",".join("?" for _ in experiment_ids)
    rows = conn.execute(
        f"""SELECT experiment_id,
                   MAX(val_loss IS NOT NULL) AS has_val,
                   MAX(train_loss IS NOT NULL) AS has_train
            FROM metrics
            WHERE experiment_id IN ({placeholders})
            GROUP BY experiment_id""",
        list(experiment_ids),
    ).fetchall()
    result = {}
    for row in rows:
        if row["has_val"]:
            result[row["experiment_id"]] = "val_loss"
        elif row["has_train"]:
            result[row["experiment_id"]] = "train_loss"
    return result


# =============================================================================
# Log parsing
# =============================================================================

# Metric line fields: step:N/M, val_loss:X, train_time:Y(ms|s), step_avg:Zms.
# Each field matched independently so unknown fields (e.g. epoch:3.5) can be
# interleaved without breaking parsing. Numbers require a leading digit so
# unfilled f-string placeholders like `val_loss:{final_val_loss:.4f}` don't
# hit the matcher (the `.4` in `:.4f}` would otherwise parse as 0.4).
# The exponent suffix is matched so that `val_loss:1e-3` reads as 0.001. Without
# it the pattern still matched the leading `1` and silently recorded 1.0 — a
# 1000x wrong loss with no error. Leading digit is still required, which is what
# keeps unfilled f-string placeholders (`val_loss:{loss:.4f}`) from matching.
_NUM = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_STEP_FIELD = re.compile(rf"step:(\d+)/(\d+)")
_VAL_LOSS_FIELD = re.compile(rf"val_loss:({_NUM})")
_TRAIN_LOSS_FIELD = re.compile(rf"train_loss:({_NUM})")
_TRAIN_TIME_FIELD = re.compile(rf"train_time:({_NUM})(ms|s)")
_STEP_AVG_FIELD = re.compile(rf"step_avg:({_NUM})ms")


def parse_metric_line(line: str) -> Optional[Dict]:
    """Parse a single log line for metrics. Returns dict or None.

    A line qualifies when it carries ``step:N/M`` plus at least one loss field;
    every other field is optional and matched independently, so these all work:

      - step:N/M val_loss:X train_time:Yms [step_avg:Zms]   (eval line)
      - step:N/M train_loss:X [train_time:Yms] [step_avg:Zms]
      - step:N/M val_loss:X train_loss:Y ...                (both on one line)
      - step:N/M val_loss:X                                 (no timing available)

    Timing is captured wherever it appears rather than only on val_loss lines —
    a train-loss-primary script puts train_time on its train_loss lines, and
    dropping it there left those runs with no wall-clock at all.

    Lines carrying timing but no loss (e.g. the record scripts' trailing
    ``step:N/M train_time:Xms step_avg:Yms``) are still ignored: without a loss
    there is nothing to record, and admitting them would let the source-code
    dump at the top of every log produce spurious rows.
    """
    step_match = _STEP_FIELD.search(line)
    if not step_match:
        return None

    val_loss_match = _VAL_LOSS_FIELD.search(line)
    train_loss_match = _TRAIN_LOSS_FIELD.search(line)
    if not (val_loss_match or train_loss_match):
        return None

    result = {
        "step": int(step_match.group(1)),
        "total_steps": int(step_match.group(2)),
        "val_loss": float(val_loss_match.group(1)) if val_loss_match else None,
        "train_loss": float(train_loss_match.group(1)) if train_loss_match else None,
        "train_time_ms": None,
        "step_avg_ms": None,
    }

    train_time_match = _TRAIN_TIME_FIELD.search(line)
    if train_time_match:
        train_time_val = float(train_time_match.group(1))
        result["train_time_ms"] = (
            int(train_time_val * 1000) if train_time_match.group(2) == "s" else int(train_time_val)
        )

    step_avg_match = _STEP_AVG_FIELD.search(line)
    if step_avg_match:
        result["step_avg_ms"] = float(step_avg_match.group(1))

    return result
