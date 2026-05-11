"""Experiment tracking with SQLite storage."""

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from rich.console import Console
from rich.table import Table

from .config import Config, get_track, discover_tracks

console = Console()


def get_db_path() -> Path:
    """Get path to experiments database."""
    config_dir = Config.get_config_dir()
    return config_dir / "experiments.db"


_schema_initialized = False


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and run migrations. Called once per process."""
    global _schema_initialized
    if _schema_initialized:
        return

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
            crash_log TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL REFERENCES experiments(id),
            step INTEGER NOT NULL,
            total_steps INTEGER,
            val_loss REAL,
            train_time_ms INTEGER,
            step_avg_ms REAL,
            is_final_step BOOLEAN DEFAULT 0,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(experiment_id, step)
        );

        CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON metrics(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
        CREATE INDEX IF NOT EXISTS idx_experiments_track ON experiments(track);
    """)
    conn.commit()

    # Migration: add new columns if they don't exist (for existing DBs)
    cursor = conn.execute("PRAGMA table_info(experiments)")
    columns = [row[1] for row in cursor.fetchall()]
    migrations = {
        "code_hash": "TEXT",
        "parent_hash": "TEXT",
        "remote_run_id": "TEXT",
        "crash_log": "TEXT",
        "deleted": "INTEGER DEFAULT 0",
        "gpu_type": "TEXT DEFAULT 'H100'",
        "kernels_path": "TEXT",
        "session_name": "TEXT",
    }
    for col, col_type in migrations.items():
        if col not in columns:
            conn.execute(f"ALTER TABLE experiments ADD COLUMN {col} {col_type}")
    conn.commit()

    _schema_initialized = True


def get_db() -> sqlite3.Connection:
    """Get database connection, initializing schema if needed."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


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
    session_name: Optional[str] = None

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
            crash_log=row["crash_log"] if "crash_log" in keys else None,
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            kernels_path=row["kernels_path"] if "kernels_path" in keys else None,
            session_name=row["session_name"] if "session_name" in keys else None,
        )


@dataclass
class Metric:
    """A single metric checkpoint."""
    step: int
    total_steps: Optional[int]
    val_loss: Optional[float]
    train_time_ms: Optional[int]
    step_avg_ms: Optional[float]
    is_final_step: bool
    recorded_at: datetime


# =============================================================================
# Database operations
# =============================================================================

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
    session_name: Optional[str] = None,
) -> int:
    """Create a new experiment record. Returns the experiment ID."""
    conn = get_db()
    cursor = conn.execute(
        """
        INSERT INTO experiments (name, track, script, code_hash, parent_hash, git_commit, env_vars, gpus, gpu_type, run_number, tmux_window, session_name)
        VALUES (:name, :track, :script, :code_hash, :parent_hash, :git_commit, :env_vars, :gpus, :gpu_type, :run_number, :tmux_window, :session_name)
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
            "session_name": session_name,
        }
    )
    conn.commit()
    exp_id = cursor.lastrowid
    conn.close()
    return exp_id


def record_metric(
    experiment_id: int,
    step: int,
    total_steps: Optional[int] = None,
    val_loss: Optional[float] = None,
    train_time_ms: Optional[int] = None,
    step_avg_ms: Optional[float] = None,
    is_final_step: bool = False,
) -> bool:
    """Record a metric checkpoint. Returns True if a new row was inserted."""
    conn = get_db()
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO metrics
            (experiment_id, step, total_steps, val_loss, train_time_ms, step_avg_ms, is_final_step)
        VALUES
            (:experiment_id, :step, :total_steps, :val_loss, :train_time_ms, :step_avg_ms, :is_final_step)
        """,
        {
            "experiment_id": experiment_id,
            "step": step,
            "total_steps": total_steps,
            "val_loss": val_loss,
            "train_time_ms": train_time_ms,
            "step_avg_ms": step_avg_ms,
            "is_final_step": is_final_step,
        }
    )
    inserted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return inserted


def update_experiment_status(experiment_id: int, status: str) -> None:
    """Update experiment status (running, completed, failed)."""
    conn = get_db()
    if status in ("completed", "failed", "cancelled"):
        conn.execute(
            "UPDATE experiments SET status = ?, finished_at = ? WHERE id = ?",
            (status, datetime.now(timezone.utc).isoformat(), experiment_id)
        )
    elif status == "running":
        # Set started_at when transitioning to running (only if not already set)
        conn.execute(
            "UPDATE experiments SET status = ?, started_at = COALESCE(started_at, ?) WHERE id = ?",
            (status, datetime.now(timezone.utc).isoformat(), experiment_id)
        )
    else:
        conn.execute(
            "UPDATE experiments SET status = ? WHERE id = ?",
            (status, experiment_id)
        )
    conn.commit()
    conn.close()


def update_experiment_metadata(
    experiment_id: int,
    code_hash: Optional[str] = None,
    tmux_window: Optional[str] = None,
    remote_run_id: Optional[str] = None,
    crash_log: Optional[str] = None,
    started_at: Optional[str] = None,
    git_commit: Optional[str] = None,
    parent_hash: Optional[str] = None,
    kernels_path: Optional[str] = None,
    session_name: Optional[str] = None,
) -> None:
    """Update experiment metadata.

    Used to update fields that are set by the daemon after experiment creation.
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
    if crash_log is not None:
        updates.append("crash_log = ?")
        params.append(crash_log)
    if started_at is not None:
        updates.append("started_at = ?")
        params.append(started_at)
    if git_commit is not None:
        updates.append("git_commit = ?")
        params.append(git_commit)
    if parent_hash is not None:
        updates.append("parent_hash = ?")
        params.append(parent_hash)
    if kernels_path is not None:
        updates.append("kernels_path = ?")
        params.append(kernels_path)
    if session_name is not None:
        updates.append("session_name = ?")
        params.append(session_name)

    if updates:
        params.append(experiment_id)
        query = f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?"
        conn.execute(query, params)
        conn.commit()
    conn.close()


def get_experiment(experiment_id: int) -> Optional[Experiment]:
    """Get an experiment by ID."""
    conn = get_db()
    row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    conn.close()
    return Experiment.from_row(row) if row else None


def delete_experiment(experiment_id: int) -> None:
    """Soft delete an experiment (sets deleted flag)."""
    conn = get_db()
    conn.execute("UPDATE experiments SET deleted = 1 WHERE id = ?", (experiment_id,))
    conn.commit()
    conn.close()


def get_experiment_by_window(tmux_window: str) -> Optional[Experiment]:
    """Get an experiment by tmux window name."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM experiments WHERE tmux_window = ? ORDER BY started_at DESC LIMIT 1",
        (tmux_window,)
    ).fetchone()
    conn.close()
    return Experiment.from_row(row) if row else None


def get_running_experiments(session_name: Optional[str] = None) -> List[Experiment]:
    """Get all running experiments, optionally filtered by session."""
    conn = get_db()
    query = "SELECT * FROM experiments WHERE status = 'running' AND (deleted IS NULL OR deleted = 0)"
    params = []
    if session_name:
        query += " AND session_name = ?"
        params.append(session_name)
    query += " ORDER BY started_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [Experiment.from_row(row) for row in rows]


def get_experiments_by_status(statuses: List[str]) -> List[Experiment]:
    """Get experiments with any of the given statuses."""
    conn = get_db()
    placeholders = ",".join("?" * len(statuses))
    rows = conn.execute(
        f"SELECT * FROM experiments WHERE status IN ({placeholders}) AND (deleted IS NULL OR deleted = 0) ORDER BY started_at DESC",
        statuses
    ).fetchall()
    conn.close()
    return [Experiment.from_row(row) for row in rows]


def get_all_experiment_ids() -> set:
    """Get set of all experiment IDs in local database."""
    conn = get_db()
    rows = conn.execute("SELECT id FROM experiments").fetchall()
    conn.close()
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
    session_name: Optional[str] = None,
) -> int:
    """Create experiment with explicit ID (for recovery from remote).

    Unlike create_experiment(), this uses a specific ID rather than auto-increment.
    Used when syncing experiments that were created by remote daemon.

    Returns the experiment ID.
    """
    conn = get_db()
    conn.execute(
        """
        INSERT INTO experiments (id, name, track, script, code_hash, parent_hash, git_commit, env_vars, gpus, gpu_type, tmux_window, remote_run_id, status, crash_log, started_at, finished_at, kernels_path, session_name)
        VALUES (:id, :name, :track, :script, :code_hash, :parent_hash, :git_commit, :env_vars, :gpus, :gpu_type, :tmux_window, :remote_run_id, :status, :crash_log, :started_at, :finished_at, :kernels_path, :session_name)
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
            "crash_log": crash_log,
            "started_at": started_at,
            "finished_at": finished_at,
            "kernels_path": kernels_path,
            "session_name": session_name,
        }
    )
    conn.commit()
    conn.close()
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
    conn.close()
    return [Experiment.from_row(row) for row in rows]


def get_metrics(experiment_id: int) -> List[Metric]:
    """Get all metrics for an experiment."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM metrics WHERE experiment_id = :id ORDER BY step",
        {"id": experiment_id}
    ).fetchall()
    conn.close()
    return [
        Metric(
            step=row["step"],
            total_steps=row["total_steps"],
            val_loss=row["val_loss"],
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
    conn.close()
    if not row:
        return None
    return Metric(
        step=row["step"],
        total_steps=row["total_steps"],
        val_loss=row["val_loss"],
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
    conn.close()
    if not row:
        return None
    return Metric(
        step=row["step"],
        total_steps=row["total_steps"],
        val_loss=row["val_loss"],
        train_time_ms=row["train_time_ms"],
        step_avg_ms=row["step_avg_ms"],
        is_final_step=True,
        recorded_at=datetime.fromisoformat(row["recorded_at"]) if row["recorded_at"] else None,
    )


def get_final_metrics(experiment_id: int) -> Optional[Tuple[float, int, int]]:
    """Get final val_loss, train_time_ms, steps for an experiment."""
    metric = get_latest_metric(experiment_id)
    if metric and metric.val_loss:
        return (metric.val_loss, metric.train_time_ms, metric.step)
    return None


# =============================================================================
# Log parsing
# =============================================================================

# Metric line fields: step:N/M, val_loss:X, train_time:Y(ms|s), step_avg:Zms.
# Each field matched independently so unknown fields (e.g. epoch:3.5) can be
# interleaved without breaking parsing. Numbers require a leading digit so
# unfilled f-string placeholders like `val_loss:{final_val_loss:.4f}` don't
# hit the matcher (the `.4` in `:.4f}` would otherwise parse as 0.4).
_NUM = r"\d+(?:\.\d+)?"
_STEP_FIELD = re.compile(rf"step:(\d+)/(\d+)")
_VAL_LOSS_FIELD = re.compile(rf"val_loss:({_NUM})")
_TRAIN_TIME_FIELD = re.compile(rf"train_time:({_NUM})(ms|s)")
_STEP_AVG_FIELD = re.compile(rf"step_avg:({_NUM})ms")


def parse_metric_line(line: str) -> Optional[Dict]:
    """Parse a single log line for metrics. Returns dict or None.

    Requires step/val_loss/train_time fields; step_avg is optional. Additional
    fields in the log line (e.g. epoch:3.5) are ignored. train_time accepts
    integer ms ('542512ms') or float seconds ('542.512s').
    """
    step_match = _STEP_FIELD.search(line)
    val_loss_match = _VAL_LOSS_FIELD.search(line)
    train_time_match = _TRAIN_TIME_FIELD.search(line)
    if not (step_match and val_loss_match and train_time_match):
        return None

    train_time_val = float(train_time_match.group(1))
    train_time_ms = int(train_time_val * 1000) if train_time_match.group(2) == "s" else int(train_time_val)

    step_avg_match = _STEP_AVG_FIELD.search(line)
    return {
        "step": int(step_match.group(1)),
        "total_steps": int(step_match.group(2)),
        "val_loss": float(val_loss_match.group(1)),
        "train_time_ms": train_time_ms,
        "step_avg_ms": float(step_avg_match.group(1)) if step_avg_match else None,
    }


def parse_log_content(content: str) -> List[Dict]:
    """Parse log content and return all metric checkpoints."""
    metrics = []
    for line in content.split("\n"):
        metric = parse_metric_line(line)
        if metric:
            metrics.append(metric)
    return metrics


# =============================================================================
# Display functions
# =============================================================================

def show_results(track_filter: Optional[str] = None) -> None:
    """Show results of experiments from the database."""
    experiments = get_experiments(track=track_filter, limit=100)

    if not experiments:
        console.print("[yellow]No experiments found.[/yellow]")
        console.print("Run an experiment with: nanorun run <script>")
        return

    # Group by track
    by_track: Dict[str, List[Experiment]] = {"[untracked]": []}
    for exp in experiments:
        track = exp.track or "[untracked]"
        if track not in by_track:
            by_track[track] = []
        by_track[track].append(exp)

    all_with_metrics = []

    for track_name, exps in by_track.items():
        if not exps:
            continue

        # Show track header
        if track_name != "[untracked]":
            track_info = get_track(track_name)
            desc = f" - {track_info.description}" if track_info and track_info.description else ""
            console.print(f"\n[bold cyan]{track_name}[/bold cyan]{desc}")
        else:
            console.print(f"\n[dim]{track_name}[/dim]")

        table = Table(show_header=True, header_style="dim")
        table.add_column("Name", style="cyan", max_width=30)
        table.add_column("Status", justify="center")
        table.add_column("Val Loss", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Steps", justify="right")

        for exp in exps[:10]:
            final = get_final_metrics(exp.id)

            if final:
                val_loss, train_time, steps = final
                val_loss_str = f"{val_loss:.4f}"
                if val_loss < 3.30:
                    val_loss_str = f"[green]{val_loss_str}[/green]"
                time_str = f"{train_time/1000:.1f}s" if train_time else "[dim]n/a[/dim]"
                steps_str = str(steps)
                all_with_metrics.append((exp, val_loss))
            else:
                val_loss_str = "[dim]n/a[/dim]"
                time_str = "[dim]n/a[/dim]"
                steps_str = "[dim]n/a[/dim]"

            status_str = {
                "running": "[yellow]running[/yellow]",
                "completed": "[green]done[/green]",
                "failed": "[red]failed[/red]",
                "cancelled": "[dim]cancelled[/dim]",
            }.get(exp.status, exp.status)

            table.add_row(exp.name[:30], status_str, val_loss_str, time_str, steps_str)

        console.print(table)

        if len(exps) > 10:
            console.print(f"[dim]  ... and {len(exps) - 10} more[/dim]")

    # Show overall best
    if all_with_metrics:
        best_exp, best_loss = min(all_with_metrics, key=lambda x: x[1])
        track_str = f" ({best_exp.track})" if best_exp.track else ""
        console.print(f"\n[green]Best:[/green] {best_exp.name}{track_str} - val_loss: {best_loss:.4f}")


def show_experiment_detail(experiment_id: int) -> None:
    """Show detailed info about an experiment including loss curve."""
    exp = get_experiment(experiment_id)
    if not exp:
        console.print(f"[red]Experiment {experiment_id} not found[/red]")
        return

    console.print(f"\n[bold cyan]{exp.name}[/bold cyan]")
    console.print(f"  ID: {exp.id}")
    console.print(f"  Script: {exp.script}")
    console.print(f"  Track: {exp.track or '[none]'}")
    console.print(f"  Status: {exp.status}")
    console.print(f"  GPUs: {exp.gpus}x {exp.gpu_type}")
    console.print(f"  Started: {exp.started_at}")
    if exp.finished_at:
        console.print(f"  Finished: {exp.finished_at}")
    if exp.env_vars:
        console.print(f"  Env: {exp.env_vars}")

    # Show metrics
    metrics = get_metrics(exp.id)
    if metrics:
        console.print(f"\n  [dim]Metrics ({len(metrics)} checkpoints):[/dim]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("Step", justify="right")
        table.add_column("Val Loss", justify="right")
        table.add_column("Time", justify="right")

        # Show last 10
        for m in metrics[-10:]:
            table.add_row(
                f"{m.step}/{m.total_steps}" if m.total_steps else str(m.step),
                f"{m.val_loss:.4f}" if m.val_loss else "-",
                f"{m.train_time_ms/1000:.1f}s" if m.train_time_ms else "-",
            )
        console.print(table)


def show_diff(exp1_name: str, exp2_name: str) -> None:
    """Show diff between two experiments."""
    # Find experiments by name
    conn = get_db()
    row1 = conn.execute(
        "SELECT * FROM experiments WHERE name LIKE ? ORDER BY started_at DESC LIMIT 1",
        (f"%{exp1_name}%",)
    ).fetchone()
    row2 = conn.execute(
        "SELECT * FROM experiments WHERE name LIKE ? ORDER BY started_at DESC LIMIT 1",
        (f"%{exp2_name}%",)
    ).fetchone()
    conn.close()

    if not row1:
        console.print(f"[red]Could not find experiment matching '{exp1_name}'[/red]")
        return
    if not row2:
        console.print(f"[red]Could not find experiment matching '{exp2_name}'[/red]")
        return

    exp1 = Experiment.from_row(row1)
    exp2 = Experiment.from_row(row2)

    console.print(f"[cyan]Comparing {exp1.name} vs {exp2.name}[/cyan]")

    # Get final metrics
    final1 = get_final_metrics(exp1.id)
    final2 = get_final_metrics(exp2.id)

    table = Table(title="Metrics Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column(exp1.name[:20])
    table.add_column(exp2.name[:20])
    table.add_column("Diff")

    if final1 and final2:
        val1, time1, steps1 = final1
        val2, time2, steps2 = final2

        diff = val2 - val1
        diff_str = f"{diff:+.4f}"
        if diff < 0:
            diff_str = f"[green]{diff_str}[/green]"
        elif diff > 0:
            diff_str = f"[red]{diff_str}[/red]"
        table.add_row("Val Loss", f"{val1:.4f}", f"{val2:.4f}", diff_str)

        if time1 and time2:
            time_diff = time2 - time1
            table.add_row(
                "Train Time",
                f"{time1/1000:.1f}s",
                f"{time2/1000:.1f}s",
                f"{time_diff/1000:+.1f}s",
            )

    console.print(table)

    # Show env var differences
    if exp1.env_vars or exp2.env_vars:
        all_keys = set(exp1.env_vars.keys()) | set(exp2.env_vars.keys())
        if all_keys:
            console.print("\n[dim]Environment differences:[/dim]")
            for key in sorted(all_keys):
                v1 = exp1.env_vars.get(key, "[not set]")
                v2 = exp2.env_vars.get(key, "[not set]")
                if v1 != v2:
                    console.print(f"  {key}: {v1} → {v2}")
