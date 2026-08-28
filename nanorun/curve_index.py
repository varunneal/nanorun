"""Persistent multiresolution extrema index for dashboard loss curves."""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from typing import Any, Iterable


BASE_BUCKET_WIDTH = 8
RAW_CURVE_SCAN_LIMIT = 10_000
LOSS_SERIES = ("val_loss", "train_loss")


def _winner(
    candidates: Iterable[tuple[int, float]], *, maximum: bool,
) -> tuple[int, float]:
    """Choose an extremum with the same earliest-step tie break as curves."""
    if maximum:
        return min(candidates, key=lambda item: (-item[1], item[0]))
    return min(candidates, key=lambda item: (item[1], item[0]))


def _aggregate_rows(rows: Iterable[sqlite3.Row], metric_name: str) -> tuple[int, float, int, float, int] | None:
    points = [(int(row["step"]), float(row[metric_name])) for row in rows if row[metric_name] is not None]
    if not points:
        return None
    min_step, min_value = _winner(points, maximum=False)
    max_step, max_value = _winner(points, maximum=True)
    return min_step, min_value, max_step, max_value, len(points)


def _upsert_bin(
    conn: sqlite3.Connection,
    experiment_id: int,
    metric_name: str,
    level: int,
    bucket: int,
    values: tuple[int, float, int, float, int],
) -> None:
    conn.execute(
        """INSERT INTO metric_curve_bins
               (experiment_id, metric_name, level, bucket,
                min_step, min_value, max_step, max_value, point_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(experiment_id, metric_name, level, bucket) DO UPDATE SET
               min_step=excluded.min_step, min_value=excluded.min_value,
               max_step=excluded.max_step, max_value=excluded.max_value,
               point_count=excluded.point_count""",
        (experiment_id, metric_name, level, bucket, *values),
    )


def _replace_parent_bin(
    conn: sqlite3.Connection,
    experiment_id: int,
    metric_name: str,
    level: int,
    bucket: int,
) -> bool:
    children = conn.execute(
        """SELECT min_step, min_value, max_step, max_value, point_count
           FROM metric_curve_bins
           WHERE experiment_id=? AND metric_name=? AND level=?
             AND bucket IN (?, ?)""",
        (experiment_id, metric_name, level - 1, bucket * 2, bucket * 2 + 1),
    ).fetchall()
    if not children:
        conn.execute(
            "DELETE FROM metric_curve_bins WHERE experiment_id=? AND metric_name=? AND level=? AND bucket=?",
            (experiment_id, metric_name, level, bucket),
        )
        return False
    minimum = _winner(
        ((int(row["min_step"]), float(row["min_value"])) for row in children),
        maximum=False,
    )
    maximum = _winner(
        ((int(row["max_step"]), float(row["max_value"])) for row in children),
        maximum=True,
    )
    point_count = sum(int(row["point_count"]) for row in children)
    _upsert_bin(conn, experiment_id, metric_name, level, bucket, (*minimum, *maximum, point_count))
    return True


def rebuild_curve_index(
    conn: sqlite3.Connection,
    experiment_id: int,
    metrics_revision: int,
) -> None:
    """Rebuild one experiment's derived index inside the caller transaction."""
    rows = conn.execute(
        """SELECT step, val_loss, train_loss FROM metrics
           WHERE experiment_id=? ORDER BY step""",
        (experiment_id,),
    ).fetchall()
    conn.execute("DELETE FROM metric_curve_bins WHERE experiment_id=?", (experiment_id,))
    conn.execute("DELETE FROM metric_curve_index_state WHERE experiment_id=?", (experiment_id,))

    for metric_name in LOSS_SERIES:
        grouped: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            if row[metric_name] is not None:
                grouped[int(row["step"]) // BASE_BUCKET_WIDTH].append(row)
        if not grouped:
            continue
        current: dict[int, tuple[int, float, int, float, int]] = {}
        for bucket, bucket_rows in grouped.items():
            values = _aggregate_rows(bucket_rows, metric_name)
            assert values is not None
            current[bucket] = values
            _upsert_bin(conn, experiment_id, metric_name, 0, bucket, values)
        level = 1
        while len(current) > 1:
            parents: dict[int, list[tuple[int, float, int, float, int]]] = defaultdict(list)
            for bucket, values in current.items():
                parents[bucket // 2].append(values)
            next_level: dict[int, tuple[int, float, int, float, int]] = {}
            for bucket, children in parents.items():
                minimum = _winner(((item[0], item[1]) for item in children), maximum=False)
                maximum = _winner(((item[2], item[3]) for item in children), maximum=True)
                values = (*minimum, *maximum, sum(item[4] for item in children))
                next_level[bucket] = values
                _upsert_bin(conn, experiment_id, metric_name, level, bucket, values)
            current = next_level
            level += 1
        points = [row for row in rows if row[metric_name] is not None]
        conn.execute(
            """INSERT INTO metric_curve_index_state
                   (experiment_id, metric_name, metrics_revision, min_step, max_step, point_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (experiment_id, metric_name, metrics_revision, points[0]["step"], points[-1]["step"], len(points)),
        )


def update_curve_index(
    conn: sqlite3.Connection,
    experiment_id: int,
    metrics_revision: int,
    changed_steps: Iterable[int],
) -> None:
    """Refresh touched base buckets and ancestors, rebuilding if no index exists."""
    steps = sorted(set(int(step) for step in changed_steps))
    if not steps:
        return
    state_count = conn.execute(
        "SELECT COUNT(*) FROM metric_curve_index_state WHERE experiment_id=?",
        (experiment_id,),
    ).fetchone()[0]
    if state_count == 0:
        rebuild_curve_index(conn, experiment_id, metrics_revision)
        return

    base_buckets = sorted({step // BASE_BUCKET_WIDTH for step in steps})
    for metric_name in LOSS_SERIES:
        state = conn.execute(
            """SELECT min_step, max_step FROM metric_curve_index_state
               WHERE experiment_id=? AND metric_name=?""",
            (experiment_id, metric_name),
        ).fetchone()
        dirty = set(base_buckets)
        observed_steps: list[int] = []
        for bucket in base_buckets:
            bucket_rows = conn.execute(
                """SELECT step, val_loss, train_loss FROM metrics
                   WHERE experiment_id=? AND step>=? AND step<? ORDER BY step""",
                (experiment_id, bucket * BASE_BUCKET_WIDTH, (bucket + 1) * BASE_BUCKET_WIDTH),
            ).fetchall()
            values = _aggregate_rows(bucket_rows, metric_name)
            if values is None:
                conn.execute(
                    """DELETE FROM metric_curve_bins
                       WHERE experiment_id=? AND metric_name=? AND level=0 AND bucket=?""",
                    (experiment_id, metric_name, bucket),
                )
            else:
                _upsert_bin(conn, experiment_id, metric_name, 0, bucket, values)
                observed_steps.extend((values[0], values[2]))
        if state is None and not observed_steps:
            continue

        max_bucket_row = conn.execute(
            """SELECT MAX(bucket) FROM metric_curve_bins
               WHERE experiment_id=? AND metric_name=? AND level=0""",
            (experiment_id, metric_name),
        ).fetchone()
        max_bucket = max_bucket_row[0]
        if max_bucket is None:
            conn.execute(
                "DELETE FROM metric_curve_index_state WHERE experiment_id=? AND metric_name=?",
                (experiment_id, metric_name),
            )
            continue
        max_level = max(0, int(math.floor(math.log2(max_bucket))) + 1) if max_bucket else 0
        for level in range(1, max_level + 1):
            dirty = {bucket // 2 for bucket in dirty}
            for bucket in dirty:
                _replace_parent_bin(conn, experiment_id, metric_name, level, bucket)
        min_step = min(observed_steps + ([int(state["min_step"])] if state else []))
        max_step = max(observed_steps + ([int(state["max_step"])] if state else []))
        root = conn.execute(
            """SELECT point_count FROM metric_curve_bins
               WHERE experiment_id=? AND metric_name=?
               ORDER BY level DESC LIMIT 1""",
            (experiment_id, metric_name),
        ).fetchone()
        conn.execute(
            """INSERT INTO metric_curve_index_state
                   (experiment_id, metric_name, metrics_revision, min_step, max_step, point_count)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(experiment_id, metric_name) DO UPDATE SET
                   metrics_revision=excluded.metrics_revision,
                   min_step=excluded.min_step, max_step=excluded.max_step,
                   point_count=excluded.point_count""",
            (experiment_id, metric_name, metrics_revision, min_step, max_step, int(root["point_count"])),
        )


def indexed_curve_candidates(
    conn: sqlite3.Connection,
    experiment_id: int,
    metric_name: str,
    metrics_revision: int,
    max_points: int,
    state: Any = None,
) -> list[dict[str, Any]] | None:
    """Return a small extrema candidate set, or None for a stale/missing index."""
    if state is None:
        state = conn.execute(
            """SELECT metrics_revision, min_step, max_step, point_count
               FROM metric_curve_index_state WHERE experiment_id=? AND metric_name=?""",
            (experiment_id, metric_name),
        ).fetchone()
    if state is None or int(state["metrics_revision"]) != int(metrics_revision):
        return None
    if int(state["point_count"]) <= max(max_points, RAW_CURVE_SCAN_LIMIT):
        return None
    span = max(0, int(state["max_step"]) - int(state["min_step"]))
    target_candidates = max_points * 2
    level = 0
    width = BASE_BUCKET_WIDTH
    while 2 * (span // width + 2) > target_candidates:
        level += 1
        width *= 2
    rows = conn.execute(
        """SELECT b.min_step, b.max_step,
                  minm.val_loss AS min_val_loss, minm.train_loss AS min_train_loss,
                  minm.train_time_ms AS min_train_time_ms, minm.step_avg_ms AS min_step_avg_ms,
                  maxm.val_loss AS max_val_loss, maxm.train_loss AS max_train_loss,
                  maxm.train_time_ms AS max_train_time_ms, maxm.step_avg_ms AS max_step_avg_ms
           FROM metric_curve_bins b
           JOIN metrics minm ON minm.experiment_id=b.experiment_id AND minm.step=b.min_step
           JOIN metrics maxm ON maxm.experiment_id=b.experiment_id AND maxm.step=b.max_step
           WHERE b.experiment_id=? AND b.metric_name=? AND b.level=?
           ORDER BY b.bucket""",
        (experiment_id, metric_name, level),
    ).fetchall()
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        for prefix in ("min", "max"):
            step = int(row[f"{prefix}_step"])
            by_step[step] = {
                "step": step,
                "loss": row[f"{prefix}_{metric_name}"],
                "val_loss": row[f"{prefix}_val_loss"],
                "train_loss": row[f"{prefix}_train_loss"],
                "train_time_ms": row[f"{prefix}_train_time_ms"],
                "step_avg_ms": row[f"{prefix}_step_avg_ms"],
            }
    endpoints = conn.execute(
        """SELECT step, val_loss, train_loss, train_time_ms, step_avg_ms
           FROM metrics WHERE experiment_id=? AND step IN (?, ?)""",
        (experiment_id, state["min_step"], state["max_step"]),
    ).fetchall()
    for row in endpoints:
        step = int(row["step"])
        by_step[step] = {
            "step": step,
            "loss": row[metric_name],
            "val_loss": row["val_loss"],
            "train_loss": row["train_loss"],
            "train_time_ms": row["train_time_ms"],
            "step_avg_ms": row["step_avg_ms"],
        }
    return [by_step[step] for step in sorted(by_step)]


def ensure_curve_index(
    conn: sqlite3.Connection,
    experiment_id: int,
    metric_names: Iterable[str],
    metrics_revision: int,
) -> bool:
    """Synchronously repair an absent/stale requested index. Returns whether rebuilt."""
    names = set(metric_names)
    rows = conn.execute(
        """SELECT metric_name, metrics_revision FROM metric_curve_index_state
           WHERE experiment_id=?""",
        (experiment_id,),
    ).fetchall()
    revisions = {row["metric_name"]: int(row["metrics_revision"]) for row in rows}
    if all(revisions.get(name) == int(metrics_revision) for name in names):
        return False
    rebuild_curve_index(conn, experiment_id, metrics_revision)
    conn.commit()
    return True


def backfill_next_curve_index(conn: sqlite3.Connection) -> int | None:
    """Index the newest completed experiment not current in the derived index."""
    row = conn.execute(
        """SELECT e.id, e.metrics_revision
           FROM experiments e
           WHERE e.status='completed' AND COALESCE(e.deleted, 0)=0
             AND EXISTS (SELECT 1 FROM metrics m WHERE m.experiment_id=e.id)
             AND EXISTS (
                 SELECT 1 FROM (SELECT 'val_loss' AS metric_name UNION ALL SELECT 'train_loss') names
                 WHERE EXISTS (
                     SELECT 1 FROM metrics m WHERE m.experiment_id=e.id
                       AND CASE names.metric_name WHEN 'val_loss' THEN m.val_loss ELSE m.train_loss END IS NOT NULL
                 ) AND NOT EXISTS (
                     SELECT 1 FROM metric_curve_index_state s
                     WHERE s.experiment_id=e.id AND s.metric_name=names.metric_name
                       AND s.metrics_revision=e.metrics_revision
                 )
             )
           ORDER BY e.started_at DESC, e.id DESC LIMIT 1"""
    ).fetchone()
    if row is None:
        return None
    rebuild_curve_index(conn, int(row["id"]), int(row["metrics_revision"]))
    conn.commit()
    return int(row["id"])
