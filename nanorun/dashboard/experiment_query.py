"""Batched, revision-aware experiment query engine for the dashboard."""

from __future__ import annotations

import base64
import binascii
import json
import math
import sqlite3
from collections import defaultdict
from typing import Any, Iterable, Optional

from ..tracker import get_db, read_experiment_summaries


SCHEMA_VERSION = 1
MAX_NAMED_QUERIES = 16
MAX_EXPLICIT_IDS = 10_000
MAX_DISCOVERY_GROUPS = 500
MAX_POINTS_MIN = 4
MAX_POINTS_MAX = 1200
RAW_CHUNK_ROWS = 500
LOSS_SERIES = ("val_loss", "train_loss")
PROJECTIONS = {"summary", "summaries", "curves", "metadata", "raw_metrics"}
PROJECTION_ALIASES = {
    "bounded_curves": "curves",
    "full_metadata": "metadata",
    "metrics": "raw_metrics",
}


class QueryValidationError(ValueError):
    """The request is invalid and must fail before streaming starts."""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise QueryValidationError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise QueryValidationError(f"{field} must be an integer") from exc
    if parsed <= 0:
        raise QueryValidationError(f"{field} must be positive")
    return parsed


def _normalize_projections(query: dict[str, Any]) -> set[str]:
    raw = query.get("projections", query.get("projection", ["summary"]))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise QueryValidationError("projection must be a string or non-empty list")
    projections = {PROJECTION_ALIASES.get(str(item), str(item)) for item in raw}
    if not projections <= PROJECTIONS:
        unsupported = sorted(projections - PROJECTIONS)
        raise QueryValidationError(f"unsupported projection(s): {', '.join(unsupported)}")
    if "summaries" in projections:
        projections.remove("summaries")
        projections.add("summary")
    return projections


def _normalize_selector(query: dict[str, Any]) -> dict[str, Any]:
    selector = query.get("selector")
    if selector is None:
        # Convenient short form for controlled dashboard callers.
        selector = {key: query[key] for key in (
            "type", "ids", "id", "expand_group", "track", "status", "search",
            "limit", "cursor", "group", "code_hash", "gpus", "gpu_type",
        ) if key in query}
    if not isinstance(selector, dict):
        raise QueryValidationError("selector must be an object")
    selector = dict(selector)
    kind = selector.get("type") or selector.get("kind")
    if not kind:
        if "ids" in selector:
            kind = "ids"
        elif "id" in selector:
            kind = "experiment"
        elif "group" in selector or "code_hash" in selector:
            kind = "group"
        else:
            kind = "discovery"
    aliases = {
        "filtered": "discovery", "filtered_discovery": "discovery",
        "group_identity": "group", "explicit": "ids", "explicit_ids": "ids",
        "id": "experiment", "one": "experiment",
    }
    kind = aliases.get(str(kind), str(kind))
    if kind not in {"discovery", "group", "ids", "experiment"}:
        raise QueryValidationError(f"unsupported selector type: {kind}")
    selector["type"] = kind

    if kind == "ids":
        ids = selector.get("ids")
        if not isinstance(ids, list):
            raise QueryValidationError("ids selector requires an ids list")
        if len(ids) > MAX_EXPLICIT_IDS:
            raise QueryValidationError(f"ids may contain at most {MAX_EXPLICIT_IDS} values")
        selector["ids"] = list(dict.fromkeys(_positive_int(value, "id") for value in ids))
    elif kind == "experiment":
        selector["id"] = _positive_int(selector.get("id"), "id")
        selector["expand_group"] = bool(selector.get("expand_group", False))
    elif kind == "discovery":
        limit = _positive_int(selector.get("limit", 100), "limit")
        selector["limit"] = min(limit, MAX_DISCOVERY_GROUPS)
        for field in ("track", "status", "search"):
            if selector.get(field) is not None and not isinstance(selector[field], str):
                raise QueryValidationError(f"{field} must be a string")
        _decode_cursor(selector.get("cursor"))
    else:
        group = selector.get("group", selector)
        if not isinstance(group, dict):
            raise QueryValidationError("group selector requires a group object")
        if group.get("code_hash") is None and group.get("id") is None:
            raise QueryValidationError("group selector requires code_hash or id")
        if group.get("id") is not None:
            group = dict(group)
            group["id"] = _positive_int(group["id"], "group.id")
        if group.get("code_hash") is not None and not isinstance(group["code_hash"], str):
            raise QueryValidationError("group.code_hash must be a string")
        for field in ("track", "gpu_type"):
            if group.get(field) is not None and not isinstance(group[field], str):
                raise QueryValidationError(f"group.{field} must be a string")
        if group.get("gpus") is not None:
            group = dict(group)
            group["gpus"] = _positive_int(group["gpus"], "group.gpus")
        selector["group"] = group
    return selector


def _normalize_known(raw: Any, nested_field: str) -> dict[int, int]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise QueryValidationError("known revisions must be an object")
    known: dict[int, int] = {}
    for raw_id, raw_revision in raw.items():
        experiment_id = _positive_int(raw_id, "known experiment id")
        if isinstance(raw_revision, dict):
            aliases = (
                ("revision", "experiment_revision")
                if nested_field == "revision"
                else ("metrics_revision", "metric_revision")
            )
            raw_revision = next(
                (raw_revision[name] for name in aliases if name in raw_revision), None
            )
            if raw_revision is None:
                continue
        if isinstance(raw_revision, bool):
            raise QueryValidationError("known revision must be a non-negative integer")
        try:
            revision = int(raw_revision)
        except (TypeError, ValueError) as exc:
            raise QueryValidationError("known revision must be a non-negative integer") from exc
        if revision < 0:
            raise QueryValidationError("known revision must be a non-negative integer")
        known[experiment_id] = revision
    return known


def validate_query_request(body: Any) -> dict[str, Any]:
    """Normalize the public request shape without touching the database."""
    if not isinstance(body, dict):
        raise QueryValidationError("request body must be an object")
    raw_queries = body.get("queries")
    if isinstance(raw_queries, dict):
        query_items = list(raw_queries.items())
    elif isinstance(raw_queries, list):
        query_items = []
        seen = set()
        for item in raw_queries:
            if not isinstance(item, dict) or not isinstance(item.get("key"), str):
                raise QueryValidationError("each query requires a string key")
            key = item["key"]
            if key in seen:
                raise QueryValidationError(f"duplicate query key: {key}")
            seen.add(key)
            query_items.append((key, {k: v for k, v in item.items() if k != "key"}))
    else:
        raise QueryValidationError("queries must be an object or list")
    if not query_items:
        raise QueryValidationError("at least one named query is required")
    if len(query_items) > MAX_NAMED_QUERIES:
        raise QueryValidationError(f"at most {MAX_NAMED_QUERIES} queries are allowed")

    normalized_queries: dict[str, dict[str, Any]] = {}
    for key, raw_query in query_items:
        if not isinstance(key, str) or not key or len(key) > 100:
            raise QueryValidationError("query keys must be non-empty strings up to 100 characters")
        if not isinstance(raw_query, dict):
            raise QueryValidationError(f"query {key!r} must be an object")
        projections = _normalize_projections(raw_query)
        curves = raw_query.get("curves", raw_query.get("curve", {}))
        if curves is None:
            curves = {}
        if not isinstance(curves, dict):
            raise QueryValidationError("curves must be an object")
        curves = dict(curves)
        if "series" not in curves and "series" in raw_query:
            curves["series"] = raw_query["series"]
        if "max_points" not in curves and "max_points" in raw_query:
            curves["max_points"] = raw_query["max_points"]
        series = curves.get("series", list(LOSS_SERIES))
        if isinstance(series, str):
            series = [series]
        if not isinstance(series, list) or any(item not in LOSS_SERIES for item in series):
            raise QueryValidationError("curve series must contain val_loss and/or train_loss")
        max_points = _positive_int(curves.get("max_points", 1200), "max_points")
        if not MAX_POINTS_MIN <= max_points <= MAX_POINTS_MAX:
            raise QueryValidationError(
                f"max_points must be between {MAX_POINTS_MIN} and {MAX_POINTS_MAX}"
            )
        normalized_queries[key] = {
            "selector": _normalize_selector(raw_query),
            "projections": projections,
            "series": list(dict.fromkeys(series)),
            "max_points": max_points,
        }

    cache_state = body.get("cache_state", body.get("known", {})) or {}
    if not isinstance(cache_state, dict):
        raise QueryValidationError("cache_state must be an object")
    known_experiments = cache_state.get(
        "experiments", body.get("known_experiment_revisions")
    )
    known_metrics = cache_state.get(
        "metrics", cache_state.get("metric_revisions", body.get("known_metric_revisions"))
    )
    if known_experiments is None and any(str(key).isdigit() for key in cache_state):
        known_experiments = cache_state
    if known_metrics is None and any(str(key).isdigit() for key in cache_state):
        known_metrics = cache_state
    return {
        "queries": normalized_queries,
        "known_experiments": _normalize_known(known_experiments, "revision"),
        "known_metrics": _normalize_known(known_metrics, "metrics_revision"),
    }


def _encode_cursor(started_at: Optional[str], experiment_id: int) -> str:
    raw = json.dumps([started_at or "", int(experiment_id)], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: Any) -> Optional[tuple[str, int]]:
    if cursor in (None, ""):
        return None
    if isinstance(cursor, dict):
        started_at = str(cursor.get("started_at") or "")
        return started_at, _positive_int(cursor.get("id"), "cursor.id")
    if not isinstance(cursor, str):
        raise QueryValidationError("cursor must be a string")
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(cursor + padding))
        if not isinstance(decoded, list) or len(decoded) != 2:
            raise ValueError
        return str(decoded[0]), _positive_int(decoded[1], "cursor.id")
    except (
        ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error,
    ) as exc:
        raise QueryValidationError("invalid pagination cursor") from exc


def _discovery_members(conn: sqlite3.Connection, selector: dict[str, Any]) -> dict[str, Any]:
    filters = ["COALESCE(deleted, 0) = 0"]
    params: list[Any] = []
    if selector.get("track"):
        filters.append("track = ?")
        params.append(selector["track"])
    if selector.get("status"):
        filters.append("status = ?")
        params.append(selector["status"])
    if selector.get("search"):
        pattern = f"%{selector['search']}%"
        filters.append("(name LIKE ? OR script LIKE ? OR track LIKE ? OR code_hash LIKE ?)")
        params.extend([pattern] * 4)
    cursor = _decode_cursor(selector.get("cursor"))
    cursor_sql = ""
    if cursor:
        cursor_sql = (
            "WHERE (max_started < ? OR (max_started = ? AND max_id < ?))"
        )
        params.extend([cursor[0], cursor[0], cursor[1]])
    limit = selector["limit"]
    params.append(limit + 1)
    rows = conn.execute(
        f"""
        WITH visible AS (
            SELECT *,
                   COALESCE(code_hash, '_no_hash_' || id) AS query_group_hash,
                   COALESCE(track, '') AS query_group_track,
                   COALESCE(gpu_type, 'H100') AS query_group_gpu_type
            FROM experiments WHERE COALESCE(deleted, 0) = 0
        ), matched AS (
            SELECT * FROM visible WHERE {' AND '.join(filters)}
        ), grouped AS (
            SELECT query_group_hash, query_group_track, gpus, query_group_gpu_type,
                   MAX(started_at) AS max_started,
                   MAX(id) AS max_id
            FROM matched
            GROUP BY query_group_hash, query_group_track, gpus, query_group_gpu_type
        ), page AS (
            SELECT * FROM grouped {cursor_sql}
            ORDER BY max_started DESC, max_id DESC LIMIT ?
        )
        SELECT visible.*, page.max_started AS page_started, page.max_id AS page_id
        FROM page JOIN visible
          ON visible.query_group_hash = page.query_group_hash
         AND visible.query_group_track = page.query_group_track
         AND visible.gpus = page.gpus
         AND visible.query_group_gpu_type = page.query_group_gpu_type
        ORDER BY page.max_started DESC, page.max_id DESC,
                 visible.started_at DESC, visible.id DESC
        """,
        params,
    ).fetchall()
    ordered_groups: list[tuple[Any, ...]] = []
    members: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    page_markers: dict[tuple[Any, ...], tuple[Optional[str], int]] = {}
    for row in rows:
        key = (
            row["query_group_hash"], row["query_group_track"], row["gpus"],
            row["query_group_gpu_type"],
        )
        if key not in members:
            ordered_groups.append(key)
            page_markers[key] = (row["page_started"], row["page_id"])
        members[key].append(row["id"])
    has_more = len(ordered_groups) > limit
    shown = ordered_groups[:limit]
    next_cursor = None
    if has_more and shown:
        next_cursor = _encode_cursor(*page_markers[shown[-1]])
    return {
        "ids": [experiment_id for key in shown for experiment_id in members[key]],
        "groups": [(key, members[key]) for key in shown],
        "next_cursor": next_cursor,
    }


def _group_members(conn: sqlite3.Connection, selector: dict[str, Any]) -> list[int]:
    group = selector["group"]
    if group.get("id") is not None and group.get("code_hash") is None:
        target = group["id"]
        rows = conn.execute(
            """WITH selected AS (
                   SELECT code_hash, track, gpus, COALESCE(gpu_type, 'H100') AS gpu_type
                   FROM experiments WHERE id = ? AND COALESCE(deleted, 0) = 0
               )
               SELECT e.id FROM experiments e, selected s
               WHERE COALESCE(e.deleted, 0) = 0
                 AND ((e.code_hash = s.code_hash) OR (e.code_hash IS NULL AND s.code_hash IS NULL AND e.id = ?))
                 AND COALESCE(e.track, '') = COALESCE(s.track, '')
                 AND e.gpus = s.gpus AND COALESCE(e.gpu_type, 'H100') = s.gpu_type
               ORDER BY e.started_at DESC, e.id DESC""",
            (target, target),
        ).fetchall()
        return [row["id"] for row in rows]
    clauses = ["COALESCE(deleted, 0) = 0", "code_hash = ?"]
    params: list[Any] = [group.get("code_hash")]
    if "track" in group:
        clauses.append("COALESCE(track, '') = ?")
        params.append(group.get("track") or "")
    if "gpus" in group:
        clauses.append("gpus = ?")
        params.append(_positive_int(group["gpus"], "group.gpus"))
    if "gpu_type" in group:
        clauses.append("COALESCE(gpu_type, 'H100') = ?")
        params.append(group.get("gpu_type") or "H100")
    rows = conn.execute(
        f"SELECT id FROM experiments WHERE {' AND '.join(clauses)} "
        "ORDER BY started_at DESC, id DESC",
        params,
    ).fetchall()
    return [row["id"] for row in rows]


def _resolve_selector(conn: sqlite3.Connection, selector: dict[str, Any]) -> dict[str, Any]:
    kind = selector["type"]
    if kind == "ids":
        return {"ids": selector["ids"], "groups": [], "next_cursor": None}
    if kind == "experiment" and not selector["expand_group"]:
        return {"ids": [selector["id"]], "groups": [], "next_cursor": None}
    if kind == "experiment":
        ids = _group_members(conn, {"group": {"id": selector["id"]}})
        # Preserve the requested missing id so completion can report it.
        if not ids:
            ids = [selector["id"]]
        return {"ids": ids, "groups": [], "next_cursor": None}
    if kind == "group":
        ids = _group_members(conn, selector)
        return {"ids": ids, "groups": [], "next_cursor": None}
    return _discovery_members(conn, selector)


def _aggregate_group(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    summaries.sort(key=lambda item: (item.get("started_at") or "", item["id"]), reverse=True)
    primary = summaries[0]
    losses = [item["loss"] for item in summaries if item.get("loss") is not None]
    times = [item["train_time_ms"] for item in summaries if item.get("train_time_ms") is not None]
    statuses = [item["status"] for item in summaries]
    loss_metrics = {item["loss_metric"] for item in summaries if item.get("loss") is not None}
    with_progress = next((item for item in summaries if item.get("current_step") is not None), None)
    status = "running" if "running" in statuses else (
        "completed" if "completed" in statuses else (statuses[0] if statuses else "unknown")
    )
    return {
        "id": primary["id"],
        "experiment_ids": [item["id"] for item in summaries],
        "name": primary["name"],
        "track": primary["track"],
        "script": primary["script"],
        "code_hash": primary["code_hash"],
        "status": status,
        "gpus": primary["gpus"],
        "gpu_type": primary["gpu_type"],
        "env_vars": primary["env_vars"],
        "started_at": primary["started_at"],
        "n_runs": len(summaries),
        "is_sweep": len({json.dumps(item["env_vars"], sort_keys=True) for item in summaries}) > 1,
        "current_step": with_progress["current_step"] if with_progress else None,
        "total_steps": with_progress["total_steps"] if with_progress else None,
        "val_loss": sum(losses) / len(losses) if losses else None,
        "loss": sum(losses) / len(losses) if losses else None,
        "loss_metric": next(iter(loss_metrics)) if len(loss_metrics) == 1 else None,
        "train_time_ms": sum(times) / len(times) if times else None,
        "val_losses": losses,
        "losses": losses,
        "train_times": times,
        "group": primary["group"],
        "revision": max(item["revision"] for item in summaries),
        "metrics_revision": max(item["metrics_revision"] for item in summaries),
    }


def _metric_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "step": row["step"],
        "total_steps": row["total_steps"],
        "val_loss": row["val_loss"],
        "train_loss": row["train_loss"],
        "loss": row["val_loss"] if row["val_loss"] is not None else row["train_loss"],
        "train_time_ms": row["train_time_ms"],
        "step_avg_ms": row["step_avg_ms"],
        "is_final_step": bool(row["is_final_step"]),
        "recorded_at": row["recorded_at"],
    }


def _bounded_curve(rows: list[sqlite3.Row], metric_name: str, max_points: int) -> list[dict[str, Any]]:
    series = [row for row in rows if row[metric_name] is not None]
    if len(series) <= max_points:
        selected = series
    else:
        min_step = series[0]["step"]
        max_step = series[-1]["step"]
        bucket_count = max(1, (max_points - 2) // 2)
        buckets: dict[int, list[sqlite3.Row]] = defaultdict(list)
        denominator = max(1, max_step - min_step - 1)
        for row in series[1:-1]:
            bucket = math.trunc(
                (row["step"] - min_step - 1) * bucket_count / denominator
            )
            buckets[bucket].append(row)
        selected_by_step = {min_step: series[0], max_step: series[-1]}
        for bucket_rows in buckets.values():
            minimum = min(bucket_rows, key=lambda row: (row[metric_name], row["step"]))
            maximum = min(bucket_rows, key=lambda row: (-row[metric_name], row["step"]))
            selected_by_step[minimum["step"]] = minimum
            selected_by_step[maximum["step"]] = maximum
        selected = [selected_by_step[step] for step in sorted(selected_by_step)][:max_points]
    return [{
        "step": row["step"],
        "loss": row[metric_name],
        "val_loss": row["val_loss"],
        "train_loss": row["train_loss"],
        "train_time_ms": row["train_time_ms"],
        "step_avg_ms": row["step_avg_ms"],
    } for row in selected]


def _query_metrics(
    conn: sqlite3.Connection,
    curve_requirements: dict[int, tuple[set[str], int]],
    raw_ids: set[int],
) -> tuple[dict[int, dict[str, list[dict[str, Any]]]], dict[int, list[dict[str, Any]]]]:
    scan_ids = sorted(set(curve_requirements) | raw_ids)
    if not scan_ids:
        return {}, {}
    placeholders = ",".join("?" for _ in scan_ids)
    # One statistics query for every curve candidate.  Besides documenting the
    # bounded work, this lets SQLite use the experiment range index and is kept
    # separate from the single ordered row scan used by curves and raw metrics.
    curve_ids = sorted(curve_requirements)
    if curve_ids:
        curve_placeholders = ",".join("?" for _ in curve_ids)
        conn.execute(
            f"SELECT experiment_id, COUNT(*) AS metric_count, "
            f"SUM(val_loss IS NOT NULL) AS val_count, "
            f"SUM(train_loss IS NOT NULL) AS train_count "
            f"FROM metrics WHERE experiment_id IN ({curve_placeholders}) "
            f"GROUP BY experiment_id",
            curve_ids,
        ).fetchall()
    rows = conn.execute(
        f"SELECT experiment_id, step, total_steps, val_loss, train_loss, "
        f"train_time_ms, step_avg_ms, is_final_step, recorded_at "
        f"FROM metrics WHERE experiment_id IN ({placeholders}) "
        f"ORDER BY experiment_id, step",
        scan_ids,
    ).fetchall()
    by_experiment: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        by_experiment[row["experiment_id"]].append(row)
    curves: dict[int, dict[str, list[dict[str, Any]]]] = {}
    raw: dict[int, list[dict[str, Any]]] = {}
    for experiment_id, (series, max_points) in curve_requirements.items():
        curves[experiment_id] = {
            metric_name: _bounded_curve(by_experiment[experiment_id], metric_name, max_points)
            for metric_name in LOSS_SERIES if metric_name in series
        }
    for experiment_id in raw_ids:
        raw[experiment_id] = [_metric_row(row) for row in by_experiment[experiment_id]]
    return curves, raw


def execute_experiment_query(normalized: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute all named selectors, finish DB reads, then return NDJSON frames."""
    conn = get_db()
    event_cursor = conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS newest FROM dashboard_events"
    ).fetchone()["newest"]
    frames: list[dict[str, Any]] = [{
        "type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "event_cursor": event_cursor,
        "sse_cursor": event_cursor,
        "cursor": event_cursor,
    }]
    resolutions: dict[str, dict[str, Any]] = {}
    runtime_errors: dict[str, str] = {}
    for key, query in normalized["queries"].items():
        try:
            resolutions[key] = _resolve_selector(conn, query["selector"])
        except Exception as exc:  # one runtime selector failure must not cancel siblings
            runtime_errors[key] = str(exc)

    all_ids = list(dict.fromkeys(
        experiment_id
        for resolution in resolutions.values()
        for experiment_id in resolution["ids"]
    ))
    summaries = read_experiment_summaries(all_ids, conn=conn)

    queries_by_id: dict[int, list[str]] = defaultdict(list)
    curve_requirements: dict[int, tuple[set[str], int]] = {}
    raw_ids: set[int] = set()
    for key, resolution in resolutions.items():
        query = normalized["queries"][key]
        for experiment_id in resolution["ids"]:
            if experiment_id not in summaries:
                continue
            queries_by_id[experiment_id].append(key)
            summary = summaries[experiment_id]
            if "curves" in query["projections"] and (
                normalized["known_metrics"].get(experiment_id) != summary["metrics_revision"]
            ):
                current_series, current_max = curve_requirements.get(experiment_id, (set(), 0))
                curve_requirements[experiment_id] = (
                    current_series | set(query["series"]), max(current_max, query["max_points"])
                )
            if "raw_metrics" in query["projections"]:
                raw_ids.add(experiment_id)
    curves, raw = _query_metrics(conn, curve_requirements, raw_ids)
    # No explicit BEGIN is used, so SQLite's default deferred mode does not hold
    # a read transaction here.  Every frame is materialized before the endpoint
    # constructs its StreamingResponse.

    for key, error in runtime_errors.items():
        frames.append({"type": "error", "query": key, "error": error})

    for key, resolution in resolutions.items():
        query = normalized["queries"][key]
        if query["selector"]["type"] == "discovery":
            groups = []
            for _identity, member_ids in resolution["groups"]:
                group_summaries = [summaries[experiment_id] for experiment_id in member_ids if experiment_id in summaries]
                if group_summaries:
                    groups.append(_aggregate_group(group_summaries))
            frames.append({
                "type": "group_page",
                "query": key,
                "groups": groups,
                "next_cursor": resolution["next_cursor"],
                "pagination_cursor": resolution["next_cursor"],
                "membership": {
                    str(group["id"]): group["experiment_ids"] for group in groups
                },
            })

    for experiment_id in all_ids:
        summary = summaries.get(experiment_id)
        if not summary:
            continue
        summary_changed = normalized["known_experiments"].get(experiment_id) != summary["revision"]
        has_curves = experiment_id in curves
        if summary_changed or has_curves:
            experiment = dict(summary)
            if has_curves:
                experiment["loss_curves"] = curves[experiment_id]
                experiment["loss_curve"] = curves[experiment_id].get(summary["loss_metric"], [])
                experiment["curve_max_points"] = curve_requirements[experiment_id][1]
            frames.append({
                "type": "experiment",
                "query": queries_by_id[experiment_id][0] if len(queries_by_id[experiment_id]) == 1 else None,
                "queries": queries_by_id[experiment_id],
                "experiment": experiment,
            })
        if experiment_id in raw:
            rows = raw[experiment_id]
            if not rows:
                frames.append({
                    "type": "raw_metrics", "queries": queries_by_id[experiment_id],
                    "experiment_id": experiment_id, "rows": [], "chunk": 0, "done": True,
                })
            for offset in range(0, len(rows), RAW_CHUNK_ROWS):
                frames.append({
                    "type": "raw_metrics",
                    "queries": queries_by_id[experiment_id],
                    "experiment_id": experiment_id,
                    "rows": rows[offset:offset + RAW_CHUNK_ROWS],
                    "chunk": offset // RAW_CHUNK_ROWS,
                    "done": offset + RAW_CHUNK_ROWS >= len(rows),
                })

    for key, resolution in resolutions.items():
        ids = resolution["ids"]
        frames.append({
            "type": "complete",
            "query": key,
            "experiment_ids": [experiment_id for experiment_id in ids if experiment_id in summaries],
            "membership": [experiment_id for experiment_id in ids if experiment_id in summaries],
            "missing_ids": [experiment_id for experiment_id in ids if experiment_id not in summaries],
            "revisions": {
                str(experiment_id): summaries[experiment_id]["revision"]
                for experiment_id in ids if experiment_id in summaries
            },
            "metrics_revisions": {
                str(experiment_id): summaries[experiment_id]["metrics_revision"]
                for experiment_id in ids if experiment_id in summaries
            },
            "next_cursor": resolution["next_cursor"],
            "pagination_cursor": resolution["next_cursor"],
        })
    return frames


def encode_ndjson(frames: Iterable[dict[str, Any]]) -> Iterable[bytes]:
    for frame in frames:
        yield (json.dumps(frame, separators=(",", ":"), allow_nan=False) + "\n").encode()
