"""Oracle-compatible consolidation fold operations."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..pg import consolidation as pg_consolidation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...consolidation.consolidator import _TemporalBounds


def _uuid_list(value: Any) -> list[uuid.UUID]:
    """Normalize an Oracle JSON CLOB or decoded list into UUIDs."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(value, (list, tuple)):
        return []
    out: list[uuid.UUID] = []
    for item in value:
        if isinstance(item, uuid.UUID):
            out.append(item)
            continue
        try:
            out.append(uuid.UUID(str(item)))
        except (ValueError, AttributeError, TypeError):
            continue
    return out


def _merge_source_uuids(
    existing: Any,
    additions: list[uuid.UUID],
) -> tuple[list[uuid.UUID], list[uuid.UUID]]:
    """Merge ``additions`` onto ``existing`` preserving order, returning ``(merged, new_only)``."""
    existing_uuids = _uuid_list(existing)
    existing_set = set(existing_uuids)
    merged = list(dict.fromkeys([*existing_uuids, *additions]))
    new_only = [sid for sid in merged if sid not in existing_set]
    return merged, new_only


def _utc_dt(value: datetime | None) -> datetime | None:
    """Attach explicit UTC tzinfo to naive datetimes returned by Oracle."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


async def _sync_observation_sources(
    *,
    conn,
    fq_table: Callable[[str], str],
    observation_id: uuid.UUID,
    source_ids: list[uuid.UUID],
) -> None:
    if not source_ids:
        return
    await conn.executemany(
        f"""
        INSERT INTO {fq_table("observation_sources")} (observation_id, source_id)
        VALUES ($1, $2)
        ON CONFLICT (observation_id, source_id) DO NOTHING
        """,
        [(observation_id, sid) for sid in source_ids],
    )


async def fold_sources_into_observation(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    observation_id: str,
    expected_text: str,
    merged_text: str,
    source_memory_ids: list[uuid.UUID],
    bounds: _TemporalBounds,
) -> bool:
    """Oracle-safe fold of ``source_memory_ids``, ``merged_text`` and ``bounds`` into an observation.

    Locks the target row ``FOR UPDATE`` and checks ``expected_text`` in Python rather than in
    SQL ``WHERE text = :bind`` (which raises ORA-00932 on CLOB columns), merges
    ``source_memory_ids`` in Python instead of PG ``unnest``/``array_agg``, widens dates via
    Oracle-safe ``COALESCE(LEAST/GREATEST(...))``, and appends new postings to
    ``observation_sources``.
    """
    mu = fq_table("memory_units")
    obs_uuid = uuid.UUID(observation_id)
    row = await conn.fetchrow(
        f"SELECT text, source_memory_ids FROM {mu} WHERE id = $1 AND bank_id = $2 FOR UPDATE",
        obs_uuid,
        bank_id,
    )
    if row is None or row["text"] != expected_text:
        return False

    merged_sources, new_sources = _merge_source_uuids(row["source_memory_ids"], source_memory_ids)
    updated_rows = await conn.execute_rows_affected(
        f"""
        UPDATE {mu}
        SET text = $1,
            source_memory_ids = $2,
            proof_count = $3,
            event_date = COALESCE(LEAST(event_date, COALESCE($6, event_date)), $6),
            occurred_start = COALESCE(LEAST(occurred_start, COALESCE($7, occurred_start)), $7),
            occurred_end = COALESCE(GREATEST(occurred_end, COALESCE($8, occurred_end)), $8),
            mentioned_at = COALESCE(GREATEST(mentioned_at, COALESCE($9, mentioned_at)), $9),
            updated_at = now()
        WHERE id = $4 AND bank_id = $5
        """,
        merged_text,
        merged_sources,
        len(merged_sources),
        obs_uuid,
        bank_id,
        _utc_dt(bounds.event_date),
        _utc_dt(bounds.occurred_start),
        _utc_dt(bounds.occurred_end),
        _utc_dt(bounds.mentioned_at),
    )
    if updated_rows == 0:
        return False

    await _sync_observation_sources(
        conn=conn,
        fq_table=fq_table,
        observation_id=obs_uuid,
        source_ids=new_sources,
    )
    return True


async def fold_observation_into_twin(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    observation_id: str,
    observation_text: str,
    twin_id: str,
    twin_text: str,
    merged_text: str,
) -> bool:
    """Oracle-safe fold of an observation's live sources and dates into its twin.

    Keeps the sources-before-observations lock order, locks both observation rows in ``ORDER BY id``
    order, compares CLOB ``text`` values in Python, merges sources without ``unnest``/``UPDATE FROM``,
    and appends new postings to ``observation_sources``.
    """
    mu = fq_table("memory_units")
    obs_uuid = uuid.UUID(observation_id)
    twin_uuid = uuid.UUID(twin_id)

    updated_row = await conn.fetchrow(
        f"SELECT text, source_memory_ids FROM {mu} WHERE id = $1 AND bank_id = $2",
        obs_uuid,
        bank_id,
    )
    if updated_row is None or updated_row["text"] != observation_text:
        return False

    sources = _uuid_list(updated_row["source_memory_ids"])
    if not sources:
        return False

    live = await pg_consolidation.lock_live_memory_ids(
        conn=conn,
        fq_table=fq_table,
        bank_id=bank_id,
        unit_ids=sources,
    )
    if not any(str(mid) in live for mid in sources):
        return False

    locked_rows = await conn.fetch(
        f"""
        SELECT id, text, source_memory_ids, event_date, occurred_start, occurred_end, mentioned_at
        FROM {mu}
        WHERE bank_id = $1 AND id IN ($2, $3)
        ORDER BY id
        FOR UPDATE
        """,
        bank_id,
        obs_uuid,
        twin_uuid,
    )
    rows_by_id = {str(r["id"]): r for r in locked_rows}
    obs_locked = rows_by_id.get(str(obs_uuid))
    twin_locked = rows_by_id.get(str(twin_uuid))
    if (
        obs_locked is None
        or twin_locked is None
        or obs_locked["text"] != observation_text
        or twin_locked["text"] != twin_text
    ):
        return False

    live_u_sources = [mid for mid in _uuid_list(obs_locked["source_memory_ids"]) if str(mid) in live]
    if not live_u_sources:
        return False

    merged_sources, new_sources = _merge_source_uuids(twin_locked["source_memory_ids"], live_u_sources)
    updated_rows = await conn.execute_rows_affected(
        f"""
        UPDATE {mu}
        SET text = $1,
            source_memory_ids = $2,
            proof_count = $3,
            event_date = COALESCE(LEAST(event_date, COALESCE($6, event_date)), $6),
            occurred_start = COALESCE(LEAST(occurred_start, COALESCE($7, occurred_start)), $7),
            occurred_end = COALESCE(GREATEST(occurred_end, COALESCE($8, occurred_end)), $8),
            mentioned_at = COALESCE(GREATEST(mentioned_at, COALESCE($9, mentioned_at)), $9),
            updated_at = now()
        WHERE id = $4 AND bank_id = $5
        """,
        merged_text,
        merged_sources,
        len(merged_sources),
        twin_uuid,
        bank_id,
        _utc_dt(obs_locked["event_date"]),
        _utc_dt(obs_locked["occurred_start"]),
        _utc_dt(obs_locked["occurred_end"]),
        _utc_dt(obs_locked["mentioned_at"]),
    )
    if updated_rows == 0:
        return False

    await _sync_observation_sources(
        conn=conn,
        fq_table=fq_table,
        observation_id=twin_uuid,
        source_ids=new_sources,
    )
    return True
