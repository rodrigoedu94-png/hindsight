"""Oracle-specific watermark and range reads over ``memory_units``."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ...db.oracle import OracleLiteralStr
from ...search.tags import TagsMatch, build_tag_groups_where_clause, build_tags_where_clause
from ..base import StoredMemory
from ..pg.reads import _MEMORY_COLUMNS, _ScopeWhere, _stored_from_row


def _refresh_scope_where(
    bank_id: str,
    until: datetime,
    since: datetime | None,
    fact_types: list[str] | None,
    tags: list[str] | None,
    tags_match: TagsMatch,
    tag_groups: list | None,
) -> _ScopeWhere:
    """A refresh's scope within ``(since, until]`` as an Oracle UTC-normalized WHERE clause."""
    until_bound = until if until.tzinfo is not None else until.replace(tzinfo=timezone.utc)
    params: list[Any] = [OracleLiteralStr(bank_id), until_bound]
    where = [
        "bank_id = $1",
        "SYS_EXTRACT_UTC(updated_at) <= SYS_EXTRACT_UTC(CAST($2 AS TIMESTAMP WITH TIME ZONE))",
    ]
    if since is not None:
        since_bound = since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
        params.append(since_bound)
        idx = len(params)
        where.append(f"SYS_EXTRACT_UTC(updated_at) > SYS_EXTRACT_UTC(CAST(${idx} AS TIMESTAMP WITH TIME ZONE))")

    built = build_tags_where_clause(tags, param_offset=len(params) + 1, match=tags_match)
    if built.sql:
        where.append(built.sql.removeprefix("AND "))
        params.extend(built.params)

    built = build_tag_groups_where_clause(tag_groups, param_offset=len(params) + 1)
    if built.sql:
        where.append(built.sql.removeprefix("AND "))
        params.extend(built.params)

    if fact_types:
        params.append(list(fact_types))
        where.append(f"fact_type = ANY(${len(params)}::text[])")
    return _ScopeWhere(sql=" AND ".join(where), params=params)


async def newest_memory_updated_at(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    until: datetime,
    since: datetime | None = None,
    fact_types: list[str] | None = None,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list | None = None,
) -> datetime | None:
    """Return the newest ``updated_at`` in scope using ``idx_mu_bank_updated_at``."""
    scope = _refresh_scope_where(bank_id, until, since, fact_types, tags, tags_match, tag_groups)
    return await conn.fetchval(
        f"SELECT /*+ INDEX(m idx_mu_bank_updated_at) */ m.updated_at FROM {fq_table('memory_units')} m "
        f"WHERE {scope.sql} ORDER BY SYS_EXTRACT_UTC(m.updated_at) DESC FETCH FIRST 1 ROW ONLY",
        *scope.params,
    )


async def memories_updated_between(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    until: datetime,
    since: datetime,
    limit: int,
    fact_types: list[str] | None = None,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list | None = None,
) -> list[StoredMemory]:
    """Return memories in ``(since, until]`` ordered by ``SYS_EXTRACT_UTC(updated_at), id``."""
    scope = _refresh_scope_where(bank_id, until, since, fact_types, tags, tags_match, tag_groups)
    rows = await conn.fetch(
        f"SELECT {_MEMORY_COLUMNS} FROM {fq_table('memory_units')} WHERE {scope.sql} "
        f"ORDER BY SYS_EXTRACT_UTC(updated_at), id LIMIT ${len(scope.params) + 1}",
        *scope.params,
        limit,
    )
    return [_stored_from_row(row) for row in rows]


async def latest_memory_write_at(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
) -> datetime | None:
    """Return the newest ``updated_at`` for ``bank_id`` via backward index probe."""
    return await conn.fetchval(
        f"SELECT /*+ INDEX(m idx_mu_bank_updated_at) */ m.updated_at FROM {fq_table('memory_units')} m "
        "WHERE m.bank_id = $1 ORDER BY SYS_EXTRACT_UTC(m.updated_at) DESC FETCH FIRST 1 ROW ONLY",
        OracleLiteralStr(bank_id),
    )
