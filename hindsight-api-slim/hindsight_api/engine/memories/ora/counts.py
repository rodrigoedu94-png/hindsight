"""Oracle-compatible count queries for memory scopes."""

import json
from collections import Counter
from collections.abc import Callable
from typing import Any

from ...search.tags import TagGroup, build_tag_groups_where_clause


def _tags_list(value: Any) -> list[str]:
    """Normalize an Oracle JSON CLOB (or a decoded JSON value) into sorted tags."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            value = []
    if not isinstance(value, (list, tuple)):
        return []
    return sorted(tag for tag in value if isinstance(tag, str))


async def observation_scope_counts(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    limit: int = 100,
    offset: int = 0,
    tag_groups: list[TagGroup] | None = None,
) -> dict[str, Any]:
    """Return a sorted, paged histogram of observation tag scopes.

    Oracle stores ``tags`` as a JSON CLOB, which cannot be grouped directly.
    Compact values are hash-aggregated through a bounded VARCHAR2 projection.
    Longer values are fetched individually so no truncated key can merge distinct
    scopes; this keeps the common path compact while preserving exactness.
    """
    if limit < 0 or offset < 0:
        raise ValueError("limit and offset must be non-negative")

    groups = build_tag_groups_where_clause(tag_groups, 2)
    table = fq_table("memory_units")
    # 900 characters fit within 3600 bytes even with a 4-byte UTF-8 charset.
    rows = await conn.fetch(
        f"""
        WITH filtered AS (
            SELECT tags
            FROM {table}
            WHERE bank_id = $1 AND fact_type = 'observation' {groups.sql}
        ),
        compact_groups AS (
            SELECT DBMS_LOB.SUBSTR(tags, 900, 1) AS short_tags, COUNT(*) AS cnt
            FROM filtered
            WHERE tags IS NULL OR DBMS_LOB.GETLENGTH(tags) <= 900
            GROUP BY DBMS_LOB.SUBSTR(tags, 900, 1)
        ),
        raw_counts AS (
            SELECT TO_CLOB(short_tags) AS tags, cnt
            FROM compact_groups
            UNION ALL
            SELECT tags AS tags, 1 AS cnt
            FROM filtered
            WHERE DBMS_LOB.GETLENGTH(tags) > 900
        )
        SELECT tags, cnt FROM raw_counts
        """,
        bank_id,
        *groups.params,
    )

    counts: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        counts[tuple(_tags_list(row["tags"]))] += int(row["cnt"])

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    page = ordered[offset : offset + limit]
    return {
        "scopes": [{"tags": list(tags), "count": count} for tags, count in page],
        "total": len(ordered),
        "limit": limit,
        "offset": offset,
    }
