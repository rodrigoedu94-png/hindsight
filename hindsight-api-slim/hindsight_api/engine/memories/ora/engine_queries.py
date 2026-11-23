"""Oracle-specific SQL helpers called from ``MemoryEngine``."""

from __future__ import annotations

from collections.abc import Callable


async def trim_mental_model_history(
    *,
    conn,
    fq_table: Callable[[str], str],
    mental_model_id: str,
    bank_id: str,
    kind_predicate: str,
    max_entries: int,
) -> None:
    """Trim ``mental_model_history`` using ``ROW_NUMBER()`` over ``SYS_EXTRACT_UTC(changed_at)``."""
    await conn.execute(
        f"""
        DELETE FROM {fq_table("mental_model_history")}
        WHERE id IN (
            SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (ORDER BY SYS_EXTRACT_UTC(changed_at) DESC, id DESC) AS rn
                FROM {fq_table("mental_model_history")}
                WHERE mental_model_id = $1 AND bank_id = $2
                  AND {kind_predicate}
            )
            WHERE rn > $3
        )
        """,
        mental_model_id,
        bank_id,
        max_entries,
    )


def async_operations_list_clauses(*, single_condition: bool) -> tuple[str, str]:
    """Return ``(optimizer_hint, order_by_expr)`` for listing ``async_operations`` on Oracle."""
    hint = "/*+ INDEX(ao idx_async_operations_bank_created_desc) */ " if single_condition else ""
    return hint, "SYS_EXTRACT_UTC(ao.created_at) DESC"
