"""Oracle-specific bank purge ordering."""

from __future__ import annotations

from collections.abc import Callable

from ..pg import engine_curation as pg_engine_curation


async def purge_bank_rows(*, conn, fq_table: Callable[[str], str], bank_id: str) -> list[str]:
    """Lock ``memory_links`` in canonical endpoint order on Oracle before delegating purge."""
    await conn.fetch(
        f"SELECT ROWID FROM {fq_table('memory_links')} "
        f"WHERE bank_id = $1 ORDER BY from_unit_id, to_unit_id, link_type, entity_id FOR UPDATE",
        bank_id,
    )
    return await pg_engine_curation.purge_bank_rows(conn=conn, fq_table=fq_table, bank_id=bank_id)
