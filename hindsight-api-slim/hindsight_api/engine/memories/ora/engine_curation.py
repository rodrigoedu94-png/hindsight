"""Oracle-specific bank purge ordering."""

from __future__ import annotations

from collections.abc import Callable

from ...db.oracle import OracleLiteralStr


async def purge_bank_rows(*, conn, fq_table: Callable[[str], str], bank_id: str) -> list[str]:
    """Read legacy file keys, lock ``memory_links`` in canonical order, and purge bank rows on Oracle."""
    lit_bank_id = OracleLiteralStr(bank_id)
    legacy_files = [
        row["storage_key"]
        for row in await conn.fetch(
            f"SELECT storage_key FROM {fq_table('attachments')} "
            f"WHERE bank_id = $1 AND storage_key NOT LIKE 'tenants/%' "
            f"UNION ALL SELECT file_storage_key FROM {fq_table('documents')} "
            f"WHERE bank_id = $1 AND file_storage_key IS NOT NULL "
            f"AND file_storage_key NOT LIKE 'tenants/%'",
            lit_bank_id,
        )
    ]

    await conn.fetch(
        f"SELECT ROWID FROM {fq_table('memory_links')} "
        f"WHERE bank_id = $1 ORDER BY from_unit_id, to_unit_id, link_type, entity_id FOR UPDATE",
        lit_bank_id,
    )
    await conn.execute(f"DELETE FROM {fq_table('memory_links')} WHERE bank_id = $1", lit_bank_id)
    await conn.execute(f"DELETE FROM {fq_table('documents')} WHERE bank_id = $1", lit_bank_id)
    await conn.execute(f"DELETE FROM {fq_table('attachments')} WHERE bank_id = $1", lit_bank_id)
    await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", lit_bank_id)
    await conn.execute(f"DELETE FROM {fq_table('observation_history')} WHERE bank_id = $1", lit_bank_id)
    await conn.execute(f"DELETE FROM {fq_table('invalidated_memory_units')} WHERE bank_id = $1", lit_bank_id)
    await conn.execute(f"DELETE FROM {fq_table('entities')} WHERE bank_id = $1", lit_bank_id)
    await conn.execute(f"DELETE FROM {fq_table('graph_maintenance_queue')} WHERE bank_id = $1", lit_bank_id)
    await conn.execute(f"DELETE FROM {fq_table('entity_maintenance_queue')} WHERE bank_id = $1", lit_bank_id)
    return legacy_files
