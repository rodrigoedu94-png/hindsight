"""Oracle-specific bank listing and watermark queries."""

from __future__ import annotations

from collections.abc import Callable


async def list_bank_rows(*, conn, fq_table: Callable[[str], str], where_clause: str, params: list[str]) -> list:
    """Every matching bank with its document and memory write watermarks, by bank_id."""
    banks_table = fq_table("banks")
    docs_table = fq_table("documents")
    mu_table = fq_table("memory_units")
    last_fact_expr = (
        f"(SELECT /*+ INDEX(m idx_mu_bank_updated_at) */ m.updated_at FROM {mu_table} m "
        f"WHERE m.bank_id = b.bank_id ORDER BY SYS_EXTRACT_UTC(m.updated_at) DESC FETCH FIRST 1 ROW ONLY)"
    )
    return await conn.fetch(
        f"""
            SELECT
                b.bank_id, b.name, b.disposition, b.mission,
                b.created_at, b.updated_at,
                d.last_document_at,
                d.last_document_write_at,
                {last_fact_expr} AS last_fact_at
            FROM {banks_table} b
            LEFT JOIN (
                SELECT bank_id,
                       MAX(created_at) AS last_document_at,
                       MAX(updated_at) AS last_document_write_at
                FROM {docs_table}
                GROUP BY bank_id
            ) d ON d.bank_id = b.bank_id
            {where_clause}
            ORDER BY b.bank_id
            """,
        *params,
    )


async def bank_page_rows(*, conn, fq_table: Callable[[str], str], bank_ids: list[str], sql_owned: list[str]) -> list:
    """The named banks' rows with their watermarks; `memory_units` is probed only for ``sql_owned``."""
    banks_table = fq_table("banks")
    docs_table = fq_table("documents")
    if sql_owned:
        mu_table = fq_table("memory_units")
        fact_select = (
            f"CASE WHEN b.bank_id = ANY($2::text[]) THEN "
            f"(SELECT /*+ INDEX(m idx_mu_bank_updated_at) */ m.updated_at FROM {mu_table} m "
            f"WHERE m.bank_id = b.bank_id ORDER BY SYS_EXTRACT_UTC(m.updated_at) DESC FETCH FIRST 1 ROW ONLY) "
            f"END AS last_fact_at"
        )
        params: tuple = (bank_ids, sql_owned)
    else:
        fact_select = "NULL::timestamptz AS last_fact_at"
        params = (bank_ids,)
    return await conn.fetch(
        f"""
            SELECT b.bank_id, b.name, b.disposition, b.mission, b.created_at, b.updated_at,
                   d.last_document_at, d.last_document_write_at,
                   {fact_select}
            FROM {banks_table} b
            LEFT JOIN (
                SELECT bank_id, MAX(created_at) AS last_document_at, MAX(updated_at) AS last_document_write_at
                FROM {docs_table}
                WHERE bank_id = ANY($1::text[])
                GROUP BY bank_id
            ) d ON d.bank_id = b.bank_id
            WHERE b.bank_id = ANY($1::text[])
            """,
        *params,
    )
