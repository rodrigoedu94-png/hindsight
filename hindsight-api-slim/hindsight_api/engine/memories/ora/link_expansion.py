"""Oracle-specific Link Expansion graph retriever with SQL/PGQ support."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from ....config import get_config
from ...db._oracle_helpers import is_oracle_timeout
from ...db.ops import LinkExpansionRows, UpdatedWindow
from ...db.oracle import OracleConnection
from ...schema import fq_store_table as fq_table
from ..pg.link_expansion import LinkExpansionRetriever
from .pgq import (
    build_pgq_entity_expansion_cte,
    build_pgq_semantic_causal_cte,
    expand_observations_pgq,
)

logger = logging.getLogger(__name__)

_PGQ_ORACLE_ERROR_CODES = ("ORA-49022", "ORA-42404", "ORA-40998", "ORA-00942")


def _is_pgq_oracle_error(exc: BaseException) -> bool:
    """Return whether ``exc`` is a SQL/PGQ availability/syntax error safe to fall back on."""
    msg = str(exc)
    return any(code in msg for code in _PGQ_ORACLE_ERROR_CODES)


def _split_expansion_rows(all_rows: list) -> LinkExpansionRows:
    grouped: dict[str, list] = {"entity": [], "semantic": [], "causal": []}
    for row in all_rows:
        grouped.setdefault(row["source"], []).append(row)
    return LinkExpansionRows(entity=grouped["entity"], semantic=grouped["semantic"], causal=grouped["causal"])


def _classify_pgq_error(err: BaseException, context: str, timeout: float) -> str:
    """Log SQL/PGQ timeout or relational fallback reason; return ``'timeout'``/``'relational'`` or re-raise."""
    if is_oracle_timeout(err):
        logger.warning("[LinkExpansion] %s timed out after %ss", context, timeout)
        return "timeout"
    if _is_pgq_oracle_error(err):
        logger.warning("[LinkExpansion] %s failed (%r), falling back to relational expansion", context, err)
        return "relational"
    raise err


async def _use_pgq(conn, ops) -> bool:
    """Whether the SQL/PGQ path applies: an Oracle connection whose schema exposes the graph.

    For real ``OracleConnection`` instances, let ``ops.has_property_graph`` pass ``timeout``
    to ``conn.fetchval`` (driver ``call_timeout``) instead of cancelling the probe task at the
    same deadline and invalidating the connection. Stubs without ``call_timeout`` keep the
    ``asyncio.wait_for`` bound.
    """
    if getattr(conn, "backend_type", None) != "oracle" or not callable(getattr(ops, "has_property_graph", None)):
        return False
    timeout = get_config().link_expansion_timeout
    try:
        if isinstance(conn, OracleConnection):
            return bool(await ops.has_property_graph(conn))
        return bool(await asyncio.wait_for(ops.has_property_graph(conn), timeout=timeout))
    except Exception as exc:
        if is_oracle_timeout(exc):
            logger.warning(
                "[LinkExpansion] SQL/PGQ capability check timed out after %ss, using relational expansion", timeout
            )
            return False
        raise


async def _fetch_pgq_fallback_rows(conn, semantic_causal_cte: str, params: list, timeout: float) -> list | None:
    """Run semantic+causal PGQ query after timeout; return ``[]`` on timeout or ``None`` for relational."""
    fallback_query = f"""
        WITH {semantic_causal_cte}
        SELECT * FROM semantic_expanded
        UNION ALL
        SELECT * FROM causal_expanded
        LIMIT $3
        """
    try:
        return list(await conn.fetch(fallback_query, *params, timeout=timeout))
    except Exception as fb_err:
        kind = _classify_pgq_error(fb_err, "Oracle SQL/PGQ timeout fallback", timeout)
        return [] if kind == "timeout" else None


def _build_pgq_combined_queries(
    created_after: datetime | None, created_before: datetime | None
) -> tuple[str, str, list]:
    """Build combined SQL/PGQ query, semantic+causal fallback CTE, and trailing window params."""
    config = get_config()
    mu = fq_table("memory_units")
    graph_name = fq_table("hindsight_memory_graph")
    window = UpdatedWindow(after=created_after, before=created_before, first_param_index=4)
    entity_cte = build_pgq_entity_expansion_cte(
        graph_name, mu, config.link_expansion_per_entity_limit, window
    )
    semantic_causal_cte = build_pgq_semantic_causal_cte(
        graph_name, mu, window, max_causal_hops=getattr(config, "oracle_pgq_max_hops", 2)
    )
    full_query = f"""
        WITH {entity_cte},
        {semantic_causal_cte}
        SELECT * FROM entity_expanded
        UNION ALL
        SELECT * FROM semantic_expanded
        UNION ALL
        SELECT * FROM causal_expanded
        """
    return full_query, semantic_causal_cte, window.params


class OracleLinkExpansionRetriever(LinkExpansionRetriever):
    """LinkExpansionRetriever that uses Oracle 26ai SQL/PGQ when available, with relational fallback."""

    async def _expand_combined(
        self,
        conn,
        seed_ids: list,
        fact_type: str,
        budget: int,
        *,
        ops,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> LinkExpansionRows:
        if not seed_ids or budget <= 0:
            return LinkExpansionRows(entity=[], semantic=[], causal=[])

        if await _use_pgq(conn, ops):
            timeout = get_config().link_expansion_timeout
            full_query, semantic_causal_cte, window_params = _build_pgq_combined_queries(created_after, created_before)
            params = [seed_ids, fact_type, budget, *window_params]
            try:
                return _split_expansion_rows(await conn.fetch(full_query, *params, timeout=timeout))
            except Exception as pgq_err:
                kind = _classify_pgq_error(pgq_err, f"Oracle SQL/PGQ entity expansion (fact_type={fact_type})", timeout)
                if kind == "timeout":
                    rows = await _fetch_pgq_fallback_rows(conn, semantic_causal_cte, params, timeout)
                    if rows is not None:
                        return _split_expansion_rows(rows)

        return await super()._expand_combined(
            conn,
            seed_ids,
            fact_type,
            budget,
            ops=ops,
            created_after=created_after,
            created_before=created_before,
        )

    async def _expand_observations(
        self,
        conn,
        seed_ids: list,
        budget: int,
        *,
        ops,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> LinkExpansionRows:
        if not seed_ids or budget <= 0:
            return LinkExpansionRows(entity=[], semantic=[], causal=[])

        if await _use_pgq(conn, ops):
            config = get_config()
            window = UpdatedWindow(after=created_after, before=created_before, first_param_index=3)
            try:
                return await expand_observations_pgq(
                    conn,
                    fq_table("hindsight_memory_graph"),
                    fq_table("memory_units"),
                    seed_ids,
                    budget,
                    config.link_expansion_per_entity_limit,
                    window,
                    max_causal_hops=getattr(config, "oracle_pgq_max_hops", 2),
                    os_table=fq_table("observation_sources"),
                    timeout=config.link_expansion_timeout,
                )
            except Exception as pgq_err:
                if (
                    _classify_pgq_error(
                        pgq_err, "Oracle SQL/PGQ observation expansion", config.link_expansion_timeout
                    )
                    == "timeout"
                ):
                    return LinkExpansionRows(entity=[], semantic=[], causal=[])

        return await super()._expand_observations(
            conn,
            seed_ids,
            budget,
            ops=ops,
            created_after=created_after,
            created_before=created_before,
        )
