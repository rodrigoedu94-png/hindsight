"""Oracle-specific Link Expansion graph retriever with SQL/PGQ support."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from ....config import get_config
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
_ORACLE_TIMEOUT_ERROR_CODES = ("DPY-4024", "ORA-03156")


def _is_oracle_timeout_error(exc: BaseException) -> bool:
    """Return whether ``exc`` is an asyncio or Oracle call-timeout error."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    msg = str(exc)
    return any(code in msg for code in _ORACLE_TIMEOUT_ERROR_CODES)


def _is_pgq_oracle_error(exc: BaseException) -> bool:
    """Return whether ``exc`` is a SQL/PGQ availability/syntax error safe to fall back on."""
    msg = str(exc)
    return any(code in msg for code in _PGQ_ORACLE_ERROR_CODES)


def _split_expansion_rows(all_rows: list) -> LinkExpansionRows:
    return LinkExpansionRows(
        entity=[r for r in all_rows if r["source"] == "entity"],
        semantic=[r for r in all_rows if r["source"] == "semantic"],
        causal=[r for r in all_rows if r["source"] == "causal"],
    )


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
        if _is_oracle_timeout_error(exc):
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
        if _is_oracle_timeout_error(fb_err):
            logger.warning("[LinkExpansion] Oracle SQL/PGQ timeout fallback timed out after %ss, skipping graph results", timeout)
            return []
        if _is_pgq_oracle_error(fb_err):
            logger.warning("[LinkExpansion] Oracle SQL/PGQ timeout fallback failed (%r), falling back to relational expansion", fb_err)
            return None
        raise


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


async def _handle_pgq_combined_error(
    pgq_err: Exception,
    fact_type: str,
    fallback_ctx: tuple[Any, str, list, float],
) -> list | None:
    """Resolve a SQL/PGQ error into fallback rows, or return None when relational fallback should run."""
    conn, semantic_causal_cte, params, timeout = fallback_ctx
    if _is_oracle_timeout_error(pgq_err):
        logger.warning(
            "[LinkExpansion] Entity expansion timed out after %ss for fact_type=%s, falling back to semantic+causal only",
            timeout,
            fact_type,
        )
        return await _fetch_pgq_fallback_rows(conn, semantic_causal_cte, params, timeout)
    if _is_pgq_oracle_error(pgq_err):
        logger.warning("[LinkExpansion] Oracle SQL/PGQ query failed (%s), falling back to relational expansion", pgq_err)
        return None
    raise pgq_err


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
                rows = await _handle_pgq_combined_error(
                    pgq_err, fact_type, (conn, semantic_causal_cte, params, timeout)
                )
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

        use_pgq = await _use_pgq(conn, ops)
        if not use_pgq:
            return await super()._expand_observations(
                conn,
                seed_ids,
                budget,
                ops=ops,
                created_after=created_after,
                created_before=created_before,
            )

        config = get_config()
        mu = fq_table("memory_units")
        graph_name = fq_table("hindsight_memory_graph")
        per_entity_limit = config.link_expansion_per_entity_limit
        window = UpdatedWindow(after=created_after, before=created_before, first_param_index=3)

        try:
            return await expand_observations_pgq(
                conn,
                graph_name,
                mu,
                seed_ids,
                budget,
                per_entity_limit,
                window,
                max_causal_hops=getattr(config, "oracle_pgq_max_hops", 2),
                os_table=fq_table("observation_sources"),
                timeout=config.link_expansion_timeout,
            )
        except Exception as pgq_err:
            if _is_oracle_timeout_error(pgq_err):
                # The deadline is spent: like the relational path on timeout, the graph arm yields
                # nothing and recall continues on its other strategies. Running the relational
                # query now would let observation expansion take twice link_expansion_timeout.
                logger.warning(
                    "[LinkExpansion] Oracle SQL/PGQ observation expansion timed out after %ss, "
                    "skipping graph results for fact_type=observation",
                    config.link_expansion_timeout,
                )
                return LinkExpansionRows(entity=[], semantic=[], causal=[])
            if _is_pgq_oracle_error(pgq_err):
                logger.warning(
                    "[LinkExpansion] Oracle SQL/PGQ observation expansion failed (%s), "
                    "falling back to relational expansion",
                    pgq_err,
                )
                return await super()._expand_observations(
                    conn,
                    seed_ids,
                    budget,
                    ops=ops,
                    created_after=created_after,
                    created_before=created_before,
                )
            raise
