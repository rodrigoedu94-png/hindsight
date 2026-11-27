"""Oracle-specific Link Expansion graph retriever with SQL/PGQ support."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from ....config import get_config
from ...db.ops import LinkExpansionRows, UpdatedWindow
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


async def _use_pgq(conn, ops) -> bool:
    """Whether the SQL/PGQ path applies: an Oracle connection whose schema exposes the graph.

    OracleOps caches the capability per schema, but the first check is still a query, so it
    gets the expansion's deadline too: a probe that stalls means relational expansion.
    """
    if getattr(conn, "backend_type", None) != "oracle" or not callable(getattr(ops, "has_property_graph", None)):
        return False
    timeout = get_config().link_expansion_timeout
    try:
        return bool(await asyncio.wait_for(ops.has_property_graph(conn), timeout=timeout))
    except asyncio.TimeoutError:
        logger.warning(
            "[LinkExpansion] SQL/PGQ capability check timed out after %ss, using relational expansion", timeout
        )
        return False


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

        use_pgq = await _use_pgq(conn, ops)
        if not use_pgq:
            return await super()._expand_combined(
                conn,
                seed_ids,
                fact_type,
                budget,
                ops=ops,
                created_after=created_after,
                created_before=created_before,
            )

        config = get_config()
        mu = fq_table("memory_units")
        graph_name = fq_table("hindsight_memory_graph")
        per_entity_limit = config.link_expansion_per_entity_limit
        window = UpdatedWindow(after=created_after, before=created_before, first_param_index=4)
        entity_cte = build_pgq_entity_expansion_cte(graph_name, mu, per_entity_limit, window)
        semantic_causal_cte = build_pgq_semantic_causal_cte(
            graph_name,
            mu,
            window,
            max_causal_hops=getattr(config, "oracle_pgq_max_hops", 2),
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
        params = [seed_ids, fact_type, budget, *window.params]

        try:
            all_rows = await conn.fetch(
                full_query,
                *params,
                timeout=config.link_expansion_timeout,
            )
        except Exception as pgq_err:
            if _is_oracle_timeout_error(pgq_err):
                logger.warning(
                    "[LinkExpansion] Entity expansion timed out after %ss for fact_type=%s, "
                    "falling back to semantic+causal only",
                    config.link_expansion_timeout,
                    fact_type,
                )
                fallback_query = f"""
                    WITH {semantic_causal_cte}
                    SELECT * FROM semantic_expanded
                    UNION ALL
                    SELECT * FROM causal_expanded
                    LIMIT $3
                    """
                try:
                    all_rows = await conn.fetch(
                        fallback_query,
                        *params,
                        timeout=config.link_expansion_timeout,
                    )
                except Exception as fb_err:
                    if _is_oracle_timeout_error(fb_err):
                        logger.warning(
                            "[LinkExpansion] Oracle SQL/PGQ timeout fallback timed out after %ss "
                            "for fact_type=%s, skipping graph results",
                            config.link_expansion_timeout,
                            fact_type,
                        )
                        return LinkExpansionRows(entity=[], semantic=[], causal=[])
                    if _is_pgq_oracle_error(fb_err):
                        logger.warning(
                            "[LinkExpansion] Oracle SQL/PGQ timeout fallback failed (%r), "
                            "falling back to relational expansion",
                            fb_err,
                        )
                        return await super()._expand_combined(
                            conn,
                            seed_ids,
                            fact_type,
                            budget,
                            ops=ops,
                            created_after=created_after,
                            created_before=created_before,
                        )
                    raise
            elif _is_pgq_oracle_error(pgq_err):
                logger.warning(
                    "[LinkExpansion] Oracle SQL/PGQ query failed (%s), falling back to relational expansion",
                    pgq_err,
                )
                return await super()._expand_combined(
                    conn,
                    seed_ids,
                    fact_type,
                    budget,
                    ops=ops,
                    created_after=created_after,
                    created_before=created_before,
                )
            else:
                raise

        return LinkExpansionRows(
            entity=[r for r in all_rows if r["source"] == "entity"],
            semantic=[r for r in all_rows if r["source"] == "semantic"],
            causal=[r for r in all_rows if r["source"] == "causal"],
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
