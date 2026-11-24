"""Unit tests for LIN-273 isolated Oracle 26ai SQL/PGQ expansion (engine/memories/ora/pgq.py)."""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.db.ops import MEMORY_UNIT_COLUMNS, UpdatedWindow
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.memories.ora import link_expansion as ora_le
from hindsight_api.engine.memories.ora.link_expansion import OracleLinkExpansionRetriever
from hindsight_api.engine.memories.ora.pgq import (
    build_pgq_entity_expansion_cte,
    build_pgq_semantic_causal_cte,
    expand_observations_pgq,
    query_entity_neighborhood_pgq,
)
from hindsight_api.engine.memories.pg.link_expansion import LinkExpansionRetriever


def _arm_projections(sql: str) -> dict[str, list[str]]:
    arms: dict[str, list[str]] = {}
    for m in re.finditer(r"(\w+_expanded)\s+AS\s*\(\s*SELECT\s+(.*?)\s+FROM\b", sql, re.S | re.I):
        cols = [c.strip().split()[-1].split(".")[-1] for c in m.group(2).split(",")]
        arms[m.group(1)] = [c for c in cols if c not in ("score", "source")]
    return arms


@pytest.mark.parametrize("max_hops", [1, 2])
def test_pgq_semantic_causal_cte_excludes_seeds_and_preserves_projection(max_hops):
    window = UpdatedWindow(
        after=datetime(2026, 1, 1, tzinfo=timezone.utc),
        before=None,
        first_param_index=4,
    )
    cte = (
        build_pgq_entity_expansion_cte("hindsight_memory_graph", "mu_t", 7, window)
        + ",\n"
        + build_pgq_semantic_causal_cte("hindsight_memory_graph", "mu_t", window, max_causal_hops=max_hops)
    )
    arms = _arm_projections(cte)
    for name in ("entity_expanded", "semantic_expanded", "causal_expanded"):
        assert arms[name] == list(MEMORY_UNIT_COLUMNS), f"Unexpected projection on {name}: {arms.get(name)}"
    assert "MATCH (src IS memory_unit) -[m1 IS mentions]-> (e IS entity)" in cte
    assert "WHERE eur.rn <= 7" in cte
    # entity + semantic + 1-hop causal (+ optional 2-hop causal)
    expected_exclusions = 4 if max_hops == 2 else 3
    assert cte.count("dst.id != ALL($1::uuid[])") == expected_exclusions
    if max_hops == 2:
        assert "l1.weight * l2.weight * 0.7" in cte
    else:
        assert "l1.weight * l2.weight * 0.7" not in cte


@pytest.mark.asyncio
@pytest.mark.parametrize("max_hops", [1, 2])
async def test_expand_observations_pgq_uses_derived_from_and_guards_empty(max_hops):
    conn = MagicMock()
    conn.fetch = AsyncMock(
        return_value=[
            {"source": "entity", "id": "obs-1"},
            {"source": "semantic", "id": "obs-2"},
            {"source": "causal", "id": "obs-3"},
        ]
    )
    window = UpdatedWindow(
        after=datetime(2026, 1, 1, tzinfo=timezone.utc),
        before=None,
        first_param_index=3,
    )
    empty = await expand_observations_pgq(conn, "hindsight_memory_graph", "mu_t", [], 15, 5, window)
    assert empty.entity == [] and empty.semantic == [] and empty.causal == []
    conn.fetch.assert_not_awaited()

    rows = await expand_observations_pgq(
        conn,
        "hindsight_memory_graph",
        "mu_t",
        ["seed-obs-1"],
        15,
        5,
        window,
        max_causal_hops=max_hops,
    )
    assert len(rows.entity) == 1 and len(rows.semantic) == 1 and len(rows.causal) == 1
    sql = conn.fetch.await_args.args[0]
    assert "-[d IS derived_from]->" in sql
    assert "-[m1 IS mentions]-> (e IS entity)" in sql
    assert "obs.id != ALL($1::uuid[])" in sql
    assert "WHERE eur.rn <= 5" in sql
    arms = _arm_projections(sql)
    for name in ("observation_entity_expanded", "semantic_expanded", "causal_expanded"):
        assert arms[name] == list(MEMORY_UNIT_COLUMNS)
    if max_hops == 2:
        assert "l1.weight * l2.weight * 0.7" in sql
    else:
        assert "l1.weight * l2.weight * 0.7" not in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("max_hops", [1, 2])
async def test_query_entity_neighborhood_pgq_emits_explicit_hop_distances(max_hops):
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])
    e1 = uuid.uuid4()
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 1, tzinfo=timezone.utc)

    await query_entity_neighborhood_pgq(
        conn,
        mu_table='"ADMIN".memory_units',
        graph_name='"ADMIN".hindsight_memory_graph',
        bank_id="ecossistema-ia",
        entity_ids=[e1],
        max_hops=max_hops,
        occurred_after=t0,
        occurred_before=t1,
        limit=25,
    )
    sql = conn.fetch.await_args.args[0]
    assert "0 AS hop_distance" in sql
    assert "1 AS hop_distance" in sql
    assert "-[co1 IS cooccurs_with]-" in sql
    assert "e2.id != e1.id" in sql
    if max_hops == 2:
        assert "2 AS hop_distance" in sql
        assert "-[co2 IS cooccurs_with]-" in sql
    else:
        assert "2 AS hop_distance" not in sql
    assert "LIMIT $5" in sql


@pytest.mark.asyncio
async def test_oracle_link_expansion_retriever_uses_pgq_and_falls_back(monkeypatch):
    monkeypatch.setattr(ora_le, "fq_table", lambda t: t)
    ops = OracleOps()
    monkeypatch.setattr(ops, "has_property_graph", AsyncMock(return_value=True))

    conn = MagicMock()
    conn.backend_type = "oracle"
    conn.fetch = AsyncMock(return_value=[{"source": "entity", "id": "obs-10"}]_)

    retriever = OracleLinkExpansionRetriever()
    res_obs = await retriever._expand_observations(conn, ["seed-obs"], 20, ops=ops)
    assert len(res_obs.entity) == 1
    assert "-[d IS derived_from]->" in conn.fetch.await_args.args[0]

    conn.fetch = AsyncMock(side_effect=RuntimeError("ORA-49022: graph unavailable"))
    fallback_rows = MagicMock()
    monkeypatch.setattr(LinkExpansionRetriever, "_expand_observations", AsyncMock(return_value=fallback_rows))
    assert await retriever._expand_observations(conn, ["seed-obs"], 20, ops=ops) is fallback_rows

    conn.fetch = AsyncMock(side_effect=asyncio.TimeoutError())
    assert await retriever._expand_observations(conn, ["seed-obs"], 20, ops=ops) is fallback_rows
