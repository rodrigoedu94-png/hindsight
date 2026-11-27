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
    # entity + semantic (outgoing, incoming) + 1-hop causal (+ optional 2-hop causal)
    expected_exclusions = 5 if max_hops == 2 else 4
    assert cte.count("dst.id != ALL($1::uuid[])") == expected_exclusions
    # Semantic links are matched one direction per GRAPH_TABLE, never undirected.
    assert "-[l IS memory_link]- (" not in cte
    assert "MATCH (src IS memory_unit) -[l IS memory_link]-> (dst IS memory_unit)" in cte
    assert "MATCH (dst IS memory_unit) -[l IS memory_link]-> (src IS memory_unit)" in cte
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
    # The last hop back to observations reads observation_sources by source_id.
    assert "JOIN observation_sources os ON os.source_id = cs.source_id" in sql
    assert "os.observation_id != ALL($1::uuid[])" in sql
    assert "-[l IS memory_link]- (" not in sql
    assert "MATCH (dst IS memory_unit) -[l IS memory_link]-> (src IS memory_unit)" in sql
    assert "WHERE eur.rn <= 5" in sql
    arms = _arm_projections(sql)
    for name in ("observation_entity_expanded", "semantic_expanded", "causal_expanded"):
        assert arms[name] == list(MEMORY_UNIT_COLUMNS)
    if max_hops == 2:
        assert "l1.weight * l2.weight * 0.7" in sql
    else:
        assert "l1.weight * l2.weight * 0.7" not in sql


@pytest.mark.asyncio
async def test_query_entity_neighborhood_pgq_zero_hops_keeps_only_direct_mentions():
    from hindsight_api.engine.db.oracle import OracleLiteralStr

    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])
    uuid_bank = "11111111-2222-3333-4444-555555555555"
    await query_entity_neighborhood_pgq(
        conn, mu_table="mu_t", graph_name="g", bank_id=uuid_bank, entity_ids=[uuid.uuid4()], max_hops=0
    )
    sql = conn.fetch.await_args.args[0]
    assert "0 AS hop_distance" in sql
    assert "cooccurs_with" not in sql and "1 AS hop_distance" not in sql
    # A UUID-shaped bank id stays a VARCHAR2 string bind.
    assert isinstance(conn.fetch.await_args.args[2], OracleLiteralStr)


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
    conn.fetch = AsyncMock(return_value=[{"source": "entity", "id": "obs-10"}])

    retriever = OracleLinkExpansionRetriever()
    res_obs = await retriever._expand_observations(conn, ["seed-obs"], 20, ops=ops)
    assert len(res_obs.entity) == 1
    assert "-[d IS derived_from]->" in conn.fetch.await_args.args[0]

    # A SQL/PGQ error falls back to the relational expansion.
    conn.fetch = AsyncMock(side_effect=RuntimeError("ORA-49022: graph unavailable"))
    fallback_rows = MagicMock()
    relational = AsyncMock(return_value=fallback_rows)
    monkeypatch.setattr(LinkExpansionRetriever, "_expand_observations", relational)
    assert await retriever._expand_observations(conn, ["seed-obs"], 20, ops=ops) is fallback_rows

    # A SQL/PGQ timeout has spent the deadline: no graph rows, and no second (relational) query.
    relational.reset_mock()
    conn.fetch = AsyncMock(side_effect=asyncio.TimeoutError())
    timed_out = await retriever._expand_observations(conn, ["seed-obs"], 20, ops=ops)
    assert timed_out.entity == [] and timed_out.semantic == [] and timed_out.causal == []
    relational.assert_not_awaited()


@pytest.mark.asyncio
async def test_oracle_link_expansion_retriever_bounds_the_capability_probe(monkeypatch):
    """A stalled has_property_graph() probe is cut at the expansion deadline and means relational."""
    monkeypatch.setattr(ora_le, "fq_table", lambda t: t)
    monkeypatch.setattr(ora_le, "get_config", lambda: MagicMock(link_expansion_timeout=0.05))
    ops = OracleOps()

    async def stalled(_conn):
        await asyncio.sleep(5)
        return True

    monkeypatch.setattr(ops, "has_property_graph", stalled)
    conn = MagicMock()
    conn.backend_type = "oracle"
    conn.fetch = AsyncMock()
    observations, combined = MagicMock(), MagicMock()
    monkeypatch.setattr(LinkExpansionRetriever, "_expand_observations", AsyncMock(return_value=observations))
    monkeypatch.setattr(LinkExpansionRetriever, "_expand_combined", AsyncMock(return_value=combined))

    retriever = OracleLinkExpansionRetriever()
    assert await asyncio.wait_for(retriever._expand_observations(conn, ["s"], 20, ops=ops), 1) is observations
    assert await asyncio.wait_for(retriever._expand_combined(conn, ["s"], "world", 20, ops=ops), 1) is combined
    conn.fetch.assert_not_awaited()


@pytest.mark.parametrize("cap", [0, -3])
def test_pgq_entity_caps_of_zero_or_less_keep_no_units(cap):
    """Like the relational CTEs, a non-positive per-entity cap is not raised to 1."""
    window = UpdatedWindow(after=None, before=None, first_param_index=4)
    assert f"WHERE eur.rn <= {cap}" in build_pgq_entity_expansion_cte("g", "mu_t", cap, window)


@pytest.mark.asyncio
async def test_oracle_link_expansion_retriever_empty_seeds_and_non_positive_budget(monkeypatch):
    ops = OracleOps()
    ops.has_property_graph = AsyncMock(return_value=True)  # type: ignore[method-assign]
    conn = MagicMock()
    conn.backend_type = "oracle"
    conn.fetch = AsyncMock()
    retriever = OracleLinkExpansionRetriever()

    for args in (([], 10), (["seed-1"], 0), (["seed-1"], -1)):
        res_c = await retriever._expand_combined(conn, args[0], "world", args[1], ops=ops)
        res_o = await retriever._expand_observations(conn, args[0], args[1], ops=ops)
        assert res_c.entity == [] and res_c.semantic == [] and res_c.causal == []
        assert res_o.entity == [] and res_o.semantic == [] and res_o.causal == []
    ops.has_property_graph.assert_not_awaited()
    conn.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_oracle_link_expansion_retriever_narrows_errors_and_avoids_timeout_cascade(monkeypatch):
    monkeypatch.setattr(ora_le, "fq_table", lambda t: t)
    monkeypatch.setattr(ora_le, "get_config", lambda: MagicMock(link_expansion_timeout=2.5, link_expansion_per_entity_limit=5, oracle_pgq_max_hops=2))
    ops = OracleOps()
    monkeypatch.setattr(ops, "has_property_graph", AsyncMock(return_value=True))

    conn = MagicMock()
    conn.backend_type = "oracle"
    retriever = OracleLinkExpansionRetriever()

    # Passes timeout directly to conn.fetch instead of cancelling via asyncio.wait_for.
    conn.fetch = AsyncMock(return_value=[{"source": "semantic", "id": "u1"}])
    await retriever._expand_combined(conn, ["seed-1"], "world", 10, ops=ops)
    assert conn.fetch.await_args.kwargs["timeout"] == 2.5

    # Timeout errors (DPY-4024 / ORA-03156 / TimeoutError) trigger bounded fallback, and a second
    # timeout returns empty without cascading to relational super()._expand_combined.
    relational_combined = AsyncMock()
    monkeypatch.setattr(LinkExpansionRetriever, "_expand_combined", relational_combined)
    conn.fetch = AsyncMock(side_effect=[RuntimeError("DPY-4024: call timeout exceeded"), RuntimeError("ORA-03156: OCI call timed out")])
    res = await retriever._expand_combined(conn, ["seed-1"], "world", 10, ops=ops)
    assert res.entity == [] and res.semantic == [] and res.causal == []
    assert conn.fetch.await_count == 2
    assert all(call.kwargs["timeout"] == 2.5 for call in conn.fetch.await_args_list)
    relational_combined.assert_not_awaited()

    # Narrowed PGQ errors fall back to relational; arbitrary ORA-01002/ORA-03113 disconnects propagate.
    for code in ("ORA-49022", "ORA-42404", "ORA-40998", "ORA-00942"):
        relational_combined.reset_mock()
        conn.fetch = AsyncMock(side_effect=RuntimeError(f"{code}: pgq error"))
        await retriever._expand_combined(conn, ["seed-1"], "world", 10, ops=ops)
        relational_combined.assert_awaited_once()

    for fatal in (RuntimeError("ORA-01002: fetch out of sequence"), RuntimeError("ORA-03113: end-of-file on communication channel")):
        conn.fetch = AsyncMock(side_effect=fatal)
        with pytest.raises(RuntimeError):
            await retriever._expand_combined(conn, ["seed-1"], "world", 10, ops=ops)
        with pytest.raises(RuntimeError):
            await retriever._expand_observations(conn, ["seed-1"], 10, ops=ops)


# ---------------------------------------------------------------------------
# Oracle 26ai integration: the GRAPH_TABLE arms run and match the relational CTEs
# ---------------------------------------------------------------------------


class _RecordingConnection:
    """Delegates to a real connection and records the SQL each fetch() sends."""

    def __init__(self, conn):
        self._conn = conn
        self.queries: list[str] = []

    def __getattr__(self, name):
        return getattr(self._conn, name)

    async def fetch(self, query, *args, **kwargs):
        self.queries.append(query)
        return await self._conn.fetch(query, *args, **kwargs)


def _ids(rows) -> dict[str, float]:
    return {str(r["id"]): round(float(r["score"]), 6) for r in rows}


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_pgq_expansion_matches_relational_on_oracle(oracle_memory, request_context, monkeypatch, caplog):
    """On a real ADB every SQL/PGQ arm runs, without fallback, and agrees with the relational CTEs.

    The retriever turns any SQL/PGQ error into the relational fallback, so a GRAPH_TABLE query
    that Oracle rejects would still pass every mocked test above. This seeds a small graph and
    runs both shapes. Unit ids are pinned so the seed's source ranks first under every entity:
    with a small per-entity cap it then takes the only slot, which is where a ranking that
    skips the seed's sources would return more than the relational CTE.
    """
    from hindsight_api.engine.retain.types import embedding_to_pgvector
    from hindsight_api.engine.schema import fq_store_table as fq_table

    bank_id = f"test-pgq-{uuid.uuid4().hex[:8]}"
    dim = oracle_memory.embeddings.dimension
    # Highest id first: ORDER BY unit_id DESC ranks them in this order.
    names = ["w_seed", "w_a", "w_b", "x_other", "w_c", "w_far", "o_seed", "o_two", "o_one", "o_none"]
    fact_types = ["world", "world", "world", "experience", "world", "world"] + ["observation"] * 4
    pinned = [uuid.UUID(f"{0xF0 - 0x10 * i:02x}" + uuid.uuid4().hex[2:]) for i in range(len(names))]
    e1, e2, e3 = (str(uuid.uuid4()) for _ in range(3))
    n = len(names)
    graph = "hindsight_memory_graph"
    mu, ue, ml = fq_table("memory_units"), fq_table("unit_entities"), fq_table("memory_links")
    try:
        await oracle_memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
        backend = await oracle_memory._get_backend()
        ops = backend.ops
        async with backend.transaction() as conn:
            with monkeypatch.context() as m:
                m.setattr(uuid, "uuid4", iter(pinned).__next__)
                ids = await ops.insert_facts_batch(
                    conn,
                    bank_id,
                    fact_texts=[f"pgq {name}" for name in names],
                    embeddings=[embedding_to_pgvector([1.0] + [0.0] * (dim - 1))] * n,
                    event_dates=[datetime(2024, 1, 1, tzinfo=timezone.utc)] * n,
                    occurred_starts=[None] * n,
                    occurred_ends=[None] * n,
                    mentioned_ats=[None] * n,
                    contexts=[""] * n,
                    fact_types=fact_types,
                    metadata_jsons=["{}"] * n,
                    chunk_ids=[None] * n,
                    document_ids=[None] * n,
                    tags_list=["[]"] * n,
                    observation_scopes_list=[None] * n,
                    text_signals_list=[None] * n,
                    attachment_ids_list=[None] * n,
                )
            u = dict(zip(names, (str(i) for i in ids)))
            assert u["w_seed"] == str(pinned[0])
            for eid, name in ((e1, "e1"), (e2, "e2"), (e3, "e3")):
                await conn.execute(
                    f"INSERT INTO {fq_table('entities')} (id, bank_id, canonical_name) VALUES ($1, $2, $3)",
                    eid,
                    bank_id,
                    f"pgq {name}",
                )
            for unit, eid in (
                ("w_seed", e1), ("w_a", e1), ("w_b", e1), ("x_other", e1),
                ("w_seed", e2), ("w_c", e2),
                ("w_far", e3),
            ):  # fmt: skip
                await conn.execute(f"INSERT INTO {ue} (unit_id, entity_id) VALUES ($1, $2)", u[unit], eid)
            for a, b in ((e1, e2), (e2, e3)):
                await conn.execute(
                    f"INSERT INTO {fq_table('entity_cooccurrences')} (entity_id_1, entity_id_2) VALUES ($1, $2)", a, b
                )
            for src, dst, link_type, weight in (
                ("w_seed", "w_a", "semantic", 0.9),
                ("w_b", "w_seed", "semantic", 0.8),
                ("w_seed", "w_c", "causes", 0.7),
                ("w_c", "w_far", "enables", 0.5),
                ("o_seed", "o_one", "semantic", 0.6),
                ("o_seed", "o_two", "causes", 0.4),
            ):
                await conn.execute(
                    f"INSERT INTO {ml} (from_unit_id, to_unit_id, link_type, weight, bank_id) VALUES ($1, $2, $3, $4, $5)",
                    u[src],
                    u[dst],
                    link_type,
                    weight,
                    bank_id,
                )
            for obs, src in (
                ("o_seed", "w_seed"),
                ("o_two", "w_a"),
                ("o_two", "w_b"),
                ("o_one", "w_c"),
                ("o_none", "w_far"),
            ):
                await conn.execute(
                    f"INSERT INTO {fq_table('observation_sources')} (observation_id, source_id) VALUES ($1, $2)",
                    u[obs],
                    u[src],
                )

        by_id = {v: k for k, v in u.items()}

        def named(rows) -> dict[str, float]:
            return {by_id[i]: s for i, s in _ids(rows).items()}

        window = UpdatedWindow(after=None, before=None, first_param_index=4)
        obs_window = UpdatedWindow(after=None, before=None, first_param_index=3)
        async with backend.acquire() as conn:
            if not await ops.has_property_graph(conn):
                # The migration skips the graph without CREATE PROPERTY GRAPH (ORA-01031).
                pytest.skip("hindsight_memory_graph is absent: the test user lacks CREATE PROPERTY GRAPH")

            for cap, want_entity, want_obs in (
                (0, {}, {}),
                (1, {"w_a": 1.0, "w_c": 1.0}, {}),
                (2, {"w_a": 1.0, "w_b": 1.0, "w_c": 1.0}, {"o_two": 1.0, "o_one": 1.0}),
                (4, {"w_a": 1.0, "w_b": 1.0, "w_c": 1.0}, {"o_two": 2.0, "o_one": 1.0}),
            ):
                rel_rows = await conn.fetch(
                    f"""WITH {ops.build_entity_expansion_cte(mu, ue, cap, window)},
                    {ops.build_semantic_causal_cte(ml, mu, window)}
                    SELECT * FROM entity_expanded UNION ALL SELECT * FROM semantic_expanded
                    UNION ALL SELECT * FROM causal_expanded""",
                    [u["w_seed"]],
                    "world",
                    10,
                )
                pgq_rows = await conn.fetch(
                    f"""WITH {build_pgq_entity_expansion_cte(graph, mu, cap, window)},
                    {build_pgq_semantic_causal_cte(graph, mu, window, max_causal_hops=2)}
                    SELECT * FROM entity_expanded UNION ALL SELECT * FROM semantic_expanded
                    UNION ALL SELECT * FROM causal_expanded""",
                    [u["w_seed"]],
                    "world",
                    10,
                )
                arm = {"rel": {}, "pgq": {}}
                for key, rows in (("rel", rel_rows), ("pgq", pgq_rows)):
                    for source in ("entity", "semantic", "causal"):
                        arm[key][source] = named([r for r in rows if r["source"] == source])
                assert arm["pgq"]["entity"] == arm["rel"]["entity"] == want_entity, (cap, arm)
                assert arm["pgq"]["semantic"] == arm["rel"]["semantic"] == {"w_a": 0.9, "w_b": 0.8}
                # The relational causal arm is one hop; SQL/PGQ adds the 2-hop w_seed -> w_c -> w_far.
                assert arm["rel"]["causal"] == {"w_c": 0.7}
                assert arm["pgq"]["causal"] == {"w_c": 0.7, "w_far": round(0.7 * 0.5 * 0.7, 6)}

                rel_obs = await ops.expand_observations(conn, mu, ue, ml, [u["o_seed"]], 10, cap, obs_window)
                pgq_obs = await expand_observations_pgq(
                    conn, graph, mu, [u["o_seed"]], 10, cap, obs_window, max_causal_hops=2
                )
                assert named(pgq_obs.entity) == named(rel_obs.entity) == want_obs, cap
                assert named(pgq_obs.semantic) == named(rel_obs.semantic) == {"o_one": 0.6}
                assert named(pgq_obs.causal) == named(rel_obs.causal) == {"o_two": 0.4}

            neighborhood = await query_entity_neighborhood_pgq(
                conn, mu_table=mu, graph_name=graph, bank_id=bank_id, entity_ids=[e1], max_hops=2, limit=50
            )
            hops = {(by_id[str(r["id"])], str(r["neighbor_entity_id"])): int(r["hop_distance"]) for r in neighborhood}
            assert hops == {
                ("w_seed", e1): 0, ("w_a", e1): 0, ("w_b", e1): 0, ("x_other", e1): 0,
                ("w_seed", e2): 1, ("w_c", e2): 1,
                ("w_far", e3): 2,
            }  # fmt: skip
            via_ops = await ops.query_entity_neighborhood_pgq(
                conn, mu_table=mu, graph_name=graph, bank_id=bank_id, entity_ids=[e1], max_hops=2, limit=50
            )
            assert {
                (by_id[str(r["id"])], str(r["neighbor_entity_id"])): int(r["hop_distance"]) for r in via_ops
            } == hops
            direct = await query_entity_neighborhood_pgq(
                conn, mu_table=mu, graph_name=graph, bank_id=bank_id, entity_ids=[e1], max_hops=0, limit=50
            )
            assert {by_id[str(r["id"])] for r in direct} == {"w_seed", "w_a", "w_b", "x_other"}
            assert {int(r["hop_distance"]) for r in direct} == {0}

            # The retriever takes the SQL/PGQ path end to end, with no fallback: each call sends
            # exactly one query, and it is the SQL/PGQ one (the relational rows are the same).
            recorded = _RecordingConnection(conn)
            caplog.clear()
            with caplog.at_level("WARNING", logger="hindsight_api.engine.memories.ora.link_expansion"):
                retriever = OracleLinkExpansionRetriever()
                combined = await retriever._expand_combined(recorded, [u["w_seed"]], "world", 10, ops=ops)
                combined_sql = list(recorded.queries)
                recorded.queries.clear()
                observed = await retriever._expand_observations(recorded, [u["o_seed"]], 10, ops=ops)
                observed_sql = list(recorded.queries)
            assert not [r for r in caplog.records if "falling back" in r.getMessage()], caplog.text
            assert len(combined_sql) == 1 and "MATCH (src IS memory_unit) -[m1 IS mentions]->" in combined_sql[0]
            assert len(observed_sql) == 1 and "-[d IS derived_from]->" in observed_sql[0]
            assert named(combined.entity) == {"w_a": 1.0, "w_b": 1.0, "w_c": 1.0}
            assert named(combined.semantic) == {"w_a": 0.9, "w_b": 0.8}
            assert named(combined.causal) == {"w_c": 0.7, "w_far": round(0.7 * 0.5 * 0.7, 6)}
            assert named(observed.entity) == {"o_two": 2.0, "o_one": 1.0}
            assert named(observed.semantic) == {"o_one": 0.6}
            assert named(observed.causal) == {"o_two": 0.4}
    finally:
        try:
            await oracle_memory.delete_bank(bank_id, request_context=request_context)
        except Exception:
            pass
