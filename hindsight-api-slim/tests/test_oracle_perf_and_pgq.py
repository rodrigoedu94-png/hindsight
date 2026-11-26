"""Unit tests for Oracle 26ai optimizations and native SQL/PGQ property graph support."""

import asyncio
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.config import HindsightConfig, clear_config_cache
from hindsight_api.engine.db.ops import MEMORY_UNIT_COLUMNS, UpdatedWindow
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.oracle import OracleConnection, _rewrite_pg_to_oracle
from hindsight_api.engine.memories.ora import recall as ora_recall
from hindsight_api.engine.memories.pg import recall as pg_recall
from hindsight_api.engine.memories.postgres import PostgresMemories
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.search.retrieval import tokenize_query
from hindsight_api.engine.sql.oracle import OracleDialect
from tests.test_link_expansion_projection import _arm_projections


class TestOracleExpandListsAndRewriterCache:
    """Verify cursor-cache friendly SQL stability and rewriter LRU isolation."""

    def test_consecutive_expansions_produce_identical_sql(self):
        q, _, _ = _rewrite_pg_to_oracle("SELECT id FROM chunks WHERE chunk_id = ANY($1)")
        sql1, params1 = OracleConnection._expand_any_lists(q, {"1": ["c1", "c2"]})
        sql2, params2 = OracleConnection._expand_any_lists(q, {"1": ["c3", "c4"]})

        assert sql1 == sql2
        assert "chunks.chunk_id" not in sql1
        assert ":any0_0" in sql1 and ":any0_1" in sql1
        assert params1 == {"any0_0": "c1", "any0_1": "c2"}
        assert params2 == {"any0_0": "c3", "any0_1": "c4"}

    def test_power_of_two_bucketing_shares_cursor_shape(self):
        q, _, _ = _rewrite_pg_to_oracle("SELECT id FROM memory_units WHERE id = ANY($1::uuid[])")
        u1, u2, u3, u4 = (uuid.uuid4() for _ in range(4))

        sql3, params3 = OracleConnection._expand_any_lists(q, {"1": [u1.bytes, u2.bytes, u3.bytes]})
        sql4, params4 = OracleConnection._expand_any_lists(q, {"1": [u1.bytes, u2.bytes, u3.bytes, u4.bytes]})

        assert sql3 == sql4
        assert len(params3) == 4
        assert params3["any0_3"] == u3.bytes
        assert len(params4) == 4
        assert params4["any0_3"] == u4.bytes

    def test_multiple_any_references_get_deterministic_slot_prefixes(self):
        q, _, _ = _rewrite_pg_to_oracle(
            "SELECT id FROM memory_units WHERE id = ANY($1::uuid[]) AND id != ALL($1::uuid[])"
        )
        ids = [uuid.uuid4().bytes, uuid.uuid4().bytes, uuid.uuid4().bytes]
        sql_a, params_a = OracleConnection._expand_any_lists(q, {"1": list(ids)})
        sql_b, params_b = OracleConnection._expand_any_lists(q, {"1": list(ids)})

        assert sql_a == sql_b
        assert ":any0_0" in sql_a and ":any0_3" in sql_a
        assert ":any1_0" in sql_a and ":any1_3" in sql_a
        assert "1" not in params_a
        assert len(params_a) == len(params_b) == 8

    def test_null_elements_skip_bucketing_safely(self):
        q, _, _ = _rewrite_pg_to_oracle("SELECT id FROM memory_units WHERE chunk_id = ANY($1)")
        sql, params = OracleConnection._expand_any_lists(q, {"1": ["a", None, "b"]})

        assert sql.count(":any0_") == 3
        assert params == {"any0_0": "a", "any0_1": None, "any0_2": "b"}

    def test_rewriter_lru_cache_hits_and_isolates_returning_list(self):
        _rewrite_pg_to_oracle.cache_clear()
        sql = "UPDATE banks SET name = $1 WHERE bank_id = $2 RETURNING bank_id, name"

        first = _rewrite_pg_to_oracle(sql)
        assert first.returning_cols == ["bank_id", "name"]
        first.returning_cols.append("mutated")

        second = _rewrite_pg_to_oracle(sql)
        assert second.returning_cols == ["bank_id", "name"]
        assert _rewrite_pg_to_oracle.cache_info().hits >= 1


class TestOracleQueryPlansAndRecall:
    """Verify SQL plan hints, single-pass vector scan, and hot query shapes."""

    def test_bm25_arm_adds_domain_index_hint(self):
        arm = OracleDialect().build_bm25_arm(
            table='"ADMIN".memory_units',
            cols="id, text",
            fact_type="world",
            bank_id_param="$2",
            limit_param="$3",
            text_param="$4",
            arm_index=1,
        )
        assert 'FROM "ADMIN".memory_units m' in arm
        assert "/*+ INDEX(m idx_mu_content_text) */" in arm
        assert "CONTAINS(m.text, $4, 11)" in arm

    def test_single_pass_semantic_arm_uses_window_partition(self):
        sql = OracleDialect().build_semantic_single_pass_arm(
            table="memory_units",
            cols="id, text",
            fact_types=["world", "experience", "observation"],
            embedding_param="$1",
            bank_id_param="$2",
            fetch_limit=50,
            min_similarity=0.3,
        )
        assert "PARTITION BY fact_type" in sql
        assert "fact_type IN ('world', 'experience', 'observation')" in sql
        assert "WHERE rn_ <= 50" in sql
        assert "ORDER BY fact_type, rn_" in sql
        assert "FETCH EXACT FIRST" not in sql

    @pytest.mark.asyncio
    async def test_recall_uses_single_pass_only_on_oracle_exact_multi_fact_type(self, monkeypatch):
        captured: list[str] = []

        class _OracleConn:
            backend_type = "oracle"

            async def fetch(self, query, *_args):
                captured.append(query)
                return []

        monkeypatch.setattr(
            ora_recall,
            "get_config",
            lambda: SimpleNamespace(
                semantic_min_similarity=0.3,
                bm25_min_score=0.0,
                oracle_vector_search="exact",
                oracle_vector_target_accuracy=95,
                text_search_extension="native",
                text_search_extension_native_language="english",
                text_search_extension_pg_search_function_schema="paradedb",
                text_search_extension_pg_search_tokenizer="",
                bm25_max_query_terms=16,
            ),
        )
        monkeypatch.setattr(ora_recall, "tokenize_query", lambda _: [])
        monkeypatch.setattr(ora_recall, "fq_table", lambda t: t)

        store = PostgresMemories({})
        await store.search(
            conn=_OracleConn(),
            query_embedding="[0.1]",
            query_text="",
            bank_id="bank-1",
            fact_types=["world", "experience", "observation"],
            limit=20,
            enable_text_search=False,
        )
        assert len(captured) == 1
        assert "PARTITION BY fact_type" in captured[0]
        assert captured[0].count("FROM memory_units") == 1

        # In approx mode on Oracle, separate per-fact-type arms remain so vector indexes can apply.
        captured.clear()
        monkeypatch.setenv("HINDSIGHT_API_ORACLE_VECTOR_SEARCH", "approx")
        clear_config_cache()
        try:
            monkeypatch.setattr(
                ora_recall,
                "get_config",
                lambda: SimpleNamespace(
                    semantic_min_similarity=0.3,
                    bm25_min_score=0.0,
                    oracle_vector_search="approx",
                    oracle_vector_target_accuracy=95,
                    text_search_extension="native",
                    text_search_extension_native_language="english",
                    text_search_extension_pg_search_function_schema="paradedb",
                    text_search_extension_pg_search_tokenizer="",
                    bm25_max_query_terms=16,
                ),
            )
            await store.search(
                conn=_OracleConn(),
                query_embedding="[0.1]",
                query_text="",
                bank_id="bank-1",
                fact_types=["world", "experience", "observation"],
                limit=20,
                enable_text_search=False,
            )
            assert captured[0].count("FETCH APPROX FIRST") == 3
        finally:
            monkeypatch.delenv("HINDSIGHT_API_ORACLE_VECTOR_SEARCH", raising=False)
            clear_config_cache()

        # Verify DRG-10599 Oracle Text fallback renumbers tags + created_after/created_before params cleanly.
        fb_calls: list[tuple[str, tuple]] = []

        class _OracleDrgConn:
            backend_type = "oracle"

            async def fetch(self, query, *args):
                fb_calls.append((query, args))
                if len(fb_calls) == 1:
                    raise RuntimeError("DRG-10599: column is not indexed")
                return []

        monkeypatch.setattr(ora_recall, "tokenize_query", lambda _: ["oracle"])
        monkeypatch.setattr(
            ora_recall,
            "build_bm25_query_text",
            AsyncMock(return_value="oracle"),
        )
        t_after = datetime(2026, 1, 1, tzinfo=timezone.utc)
        await store.search(
            conn=_OracleDrgConn(),
            query_embedding="[0.1]",
            query_text="oracle",
            bank_id="bank-1",
            fact_types=["world", "experience"],
            limit=10,
            tags=["tag-a"],
            created_after=t_after,
            enable_text_search=True,
        )
        assert len(fb_calls) == 2
        fb_sql, fb_args = fb_calls[1]
        assert "CONTAINS(" not in fb_sql
        assert fb_args == ("[0.1]", "bank-1", ["tag-a"], t_after)

        # PG path stays untouched in pg/recall.py and emits per-fact-type arms.
        pg_captured: list[str] = []

        class _PgConn:
            backend_type = "postgresql"

            async def fetch(self, query, *_args):
                pg_captured.append(query)
                return []

        monkeypatch.setattr(pg_recall, "tokenize_query", lambda _: [])
        monkeypatch.setattr(pg_recall, "fq_table", lambda t: t)
        await store.search(
            conn=_PgConn(),
            query_embedding="[0.1]",
            query_text="",
            bank_id="bank-1",
            fact_types=["world", "experience"],
            limit=20,
            enable_text_search=False,
        )
        assert len(pg_captured) == 1
        assert "PARTITION BY fact_type" not in pg_captured[0]
        assert pg_captured[0].count("UNION ALL") == 1
        # Avoid unused import lint failure if tokenize_query wasn't called directly
        assert tokenize_query("hello") == ["hello"]

    @pytest.mark.asyncio
    async def test_bank_watermark_queries_use_index_hint_on_oracle_only(self):
        store = PostgresMemories({})
        oracle_conn = MagicMock()
        oracle_conn.backend_type = "oracle"
        oracle_conn.fetch = AsyncMock(return_value=[])
        await store.list_bank_rows(conn=oracle_conn, fq_table=lambda t: t, where_clause="", params=[])
        q_list = oracle_conn.fetch.await_args.args[0]
        assert "/*+ INDEX(m idx_mu_bank_updated_at) */" in q_list
        assert "ORDER BY SYS_EXTRACT_UTC(m.updated_at) DESC FETCH FIRST 1 ROW ONLY" in q_list

        await store.bank_page_rows(conn=oracle_conn, fq_table=lambda t: t, bank_ids=["b1"], sql_owned=["b1"])
        q_page = oracle_conn.fetch.await_args.args[0]
        assert "/*+ INDEX(m idx_mu_bank_updated_at) */" in q_page

        pg_conn = MagicMock()
        pg_conn.backend_type = "postgresql"
        pg_conn.fetch = AsyncMock(return_value=[])
        await store.list_bank_rows(conn=pg_conn, fq_table=lambda t: t, where_clause="", params=[])
        assert "MAX(m.updated_at)" in pg_conn.fetch.await_args.args[0]

    @pytest.mark.asyncio
    async def test_mental_model_history_trim_uses_row_number_on_oracle(self):
        oracle_conn = MagicMock()
        oracle_conn.backend_type = "oracle"
        oracle_conn.execute = AsyncMock(return_value="DELETE 0")
        engine = object.__new__(MemoryEngine)

        await engine._insert_mental_model_history_row(
            oracle_conn,
            bank_id="b1",
            mental_model_id="mm1",
            content_json='{"kind":"snapshot"}',
            max_entries=20,
            is_failure=False,
        )
        delete_sql = oracle_conn.execute.await_args_list[1].args[0]
        assert "ROW_NUMBER() OVER (ORDER BY SYS_EXTRACT_UTC(changed_at) DESC, id DESC)" in delete_sql
        assert "WHERE rn > $3" in delete_sql
        assert "NOT IN" not in delete_sql


class TestOracleOpsAndPropertyGraph:
    """Verify OracleOps batching and Oracle 26ai SQL/PGQ query generation."""

    @pytest.mark.asyncio
    async def test_fetch_unit_dates_batches_into_single_query(self):
        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=[{"id": "u1", "event_date": None, "fact_type": "world"}])
        ops = OracleOps()

        rows = await ops.fetch_unit_dates(conn, "memory_units", ["u1", "u2"])
        assert len(rows) == 1
        conn.fetch.assert_awaited_once()
        assert "WHERE id = ANY($1::uuid[])" in conn.fetch.await_args.args[0]

        conn.fetch.reset_mock()
        assert await ops.fetch_unit_dates(conn, "memory_units", []) == []
        conn.fetch.assert_not_awaited()

    @pytest.mark.parametrize("max_hops", [1, 2])
    def test_pgq_semantic_causal_cte_preserves_projection_contract(self, max_hops):
        ops = OracleOps()
        window = UpdatedWindow(
            after=datetime(2026, 1, 1, tzinfo=timezone.utc),
            before=None,
            first_param_index=4,
        )
        cte = (
            ops.build_entity_expansion_cte("mu_t", "ue_t", 5, window)
            + ",\n"
            + ops.build_pgq_semantic_causal_cte("hindsight_memory_graph", "mu_t", window, max_causal_hops=max_hops)
        )
        arms = _arm_projections(cte)
        for name in ("entity_expanded", "semantic_expanded", "causal_expanded"):
            assert arms[name] == list(MEMORY_UNIT_COLUMNS), f"Unexpected projection on {name}: {arms.get(name)}"
        assert "GRAPH_TABLE (" in cte
        assert "hindsight_memory_graph" in cte
        assert "dst.updated_at > $4" in cte
        if max_hops == 2:
            assert "l1.weight * l2.weight * 0.7" in cte
        else:
            assert "l1.weight * l2.weight * 0.7" not in cte

    @pytest.mark.asyncio
    async def test_has_property_graph_caches_and_honors_toggle(self, monkeypatch):
        ops = OracleOps()
        conn = MagicMock()
        conn.fetchval = AsyncMock(return_value=1)

        monkeypatch.setenv("HINDSIGHT_API_ORACLE_USE_PROPERTY_GRAPH", "true")
        clear_config_cache()
        try:
            assert await ops.has_property_graph(conn) is True
            assert await ops.has_property_graph(conn) is True
            assert conn.fetchval.await_count == 1

            monkeypatch.setenv("HINDSIGHT_API_ORACLE_USE_PROPERTY_GRAPH", "false")
            clear_config_cache()
            assert await ops.has_property_graph(conn) is False
        finally:
            clear_config_cache()

    @pytest.mark.asyncio
    async def test_query_entity_neighborhood_pgq_emits_bitemporal_variable_length_query(self):
        ops = OracleOps()
        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=[])
        e1 = uuid.uuid4()
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t1 = datetime(2026, 11, 1, tzinfo=timezone.utc)

        await ops.query_entity_neighborhood_pgq(
            conn,
            mu_table='"ADMIN".memory_units',
            graph_name='"ADMIN".hindsight_memory_graph',
            bank_id="ecossistema-ia",
            entity_ids=[e1],
            max_hops=2,
            occurred_after=t0,
            occurred_before=t1,
            limit=25,
        )
        sql = conn.fetch.await_args.args[0]
        assert "GRAPH_TABLE (" in sql
        # Delegates to ora/pgq.py: explicit hop arms, no quantified edge (which reported 2 hops as 1).
        assert "{1,2}" not in sql
        assert "1 AS hop_distance" in sql and "2 AS hop_distance" in sql
        assert "COALESCE(mu.occurred_end, mu.occurred_start, mu.mentioned_at, mu.event_date) >= $3" in sql
        assert "COALESCE(mu.occurred_start, mu.mentioned_at, mu.event_date) <= $4" in sql
        assert "LIMIT $5" in sql

    def test_pgq_config_env_and_validation(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_ORACLE_USE_PROPERTY_GRAPH", "false")
        monkeypatch.setenv("HINDSIGHT_API_ORACLE_PGQ_MAX_HOPS", "1")
        try:
            cfg = HindsightConfig.from_env()
            cfg.validate()
            assert cfg.oracle_use_property_graph is False
            assert cfg.oracle_pgq_max_hops == 1

            monkeypatch.setenv("HINDSIGHT_API_ORACLE_PGQ_MAX_HOPS", "3")
            with pytest.raises(ValueError, match="oracle_pgq_max_hops"):
                HindsightConfig.from_env().validate()
        finally:
            monkeypatch.delenv("HINDSIGHT_API_ORACLE_USE_PROPERTY_GRAPH", raising=False)
            monkeypatch.delenv("HINDSIGHT_API_ORACLE_PGQ_MAX_HOPS", raising=False)

    @pytest.mark.asyncio
    async def test_oracle_link_expansion_retriever_uses_pgq_and_falls_back(self, monkeypatch):
        from hindsight_api.engine.memories.ora import link_expansion as ora_le
        from hindsight_api.engine.memories.ora.link_expansion import OracleLinkExpansionRetriever
        from hindsight_api.engine.memories.pg.link_expansion import LinkExpansionRetriever

        monkeypatch.setattr(ora_le, "fq_table", lambda t: t)
        ops = OracleOps()
        monkeypatch.setattr(ops, "has_property_graph", AsyncMock(return_value=True))

        conn = MagicMock()
        conn.backend_type = "oracle"
        conn.fetch = AsyncMock(return_value=[{"source": "semantic", "id": "u1"}])

        retriever = OracleLinkExpansionRetriever()
        res = await retriever._expand_combined(conn, ["seed-1"], "world", 20, ops=ops)
        assert len(res.semantic) == 1
        assert "GRAPH_TABLE (" in conn.fetch.await_args.args[0]

        # When PGQ query raises, it delegates cleanly to LinkExpansionRetriever._expand_combined.
        conn.fetch = AsyncMock(side_effect=RuntimeError("ORA-49022: graph unavailable"))
        fallback_rows = MagicMock()
        monkeypatch.setattr(LinkExpansionRetriever, "_expand_combined", AsyncMock(return_value=fallback_rows))
        res_fb = await retriever._expand_combined(conn, ["seed-1"], "world", 20, ops=ops)
        assert res_fb is fallback_rows

        # When PGQ times out, semantic+causal fallback runs with bounded timeout.
        conn.fetch = AsyncMock(
            side_effect=[
                asyncio.TimeoutError(),
                [{"source": "causal", "id": "u2"}],
            ]
        )
        res_to = await retriever._expand_combined(conn, ["seed-1"], "world", 20, ops=ops)
        assert len(res_to.causal) == 1
        assert conn.fetch.await_count == 2

        # When both PGQ and its timeout fallback fail/timeout, degrade to relational super()._expand_combined.
        conn.fetch = AsyncMock(side_effect=[asyncio.TimeoutError(), asyncio.TimeoutError()])
        res_to_fb = await retriever._expand_combined(conn, ["seed-1"], "world", 20, ops=ops)
        assert res_to_fb is fallback_rows

    @pytest.mark.asyncio
    async def test_uuid_shaped_bank_ids_stay_str_in_ora_modules(self):
        from hindsight_api.engine.db.oracle import OracleLiteralStr, _convert_arg
        from hindsight_api.engine.memories.ora import banks as ora_banks
        from hindsight_api.engine.memories.ora import engine_curation as ora_curation
        from hindsight_api.engine.memories.ora import recall as ora_recall

        uuid_bank = "11111111-2222-3333-4444-555555555555"
        conn = MagicMock()
        conn.backend_type = "oracle"
        conn.fetch = AsyncMock(return_value=[])
        conn.execute = AsyncMock(return_value="DELETE 0")

        await ora_banks.bank_page_rows(conn=conn, fq_table=lambda t: t, bank_ids=[uuid_bank], sql_owned=[uuid_bank])
        assert all(
            isinstance(b, OracleLiteralStr) and isinstance(_convert_arg(b), str) for b in conn.fetch.await_args.args[1]
        )

        conn.fetch.reset_mock()
        await ora_curation.purge_bank_rows(conn=conn, fq_table=lambda t: t, bank_id=uuid_bank)
        assert isinstance(conn.fetch.await_args_list[0].args[1], OracleLiteralStr)
        assert isinstance(_convert_arg(conn.fetch.await_args_list[0].args[1]), str)

        conn.fetch.reset_mock()
        await ora_recall.retrieve_semantic_bm25_combined_sql(
            conn, "[0.1]", "", uuid_bank, ["world"], 5, enable_text_search=False
        )
        assert isinstance(conn.fetch.await_args.args[2], OracleLiteralStr)
        assert isinstance(_convert_arg(conn.fetch.await_args.args[2]), str)

    def test_graph_retriever_dispatch_and_ora_importable_without_oracledb(self, monkeypatch):
        import subprocess
        import sys

        from hindsight_api.engine.memories.ora.link_expansion import OracleLinkExpansionRetriever
        from hindsight_api.engine.memories.pg.link_expansion import LinkExpansionRetriever

        store = PostgresMemories({})
        monkeypatch.setenv("HINDSIGHT_API_DATABASE_BACKEND", "oracle")
        clear_config_cache()
        try:
            assert isinstance(store.graph_retriever(), OracleLinkExpansionRetriever)
        finally:
            monkeypatch.delenv("HINDSIGHT_API_DATABASE_BACKEND", raising=False)
            clear_config_cache()

        monkeypatch.setenv("HINDSIGHT_API_DATABASE_BACKEND", "postgresql")
        clear_config_cache()
        try:
            assert type(store.graph_retriever()) is LinkExpansionRetriever
        finally:
            monkeypatch.delenv("HINDSIGHT_API_DATABASE_BACKEND", raising=False)
            clear_config_cache()

        # Verify all ora/* modules import cleanly in a fresh process without oracledb.
        probe = (
            "import importlib, pkgutil, sys\n"
            "class BlockOracle:\n"
            "    def find_spec(self, name, path, target=None):\n"
            "        if name == 'oracledb' or name.startswith('oracledb.'):\n"
            "            raise ImportError('blocked')\n"
            "sys.meta_path.insert(0, BlockOracle())\n"
            "ora_pkg = importlib.import_module('hindsight_api.engine.memories.ora')\n"
            "for mod in pkgutil.iter_modules(ora_pkg.__path__, ora_pkg.__name__ + '.'):\n"
            "    assert importlib.import_module(mod.name) is not None\n"
        )
        proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr
