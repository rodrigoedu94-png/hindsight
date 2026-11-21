"""Unit tests for Oracle 26ai optimizations and native SQL/PGQ property graph support."""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.config import HindsightConfig, clear_config_cache
from hindsight_api.engine.db.ops import MEMORY_UNIT_COLUMNS, UpdatedWindow
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.oracle import OracleConnection, _rewrite_pg_to_oracle
from hindsight_api.engine.memories.pg import banks as pg_banks
from hindsight_api.engine.memories.pg import recall as pg_recall
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
            pg_recall,
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
        monkeypatch.setattr(pg_recall, "tokenize_query", lambda _: [])
        monkeypatch.setattr(pg_recall, "fq_table", lambda t: t)

        await pg_recall.retrieve_semantic_bm25_combined_sql(
            _OracleConn(),
            "[0.1]",
            "",
            "bank-1",
            ["world", "experience", "observation"],
            20,
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
                pg_recall,
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
            await pg_recall.retrieve_semantic_bm25_combined_sql(
                _OracleConn(),
                "[0.1]",
                "",
                "bank-1",
                ["world", "experience", "observation"],
                20,
                enable_text_search=False,
            )
            assert captured[0].count("FETCH APPROX FIRST") == 3
        finally:
            monkeypatch.delenv("HINDSIGHT_API_ORACLE_VECTOR_SEARCH", raising=False)
            clear_config_cache()
        # Avoid unused import lint failure if tokenize_query wasn't called directly
        assert tokenize_query("hello") == ["hello"]

    @pytest.mark.asyncio
    async def test_bank_watermark_queries_use_index_hint_on_oracle_only(self):
        oracle_conn = MagicMock()
        oracle_conn.backend_type = "oracle"
        oracle_conn.fetch = AsyncMock(return_value=[])
        await pg_banks.list_bank_rows(conn=oracle_conn, fq_table=lambda t: t, where_clause="", params=[])
        q_list = oracle_conn.fetch.await_args.args[0]
        assert "/*+ INDEX(m idx_mu_bank_updated_at) */" in q_list
        assert "ORDER BY SYS_EXTRACT_UTC(m.updated_at) DESC FETCH FIRST 1 ROW ONLY" in q_list

        await pg_banks.bank_page_rows(conn=oracle_conn, fq_table=lambda t: t, bank_ids=["b1"], sql_owned=["b1"])
        q_page = oracle_conn.fetch.await_args.args[0]
        assert "/*+ INDEX(m idx_mu_bank_updated_at) */" in q_page

        pg_conn = MagicMock()
        pg_conn.backend_type = "postgresql"
        pg_conn.fetch = AsyncMock(return_value=[])
        await pg_banks.list_bank_rows(conn=pg_conn, fq_table=lambda t: t, where_clause="", params=[])
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
        assert "-[co IS cooccurs_with]-{1,2}" in sql
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
