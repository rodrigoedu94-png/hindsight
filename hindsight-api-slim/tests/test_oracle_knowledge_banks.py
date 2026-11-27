"""Unit tests for Oracle AI Database 26ai Knowledge Banks support."""

from __future__ import annotations

import array
import importlib.util
import json
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.db.oracle import OracleLiteralStr, _convert_arg
from hindsight_api.engine.search.tags import TagGroupLeaf
from hindsight_api.knowledge.service import KnowledgeService
from hindsight_api.knowledge.store import (
    KnowledgeBankNotFound,
    KnowledgeDocument,
    KnowledgePassage,
    PassageReplacement,
    PassageStage,
    PostgresKnowledgeStore,
    RoleVector,
    SearchFilters,
    StatusUpdate,
    StoreSearchTrace,
    load_knowledge_store,
)
from hindsight_api.knowledge.store_oracle import OracleKnowledgeStore


def _load_migration_module(name: str = "e3c4d5e6f7a8_oracle_knowledge_banks"):
    path = Path(__file__).resolve().parents[1] / "hindsight_api" / "alembic" / "versions" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _grantee_queries(plsql: str) -> list[str]:
    """The dynamic SQL each ``OPEN l_rc FOR`` in a grant block runs, with its PL/SQL quoting undone."""
    queries = []
    for stmt in re.findall(r"OPEN l_rc FOR(.*?);", plsql, re.S):
        pieces = re.findall(r"'((?:[^']|'')*)'", stmt)
        queries.append("".join(pieces).replace("''", "'"))
    return queries


def _assert_grantees_scoped_to_current_schema(plsql: str, privilege: str) -> None:
    schema = "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')"
    dba, fallback = _grantee_queries(plsql)
    assert dba.startswith(f"SELECT DISTINCT grantee FROM dba_tab_privs WHERE owner = {schema} ")
    assert fallback.startswith(f"SELECT DISTINCT grantee FROM all_tab_privs WHERE table_schema = {schema} ")
    for query in (dba, fallback):
        assert f"AND table_name = 'MEMORY_UNITS' AND privilege = '{privilege}' " in query
        assert query.endswith(f"grantee NOT IN ('PUBLIC', SYS_CONTEXT('USERENV', 'SESSION_USER'), {schema})")


class _FakeConn:
    backend_type = "oracle"

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.executemany_calls: list[tuple[str, list]] = []
        self.fetch_responses: list[list[dict]] = []
        self.fetchval_responses: list[object] = []
        self.fetchrow_responses: list[dict | None] = []
        self.tx_entered = 0

    @asynccontextmanager
    async def transaction(self):
        self.tx_entered += 1
        yield

    async def execute(self, sql: str, *args):
        self.executed.append((" ".join(sql.split()), args))
        return "OK"

    async def executemany(self, sql: str, rows: list):
        self.executemany_calls.append((" ".join(sql.split()), list(rows)))

    async def fetch(self, sql: str, *args):
        self.executed.append((" ".join(sql.split()), args))
        return self.fetch_responses.pop(0) if self.fetch_responses else []

    async def fetchval(self, sql: str, *args):
        self.executed.append((" ".join(sql.split()), args))
        return self.fetchval_responses.pop(0) if self.fetchval_responses else None

    async def fetchrow(self, sql: str, *args):
        self.executed.append((" ".join(sql.split()), args))
        return self.fetchrow_responses.pop(0) if self.fetchrow_responses else None


class _FakeBackend:
    backend_type = "oracle"

    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def acquire(self):
        return self._conn

    async def release(self, _conn) -> None:
        return None


def _sample_doc_row(doc_id: str = "doc-1", *, tags: object = '["alpha", "beta"]') -> dict:
    now = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
    return {
        "document_id": doc_id,
        "filename": f"{doc_id}.md",
        "content_type": "text/markdown",
        "media_type": "text",
        "size_bytes": 128,
        "storage_key": f"kb/{doc_id}.md",
        "content_hash": "hash-1",
        "status": "ready",
        "error": None,
        "passage_count": 2,
        "tags": tags,
        "created_at": now,
        "updated_at": now,
        "source_document_id": None,
        "content_tokens": 42,
    }


def test_oracle_knowledge_banks_migration_ddl_and_dialect():
    mod = _load_migration_module()
    assert mod.revision == "e3c4d5e6f7a8"
    assert mod.down_revision == "o2b3c4d5e6f7"

    exec_stmts: list[str] = []
    mock_bind = MagicMock()
    mock_bind.dialect.name = "oracle"
    mock_bind.exec_driver_sql.side_effect = lambda _block, params=None: exec_stmts.append(
        params["stmt"] if isinstance(params, dict) and "stmt" in params else _block
    )

    with patch.object(mod.op, "get_bind", return_value=mock_bind):
        mod.upgrade()
        mod.upgrade()
        mod.downgrade()

    joined = "\n".join(exec_stmts)
    for tbl in ("kb_documents", "kb_passages", "kb_passage_vectors", "kb_staged_passages", "kb_staged_passage_vectors"):
        assert joined.count(f"CREATE TABLE {tbl}") == 2
        assert f"DROP TABLE {tbl} CASCADE CONSTRAINTS PURGE" in joined
    assert "idx_kb_passages_text" in joined
    assert "CTXSYS.CONTEXT" in joined
    # Grantees are read through dynamic SQL (dba_tab_privs, falling back to all_tab_privs),
    # so the literals inside the PL/SQL string are quote-doubled.
    assert "privilege = ''INSERT''" in joined
    assert "FROM dba_tab_privs" in joined and "FROM all_tab_privs" in joined
    assert "user_tab_privs_made" not in joined
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON" in joined
    # A separate migration user sets CURRENT_SCHEMA to the owner: both lookups must scope to it.
    _assert_grantees_scoped_to_current_schema(mod._GRANT_KB_TABLES_PLSQL, "INSERT")

    pg_bind = MagicMock()
    pg_bind.dialect.name = "postgresql"
    with patch.object(mod.op, "get_bind", return_value=pg_bind):
        mod.upgrade()
        mod.downgrade()
    pg_bind.exec_driver_sql.assert_not_called()


def test_property_graph_grant_scoped_to_current_schema():
    from hindsight_api.alembic._oracle_grants import GRANT_PROPERTY_GRAPH_PLSQL

    _assert_grantees_scoped_to_current_schema(GRANT_PROPERTY_GRAPH_PLSQL, "SELECT")
    assert "GRANT READ ON PROPERTY GRAPH hindsight_memory_graph" in GRANT_PROPERTY_GRAPH_PLSQL
    assert "-42421" in GRANT_PROPERTY_GRAPH_PLSQL  # ORA-42421 (graph absent) is tolerated


def test_reapply_runtime_grants_migration_reruns_both_blocks_on_oracle_only():
    from hindsight_api.alembic._oracle_grants import GRANT_KB_TABLES_PLSQL, GRANT_PROPERTY_GRAPH_PLSQL

    mod = _load_migration_module("f7a8b9c0d1e2_oracle_reapply_runtime_grants")
    assert mod.revision == "f7a8b9c0d1e2"
    assert mod.down_revision == "e3c4d5e6f7a8"
    # The original revisions run the same shared blocks.
    assert _load_migration_module()._GRANT_KB_TABLES_PLSQL is GRANT_KB_TABLES_PLSQL

    ora_bind = MagicMock()
    ora_bind.dialect.name = "oracle"
    with patch.object(mod.op, "get_bind", return_value=ora_bind):
        mod.upgrade()
        mod.downgrade()
    executed = [c.args[0] for c in ora_bind.exec_driver_sql.call_args_list]
    assert executed == [GRANT_PROPERTY_GRAPH_PLSQL, GRANT_KB_TABLES_PLSQL]

    pg_bind = MagicMock()
    pg_bind.dialect.name = "postgresql"
    with patch.object(mod.op, "get_bind", return_value=pg_bind):
        mod.upgrade()
        mod.downgrade()
    pg_bind.exec_driver_sql.assert_not_called()


def test_oracle_literal_str_preserves_uuid_document_ids():
    raw_uuid = "550e8400-e29b-41d4-a716-446655440000"
    assert isinstance(_convert_arg(raw_uuid), bytes)
    assert _convert_arg(OracleLiteralStr(raw_uuid)) == raw_uuid


def test_load_knowledge_store_dispatches_oracle_and_postgres():
    async def _provider():
        return None

    assert isinstance(load_knowledge_store(_provider, None, backend_type="oracle"), OracleKnowledgeStore)
    assert isinstance(load_knowledge_store(_provider, None, backend_type="postgresql"), PostgresKnowledgeStore)


@pytest.mark.asyncio
async def test_oracle_store_put_list_get_delete_and_statuses():
    conn = _FakeConn()
    backend = _FakeBackend(conn)
    store = OracleKnowledgeStore(AsyncMock(return_value=backend))
    conn.fetch_responses.append([])  # ensure_ready dimension check on empty kb_passages
    await store.ensure_ready(1536)

    conn2 = _FakeConn()
    conn2.fetch_responses.append([{"dim": 768}])
    store2 = OracleKnowledgeStore(AsyncMock(return_value=_FakeBackend(conn2)))
    with pytest.raises(RuntimeError, match="does not match configured dimension 1536"):
        await store2.ensure_ready(1536)

    conn.fetch_responses.append([{"document_id": "doc-1", "storage_key": "old-key"}])
    prev = await store.put_documents(
        "kb-1",
        [
            KnowledgeDocument(
                document_id="doc-1",
                filename="doc-1.md",
                content_type="text/markdown",
                media_type="text",
                size_bytes=10,
                storage_key="new-key",
                content_hash="h1",
                status="pending",
                error=None,
                passage_count=0,
                tags=["alpha"],
                content_tokens=5,
            )
        ],
    )
    assert prev == {"doc-1": "old-key"}
    assert "MERGE INTO" in conn.executemany_calls[0][0]
    assert conn.tx_entered == 1

    conn.fetch_responses.append([_sample_doc_row("doc-1", tags=["alpha", "beta"])])
    docs = await store.get_documents("kb-1", ["doc-1"])
    assert docs["doc-1"].tags == ["alpha", "beta"]

    conn.fetchval_responses.append(1)
    conn.fetch_responses.append([_sample_doc_row("doc-1")])
    page = await store.list_documents("kb-1", limit=10, offset=0, query="doc", tags=["alpha"])
    assert page.total == 1
    assert page.documents[0].document_id == "doc-1"
    assert any("JSON_EXISTS(tags" in sql for sql, _ in conn.executed)

    conn.fetch_responses.append([{"document_id": "doc-1", "storage_key": "new-key"}])
    await store.set_statuses("kb-1", [StatusUpdate("doc-1", "failed", error="boom", storage_key="new-key")])
    assert any("DELETE FROM" in sql and "kb_staged_passages" in sql for sql, _ in conn.executed)

    conn.fetch_responses.append([{"document_id": "doc-1"}])
    deleted_count = await store.delete_some_documents("kb-1", 5)
    assert deleted_count == 1
    assert any("ROWNUM <= $2 FOR UPDATE SKIP LOCKED" in sql for sql, _ in conn.executed)

    conn.fetch_responses.append([_sample_doc_row("doc-1")])
    deleted = await store.delete_documents("kb-1", ["doc-1"])
    assert len(deleted) == 1


@pytest.mark.asyncio
async def test_oracle_store_stage_replace_and_passages():
    conn = _FakeConn()
    backend = _FakeBackend(conn)
    store = OracleKnowledgeStore(AsyncMock(return_value=backend))

    conn.fetch_responses.append([{"document_id": "doc-1", "storage_key": "k1"}])
    skipped = await store.stage_passages(
        "kb-1",
        [
            PassageStage(
                document_id="doc-1",
                storage_key="k1",
                staged=0,
                passages=[
                    KnowledgePassage(
                        index=0,
                        modality="text",
                        text="hello",
                        embedding=[0.1, 0.2],
                        extra_vectors=[RoleVector("window", 0, [0.3, 0.4])],
                    )
                ],
            ),
            PassageStage(document_id="doc-missing", storage_key="k2", staged=0, passages=[]),
        ],
    )
    assert skipped == ["doc-missing"]
    assert isinstance(conn.executemany_calls[0][1][0][-1], array.array)

    conn.fetch_responses.extend(
        [
            [{"document_id": "doc-1", "storage_key": "k1"}],
            [{"media_key": "stale-media"}],
            [{"media_key": "kept-media"}],
        ]
    )
    replaced = await store.replace_passages(
        "kb-1",
        [
            PassageReplacement(
                document_id="doc-1",
                storage_key="k1",
                staged=1,
                passages=[KnowledgePassage(index=1, modality="text", text="world", embedding=[0.5, 0.6])],
            )
        ],
    )
    assert replaced.stale_media_keys == ["stale-media"]
    assert replaced.skipped_document_ids == []
    assert any("INSERT INTO" in sql and "kb_passages" in sql for sql, _ in conn.executemany_calls)


@pytest.mark.asyncio
async def test_oracle_store_search_tags_stats_and_timeseries():
    conn = _FakeConn()
    backend = _FakeBackend(conn)
    store = OracleKnowledgeStore(AsyncMock(return_value=backend))

    conn.fetchval_responses.append(None)
    with pytest.raises(KnowledgeBankNotFound):
        await store.search("missing-kb", [1.0, 0.0], top_k=5, filters=SearchFilters())

    doc_row = _sample_doc_row("doc-1")
    passage_row = {
        "idx": 0,
        "modality": "text",
        "text": "Oracle 26ai vector search",
        "page": None,
        "start_ms": None,
        "end_ms": None,
        "media_key": None,
        **{k: v for k, v in doc_row.items()},
        "score": 0.92,
    }
    conn.fetchval_responses.extend([1, 10, 2])
    conn.fetch_responses.extend(
        [
            [{"document_id": "doc-1", "idx": 0, "arm_score": 0.92}],
            [],
            [{"document_id": "doc-1", "idx": 0, "arm_score": 50.0}],
            [passage_row],
            [],
            [
                {
                    "document_id": "doc-1",
                    "idx": 1,
                    "modality": "text",
                    "text": "neighbor",
                    "page": None,
                    "start_ms": None,
                    "end_ms": None,
                    "media_key": None,
                }
            ],
        ]
    )
    trace = StoreSearchTrace()
    res = await store.search(
        "kb-1",
        [1.0, 0.0],
        top_k=5,
        filters=SearchFilters(tags=["alpha"], tag_groups=[TagGroupLeaf(tags=["beta"], match="all_strict")]),
        query_text="oracle",
        around=1,
        trace=trace,
    )
    assert len(res.hits) == 1
    assert res.hits[0].document.document_id == "doc-1"
    assert ("doc-1", 1) in res.nearby
    assert trace.candidates == 1
    vec_arm_sql = next(sql for sql, _ in conn.executed if "FETCH EXACT FIRST" in sql)
    assert "$[*]?(@ == $t)" in vec_arm_sql
    assert "= '[]'" not in vec_arm_sql

    # Exact tag matching: non-empty checks count, empty [] filters to untagged
    conn.fetchval_responses.append(1)
    conn.fetch_responses.extend([[], []])
    await store.search(
        "kb-1",
        [1.0, 0.0],
        top_k=5,
        filters=SearchFilters(tags=["alpha"], tags_match="exact"),
    )
    exact_sql = next(sql for sql, _ in reversed(conn.executed) if "FETCH EXACT FIRST" in sql)
    assert "COUNT(DISTINCT jt.tag)" in exact_sql

    conn.fetchval_responses.append(1)
    conn.fetch_responses.extend([[], []])
    await store.search(
        "kb-1",
        [1.0, 0.0],
        top_k=5,
        filters=SearchFilters(tags=[], tags_match="exact"),
    )
    empty_exact_sql = next(sql for sql, _ in reversed(conn.executed) if "FETCH EXACT FIRST" in sql)
    assert "NOT JSON_EXISTS(d.tags, '$[*]')" in empty_exact_sql

    conn.fetchval_responses.append(1)
    conn.fetch_responses.append([{"tag": "alpha", "documents": 2}])
    tag_page = await store.list_tags("kb-1", pattern="alp*", limit=10, offset=0)
    assert tag_page.total == 1
    assert tag_page.tags[0].tag == "alpha"

    now = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
    conn.fetch_responses.append(
        [
            {
                "bank_id": "kb-1",
                "media_type": "text",
                "status": "ready",
                "documents": 2,
                "passages": 4,
                "content_tokens": 100,
                "last_at": now,
            }
        ]
    )
    stats = await store.bank_stats(["kb-1"])
    assert stats["kb-1"].document_count == 2
    assert stats["kb-1"].content_tokens == 100

    naive_now = datetime(2026, 11, 22, 12, 0)
    conn.fetch_responses.append([{"t": naive_now, "documents": 2, "passages": 4}])
    buckets = await store.timeseries("kb-1", since=now, trunc="day")
    assert len(buckets) == 1 and buckets[0].passages == 4
    assert buckets[0].time.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_knowledge_service_oracle_create_list_and_overrides():
    conn = _FakeConn()
    backend = _FakeBackend(conn)
    engine = SimpleNamespace(
        _db_backend=backend,
        _get_backend=AsyncMock(return_value=backend),
        _operation_validator=None,
        _authenticate_tenant=AsyncMock(),
    )
    svc = KnowledgeService(engine)  # ty: ignore[invalid-argument-type]
    assert isinstance(svc.store, OracleKnowledgeStore)

    conn.fetchval_responses.extend([None])
    conn.fetchrow_responses.append({"bank_id": "kb-ora", "created_at": datetime.now(timezone.utc)})
    conn.fetch_responses.append([])
    with (
        patch.object(svc, "_ready", AsyncMock()),
        patch("hindsight_api.knowledge.service.bank_aliases.refuse_alias_as_bank_id_on_conn", AsyncMock()),
    ):
        info = await svc.create_bank("kb-ora", MagicMock())
    assert info.bank_id == "kb-ora"
    assert any("MERGE INTO" in sql for sql, _ in conn.executed)

    conn.fetch_responses.extend(
        [
            [{"bank_id": "kb-ora", "created_at": datetime.now(timezone.utc), "total": 1}],
            [],
        ]
    )
    with patch.object(svc, "_ready", AsyncMock()):
        page = await svc.list_banks(MagicMock(), limit=10, offset=0, q="ora")
    assert page.total == 1
    assert any("INSTR(LOWER(bank_id)" in sql for sql, _ in conn.executed)

    conn.fetchval_responses.append(json.dumps({"unrelated_key": "keep", "kb_text_passage_tokens": 256}))
    with patch.object(svc, "_open_bank", AsyncMock()):
        await svc.reset_bank_config("kb-ora", MagicMock())
    update_sql, update_args = next((s, a) for s, a in conn.executed if s.startswith("UPDATE ") and "SET config =" in s)
    assert "SYSTIMESTAMP" in update_sql
    saved_cfg = json.loads(update_args[0])
    assert saved_cfg["unrelated_key"] == "keep"
    assert "kb_text_passage_tokens" not in saved_cfg


@pytest.mark.asyncio
async def test_oracle_uuid_shaped_bank_id_and_tags_stay_str():
    from hindsight_api.engine.db.oracle import OracleLiteralStr, _convert_arg
    from hindsight_api.knowledge.store_oracle import _oracle_tags_predicate

    uuid_bank = "11111111-2222-3333-4444-555555555555"
    uuid_tag = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    conn = _FakeConn()
    conn.fetchval_responses.append(0)
    conn.fetch_responses.append([])
    store = OracleKnowledgeStore(AsyncMock(return_value=_FakeBackend(conn)))

    await store.list_documents(uuid_bank, limit=10, offset=0, tags=[uuid_tag])
    count_sql, count_args = next((s, a) for s, a in conn.executed if s.startswith("SELECT COUNT(*)"))
    assert "JSON_EXISTS(tags" in count_sql
    assert all(isinstance(arg, OracleLiteralStr) and isinstance(_convert_arg(arg), str) for arg in count_args)

    params: list[object] = []
    pred = _oracle_tags_predicate([uuid_tag], params, "d.", "exact")
    assert "JSON_EXISTS" in pred
    assert len(params) == 2
    assert isinstance(params[0], OracleLiteralStr)
    assert isinstance(_convert_arg(params[0]), str)


@pytest.mark.asyncio
async def test_oracle_search_batches_multi_term_df_with_indexed_union_and_accum():
    conn = _FakeConn()
    store = OracleKnowledgeStore(AsyncMock(return_value=_FakeBackend(conn)))
    doc_row = _sample_doc_row("doc-1")
    passage_row = {
        "idx": 0,
        "modality": "text",
        "text": "Oracle vector search",
        "page": None,
        "start_ms": None,
        "end_ms": None,
        "media_key": None,
        **{k: v for k, v in doc_row.items()},
        "score": 0.91,
    }
    conn.fetchval_responses.extend([1, 20])
    conn.fetch_responses.extend(
        [
            [{"document_id": "doc-1", "idx": 0, "arm_score": 0.91}],
            [],
            [{"term_idx": 0, "df": 2}, {"term_idx": 1, "df": 3}],
            [{"document_id": "doc-1", "idx": 0, "arm_score": 42.0}],
            [passage_row],
            [],
        ]
    )
    res = await store.search(
        "kb-1",
        [1.0, 0.0],
        top_k=5,
        filters=SearchFilters(),
        query_text="oracle vector",
    )
    assert len(res.hits) == 1
    df_sql, df_args = next((s, a) for s, a in conn.executed if "AS term_idx" in s)
    assert "UNION ALL" in df_sql
    assert "WHERE bank_id = $1 AND CONTAINS(text, $2, 1) > 0" in df_sql
    assert "WHERE bank_id = $1 AND CONTAINS(text, $3, 2) > 0" in df_sql
    assert df_args[1:] == ("{oracle}", "{vector}")
    kw_sql, kw_args = next((s, a) for s, a in conn.executed if "ORDER BY SCORE(1) DESC" in s)
    assert "CONTAINS(x.text," in kw_sql
    assert " ACCUM " in kw_args[-1]
    assert " OR " not in kw_args[-1]


@pytest.mark.asyncio
async def test_oracle_store_search_skips_documents_join_when_unfiltered():
    conn = _FakeConn()
    store = OracleKnowledgeStore(AsyncMock(return_value=_FakeBackend(conn)))

    conn.fetchval_responses.extend([1, 5, 1])
    conn.fetch_responses.extend([[], [], [], []])
    await store.search(
        "kb-unfiltered",
        [1.0, 0.0],
        top_k=5,
        filters=SearchFilters(document_ids=["doc-1"]),
        query_text="oracle",
    )
    arm_sqls = [sql for sql, _ in conn.executed if "FETCH EXACT FIRST" in sql or "FETCH FIRST" in sql]
    assert len(arm_sqls) == 3
    for sql in arm_sqls:
        assert "JOIN " not in sql

    conn2 = _FakeConn()
    store2 = OracleKnowledgeStore(AsyncMock(return_value=_FakeBackend(conn2)))
    conn2.fetchval_responses.append(1)
    conn2.fetch_responses.extend([[], []])
    await store2.search(
        "kb-filtered",
        [1.0, 0.0],
        top_k=5,
        filters=SearchFilters(tags=["alpha"]),
    )
    filtered_arms = [sql for sql, _ in conn2.executed if "FETCH EXACT FIRST" in sql or "FETCH FIRST" in sql]
    assert len(filtered_arms) == 2
    for sql in filtered_arms:
        assert "JOIN " in sql and "kb_documents" in sql


@pytest.mark.asyncio
async def test_oracle_bank_kind_and_service_preserve_uuid_bank_id_and_delegate_ops():
    from hindsight_api.engine.db.oracle import OracleLiteralStr, _convert_arg
    from hindsight_api.engine.retain import bank_utils

    uuid_bank = "11111111-2222-3333-4444-555555555555"
    conn = _FakeConn()
    conn.fetchval_responses.append("knowledge")
    kind = await bank_utils.bank_kind_on_conn(conn, uuid_bank)  # ty: ignore[invalid-argument-type]
    assert kind == "knowledge"
    _, args = conn.executed[0]
    assert isinstance(args[0], OracleLiteralStr)
    assert isinstance(_convert_arg(args[0]), str)

    backend = _FakeBackend(conn)
    engine = SimpleNamespace(
        _db_backend=backend,
        _get_backend=AsyncMock(return_value=backend),
        _operation_validator=None,
        _authenticate_tenant=AsyncMock(),
        _file_storage=SimpleNamespace(delete_prefix=AsyncMock()),
        delete_bank=AsyncMock(),
        list_operations=AsyncMock(return_value={"total": 0, "operations": []}),
        get_operation_status=AsyncMock(return_value={"status": "completed"}),
        cancel_operation=AsyncMock(return_value={"success": True}),
        retry_operation=AsyncMock(return_value={"success": True}),
        delete_operation=AsyncMock(return_value={"success": True}),
    )
    svc = KnowledgeService(engine)  # ty: ignore[invalid-argument-type]

    conn.fetchval_responses.append(json.dumps({"kb_text_passage_tokens": 128}))
    overrides = await svc._overrides(uuid_bank)
    assert overrides.text_passage_tokens == 128
    _, ov_args = conn.executed[-1]
    assert isinstance(ov_args[0], OracleLiteralStr)

    rc = MagicMock()
    with patch.object(svc, "_open_bank", AsyncMock()):
        await svc.list_operations(uuid_bank, rc, status=None, task_type=None, limit=10, offset=0, exclude_parents=False)
        await svc.get_operation(uuid_bank, "op-1", rc, include_payload=False)
        await svc.cancel_operation(uuid_bank, "op-1", rc)
        await svc.retry_operation(uuid_bank, "op-1", rc)
        await svc.delete_operation(uuid_bank, "op-1", rc)

    assert engine.list_operations.await_args.kwargs["_knowledge_bank"] is True
    assert isinstance(engine.list_operations.await_args.args[0], OracleLiteralStr)
    assert engine.get_operation_status.await_args.kwargs["_knowledge_bank"] is True
    assert isinstance(engine.get_operation_status.await_args.args[0], OracleLiteralStr)
    assert engine.cancel_operation.await_args.kwargs["_knowledge_bank"] is True
    assert isinstance(engine.cancel_operation.await_args.args[0], OracleLiteralStr)
    assert engine.retry_operation.await_args.kwargs["_knowledge_bank"] is True
    assert isinstance(engine.retry_operation.await_args.args[0], OracleLiteralStr)
    assert engine.delete_operation.await_args.kwargs["_knowledge_bank"] is True
    assert isinstance(engine.delete_operation.await_args.args[0], OracleLiteralStr)

    from hindsight_api.engine import bank_aliases

    conn.fetchval_responses.append(None)
    await bank_aliases.refuse_alias_as_bank_id_on_conn(conn, uuid_bank)  # ty: ignore[invalid-argument-type]
    _, alias_args = conn.executed[-1]
    assert isinstance(alias_args[0], OracleLiteralStr)


@pytest.mark.asyncio
async def test_oracle_store_search_breaks_equal_score_ties_deterministically():
    conn = _FakeConn()
    conn.fetchval_responses.append(1)  # bank exists
    conn.fetch_responses.extend(
        [
            [
                {"document_id": "doc-b", "idx": 1, "arm_score": 0.9},
                {"document_id": "doc-a", "idx": 2, "arm_score": 0.9},
                {"document_id": "doc-a", "idx": 0, "arm_score": 0.9},
            ],
            [],  # extra vectors arm
            [  # hydrated rows returned out of order
                {
                    "document_id": "doc-b",
                    "idx": 1,
                    "modality": "text",
                    "text": "same content",
                    "page": None,
                    "start_ms": None,
                    "end_ms": None,
                    "media_key": None,
                    "filename": "b.txt",
                    "content_type": "text/plain",
                    "media_type": "text",
                    "language": None,
                    "size_bytes": 12,
                    "storage_key": None,
                    "content_hash": "hb",
                    "status": "ready",
                    "error": None,
                    "passage_count": 1,
                    "tags": "[]",
                    "source_document_id": None,
                    "content_tokens": 2,
                    "created_at": datetime(2026, 11, 22, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 11, 22, tzinfo=timezone.utc),
                    "score": 0.9,
                },
                {
                    "document_id": "doc-a",
                    "idx": 2,
                    "modality": "text",
                    "text": "same content",
                    "page": None,
                    "start_ms": None,
                    "end_ms": None,
                    "media_key": None,
                    "filename": "a.txt",
                    "content_type": "text/plain",
                    "media_type": "text",
                    "language": None,
                    "size_bytes": 12,
                    "storage_key": None,
                    "content_hash": "ha",
                    "status": "ready",
                    "error": None,
                    "passage_count": 2,
                    "tags": "[]",
                    "source_document_id": None,
                    "content_tokens": 2,
                    "created_at": datetime(2026, 11, 22, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 11, 22, tzinfo=timezone.utc),
                    "score": 0.9,
                },
                {
                    "document_id": "doc-a",
                    "idx": 0,
                    "modality": "text",
                    "text": "same content",
                    "page": None,
                    "start_ms": None,
                    "end_ms": None,
                    "media_key": None,
                    "filename": "a.txt",
                    "content_type": "text/plain",
                    "media_type": "text",
                    "language": None,
                    "size_bytes": 12,
                    "storage_key": None,
                    "content_hash": "ha",
                    "status": "ready",
                    "error": None,
                    "passage_count": 2,
                    "tags": "[]",
                    "source_document_id": None,
                    "content_tokens": 2,
                    "created_at": datetime(2026, 11, 22, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 11, 22, tzinfo=timezone.utc),
                    "score": 0.9,
                },
            ],
            [],  # role scores
        ]
    )
    store = OracleKnowledgeStore(AsyncMock(return_value=_FakeBackend(conn)))  # ty: ignore[invalid-argument-type]
    result = await store.search(
        "bank-1",
        query_text="same",
        vector=[0.1, 0.2],
        top_k=3,
        around=0,
        filters=SearchFilters(),
    )
    assert [(h.document.document_id, h.passage.index) for h in result.hits] == [
        ("doc-a", 0),
        ("doc-a", 2),
        ("doc-b", 1),
    ]


@pytest.mark.asyncio
async def test_oracle_store_search_arms_include_sql_tie_breakers():
    conn = _FakeConn()
    conn.fetchval_responses.extend([1, 10])  # bank exists, total_n
    conn.fetch_responses.extend(
        [
            [{"document_id": "doc-a", "idx": 0, "arm_score": 0.9}],  # Arm 1
            [],  # Arm 2
            [{"term_idx": 0, "df": 1}, {"term_idx": 1, "df": 1}],  # batched df probe
            [{"document_id": "doc-a", "idx": 0, "arm_score": 50.0}],  # Arm 3
            [_sample_doc_row("doc-a") | {"idx": 0, "modality": "text", "text": "alpha beta", "page": None, "start_ms": None, "end_ms": None, "media_key": None, "language": None, "score": 0.9}],
            [],  # role scores
        ]
    )
    store = OracleKnowledgeStore(AsyncMock(return_value=_FakeBackend(conn)))  # ty: ignore[invalid-argument-type]
    await store.search("bank-1", query_text="alpha beta", vector=[0.1, 0.2], top_k=3, around=0, filters=SearchFilters())

    arm1_sql = next(sql for sql, _ in conn.executed if "FETCH EXACT FIRST" in sql)
    arm2_sql = next(sql for sql, _ in conn.executed if "GROUP BY x.document_id, x.idx" in sql)
    arm3_sql = next(sql for sql, _ in conn.executed if "ORDER BY SCORE(1) DESC" in sql)
    assert "ORDER BY VECTOR_DISTANCE(x.embedding, $1, COSINE), x.document_id ASC, x.idx ASC FETCH EXACT FIRST" in arm1_sql
    assert "ORDER BY MAX(1 - VECTOR_DISTANCE(x.embedding, $1, COSINE)) DESC, x.document_id ASC, x.idx ASC FETCH FIRST" in arm2_sql
    assert "ORDER BY SCORE(1) DESC, x.document_id ASC, x.idx ASC FETCH FIRST" in arm3_sql


@pytest.mark.asyncio
async def test_uuid_knowledge_bank_delete_and_ingest_parent_rollup(monkeypatch):
    import uuid

    from hindsight_api.engine.db.ops_oracle import OracleOps
    from hindsight_api.engine.memory_engine import MemoryEngine
    from hindsight_api.engine.operation_metadata import KnowledgeIngestParentMetadata
    from hindsight_api.models import RequestContext
    from hindsight_api.worker.poller import WorkerPoller

    uuid_bank = "11111111-2222-3333-4444-555555555555"
    parent_op_id = str(uuid.uuid4())
    child_op_id = str(uuid.uuid4())

    # 1) KnowledgeIngestParentMetadata carries is_parent=True for get_operation_status self-healing.
    meta = KnowledgeIngestParentMetadata(document_ids=["doc-1"], num_sub_batches=1).to_dict()
    assert meta["is_parent"] is True

    # 2) KnowledgeService.delete_bank passes OracleLiteralStr(bank_id) to engine.delete_bank, and
    #    MemoryEngine.delete_bank preserves OracleLiteralStr on banks & bank_aliases queries.
    conn = _FakeConn()
    conn.fetchrow_responses.append({"bank_id": uuid_bank, "internal_id": str(uuid.uuid4())})
    conn.fetch_responses.append([{"alias": "alias-1"}])
    conn.fetchval_responses.append(str(uuid.uuid4()))

    def _cm_backend(c: Any, ops_obj: Any = None) -> Any:
        @asynccontextmanager
        async def _acquire():
            yield c

        return SimpleNamespace(_wraps_backend=True, acquire=_acquire, ops=ops_obj)

    backend = _cm_backend(conn, SimpleNamespace(drop_bank_vector_indexes=AsyncMock()))
    del_store = SimpleNamespace(
        store_owned_for=lambda _b: True,
        count_bank_contents=AsyncMock(return_value=SimpleNamespace(memory_units=0, entities=0, documents=0)),
        purge_bank_rows=AsyncMock(return_value=[]),
        drop_bank_storage=AsyncMock(),
    )

    eng = MemoryEngine.__new__(MemoryEngine)
    eng._initialized = True
    eng._tenant_extension = None
    eng._operation_validator = None
    eng._active_operation_canceller = None
    eng._active_bank_operations_canceller = None
    eng._finish_bank_delete = None
    eng._bank_stats_cache = SimpleNamespace(invalidate=AsyncMock())
    eng._delete_files_quietly = AsyncMock()
    eng._submit_vector_index_maintenance_quietly = AsyncMock()
    eng._authenticate_tenant = AsyncMock(return_value=None)
    eng._refuse_bank_wide_if_scoped = AsyncMock(return_value=None)
    eng._backend = backend
    eng._get_backend = AsyncMock(return_value=backend)
    eng._config_resolver = SimpleNamespace(resolve_full_config=AsyncMock(return_value=SimpleNamespace(enable_auto_consolidation=False)))

    with patch("hindsight_api.engine.memories.get_memories", return_value=del_store):
        await eng.delete_bank(uuid_bank, request_context=RequestContext(), _knowledge_bank=True)

    bank_queries = [
        args[0]
        for sql, args in conn.executed
        if ("banks" in sql or "bank_aliases" in sql) and "WHERE bank_id = $1" in sql
    ]
    assert len(bank_queries) == 3
    assert all(isinstance(arg, OracleLiteralStr) and isinstance(_convert_arg(arg), str) for arg in bank_queries)

    svc_engine = SimpleNamespace(
        _db_backend=SimpleNamespace(backend_type="oracle"),
        _get_backend=AsyncMock(return_value=_cm_backend(_FakeConn())),
        delete_bank=AsyncMock(return_value={"bank_deleted": True}),
        _file_storage=SimpleNamespace(delete_prefix=AsyncMock()),
    )
    svc = KnowledgeService(svc_engine)  # type: ignore[arg-type]
    monkeypatch.setattr(svc, "_open_bank", AsyncMock())
    monkeypatch.setattr(
        svc,
        "_store",
        SimpleNamespace(
            bank_stats=AsyncMock(return_value={}),
            delete_some_documents=AsyncMock(return_value=0),
            list_documents=AsyncMock(return_value=SimpleNamespace(documents=[])),
            bank_deleted=AsyncMock(),
        ),
    )
    monkeypatch.setattr(svc, "_on_delete_complete", AsyncMock())
    await svc.delete_bank(uuid_bank, RequestContext())
    assert isinstance(svc_engine.delete_bank.await_args.args[0], OracleLiteralStr)

    # 3) Engine & Poller _maybe_update_parent_operation + _reconcile_orphaned_parents + get_operation_status
    #    wrap plain str row['bank_id'] in OracleLiteralStr(bank_id) and discovery includes 'knowledge_ingest'.
    ops = OracleOps()
    ops_conn = _FakeConn()
    await ops.fetch_reconcilable_batch_parents(ops_conn, "async_operations")  # type: ignore[arg-type]
    assert "IN ('batch_retain', 'knowledge_ingest')" in ops_conn.executed[-1][0]

    rollup_conn = _FakeConn()
    rollup_conn.parse_json = lambda v: json.loads(v) if isinstance(v, str) else v  # type: ignore[attr-defined]
    rollup_conn.fetchrow_responses.extend([
        {"bank_id": uuid_bank, "result_metadata": {"parent_operation_id": parent_op_id}},
        {"operation_id": parent_op_id},
    ])
    rollup_conn.fetch_responses.append([{"status": "completed", "error_message": None, "result_metadata": {}}])
    await eng._maybe_update_parent_operation(child_op_id, rollup_conn)
    assert isinstance(rollup_conn.executed[1][1][1], OracleLiteralStr)
    assert isinstance(rollup_conn.executed[2][1][0], OracleLiteralStr)

    poller = WorkerPoller(backend=backend, worker_id="w1", executor=AsyncMock())
    p_conn = _FakeConn()
    p_conn.fetchrow_responses.extend([
        {"bank_id": uuid_bank, "result_metadata": {"parent_operation_id": parent_op_id}},
        {"operation_id": parent_op_id},
    ])
    p_conn.fetch_responses.append([{"status": "completed", "error_message": None}])
    await poller._maybe_update_parent_operation(child_op_id, None, p_conn)
    assert isinstance(p_conn.executed[1][1][1], OracleLiteralStr)
    assert isinstance(p_conn.executed[2][1][0], OracleLiteralStr)

    rec_conn = _FakeConn()
    rec_conn.fetchrow_responses.append({"status": "pending"})
    rec_conn.fetch_responses.append([{"status": "completed", "error_message": None}])
    rec_backend = _cm_backend(
        rec_conn,
        SimpleNamespace(
            fetch_reconcilable_batch_parents=AsyncMock(return_value=[{"operation_id": uuid.UUID(parent_op_id), "bank_id": uuid_bank}])
        ),
    )
    rec_poller = WorkerPoller(backend=rec_backend, worker_id="w1", executor=AsyncMock())  # type: ignore[arg-type]
    assert await rec_poller._reconcile_orphaned_parents(None) == 1
    siblings_call = next((args for sql, args in rec_conn.executed if "result_metadata::jsonb @>" in sql), None)
    assert siblings_call is not None and isinstance(siblings_call[0], OracleLiteralStr)

    status_conn = _FakeConn()
    status_conn.parse_json = lambda v: json.loads(v) if isinstance(v, str) else v  # type: ignore[attr-defined]
    status_conn.fetchrow_responses.append({
        "operation_id": parent_op_id,
        "operation_type": "knowledge_ingest",
        "created_at": datetime(2026, 11, 22, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 11, 22, tzinfo=timezone.utc),
        "completed_at": None,
        "status": "pending",
        "error_message": None,
        "result_metadata": meta,
        "retry_count": 0,
        "next_retry_at": None,
    })
    status_conn.fetch_responses.append([
        {"operation_id": child_op_id, "status": "completed", "error_message": None, "result_metadata": {"sub_batch_index": 0}}
    ])
    eng._get_backend = AsyncMock(return_value=_cm_backend(status_conn))
    status = await eng.get_operation_status(
        uuid_bank, parent_op_id, request_context=RequestContext(), _knowledge_bank=True
    )
    assert status["status"] == "completed"
    assert isinstance(status_conn.executed[0][1][1], OracleLiteralStr)
    assert isinstance(status_conn.executed[1][1][0], OracleLiteralStr)


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_oracle_small_bank_finds_its_passages_next_to_a_big_one(oracle_db_url, monkeypatch):
    """Live Oracle 26ai check for #6731: a small bank finds its passages after a 3,000-row big bank."""
    import uuid

    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.db.oracle import OracleBackend
    from hindsight_api.engine.schema import fq_table
    from hindsight_api.knowledge.service import KIND

    # fq_table() leaves table names bare only when the configured backend is Oracle.
    monkeypatch.setenv("HINDSIGHT_API_DATABASE_BACKEND", "oracle")
    clear_config_cache()

    backend = OracleBackend()
    await backend.initialize(oracle_db_url, min_size=1, max_size=4)

    big = f"kb-big-{uuid.uuid4().hex[:8]}"
    small = f"kb-small-{uuid.uuid4().hex[:8]}"
    dim = 384
    big_vec = array.array("f", [1.0] + [0.0] * (dim - 1))
    small_vecs = [
        [0.0, 1.0] + [0.0] * (dim - 2),
        [0.0, 0.0, 1.0] + [0.0] * (dim - 3),
        [0.0, 0.0, 0.0, 1.0] + [0.0] * (dim - 4),
    ]
    store = OracleKnowledgeStore(AsyncMock(return_value=backend))
    try:
        async with backend.acquire() as conn:
            async with conn.transaction():
                for b in (big, small):
                    await conn.execute(
                        f"INSERT INTO {fq_table('banks')} (bank_id, name, disposition, mission, internal_id, kind) "
                        "VALUES ($1, $2, $3, NULL, $4, $5)",
                        OracleLiteralStr(b),
                        OracleLiteralStr(b),
                        "{}",
                        uuid.uuid4(),
                        KIND,
                    )
                await conn.execute(
                    f"INSERT INTO {fq_table('kb_documents')} (bank_id, document_id, content_type, media_type, content_hash, passage_count) "
                    "VALUES ($1, $2, 'text/plain', 'text', 'h', 3000)",
                    OracleLiteralStr(big),
                    OracleLiteralStr("d"),
                )
                rows = [
                    (OracleLiteralStr(big), OracleLiteralStr("d"), i, "text", f"big filler {i} red", big_vec)
                    for i in range(1, 3001)
                ]
                for idx in range(0, len(rows), 500):
                    await conn.executemany(
                        f"INSERT INTO {fq_table('kb_passages')} (bank_id, document_id, idx, modality, text, embedding) "
                        "VALUES ($1, $2, $3, $4, $5, $6)",
                        rows[idx : idx + 500],
                    )
                for n, vec in enumerate(small_vecs):
                    doc_id = f"s{n}"
                    await conn.execute(
                        f"INSERT INTO {fq_table('kb_documents')} (bank_id, document_id, content_type, media_type, content_hash, passage_count) "
                        "VALUES ($1, $2, 'text/plain', 'text', 'h', 1)",
                        OracleLiteralStr(small),
                        OracleLiteralStr(doc_id),
                    )
                    await conn.execute(
                        f"INSERT INTO {fq_table('kb_passages')} (bank_id, document_id, idx, modality, text, embedding) "
                        "VALUES ($1, $2, 0, 'text', $3, $4)",
                        OracleLiteralStr(small),
                        OracleLiteralStr(doc_id),
                        f"small passage {n} red",
                        array.array("f", vec),
                    )
            try:
                await conn.execute("BEGIN DBMS_STATS.GATHER_TABLE_STATS(USER, 'KB_PASSAGES'); END;")
                await conn.execute("BEGIN DBMS_STATS.GATHER_TABLE_STATS(USER, 'KB_DOCUMENTS'); END;")
            except Exception:
                pass

        # Warm the shared cursor on the big bank first, then search the small bank.
        big_res = await store.search(big, list(big_vec), top_k=10, filters=SearchFilters(), query_text="red")
        assert len(big_res.hits) == 10

        small_vec_res = await store.search(small, small_vecs[0], top_k=10, filters=SearchFilters())
        assert {h.document.document_id for h in small_vec_res.hits} == {"s0", "s1", "s2"}

        small_hybrid_res = await store.search(small, small_vecs[0], top_k=10, filters=SearchFilters(), query_text="red")
        assert {h.document.document_id for h in small_hybrid_res.hits} == {"s0", "s1", "s2"}
    finally:
        try:
            async with backend.acquire() as conn, conn.transaction():
                for b in (big, small):
                    await conn.execute(f"DELETE FROM {fq_table('banks')} WHERE bank_id = $1", OracleLiteralStr(b))
        finally:
            await backend.shutdown()
            clear_config_cache()


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_oracle_knowledge_bank_operations_and_mcp_e2e(oracle_db_url, monkeypatch):
    """Live Oracle 26ai E2E for #6791 (KB operations & /banks refusal), #6801 (/mcp/kb/), and auth boundaries."""
    import os
    import uuid

    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from hindsight_api import MemoryEngine
    from hindsight_api.api import create_app
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.cross_encoder import CrossEncoderModel
    from hindsight_api.engine.embeddings import Embeddings
    from hindsight_api.extensions.builtin.tenant import ApiKeyTenantExtension
    from hindsight_api.extensions.operation_validator import (
        KnowledgeBankReadContext,
        OperationValidatorExtension,
        ValidationResult,
    )
    from hindsight_api.knowledge.service import get_knowledge_service
    from hindsight_api.mcp_tools import _KNOWLEDGE_TOOLS

    class _DetEmbeddings(Embeddings):
        @property
        def provider_name(self) -> str:
            return "mock"

        @property
        def dimension(self) -> int:
            return 384

        async def initialize(self) -> None:
            return None

        async def encode(self, texts: list[str]) -> list[list[float]]:
            out = []
            for t in texts:
                v = [0.0] * 384
                v[0] = 1.0 if "password" in t.lower() or "vault" in t.lower() else 0.1
                v[1] = 1.0 if "red" in t.lower() else 0.1
                out.append(v)
            return out

    class _NoopCrossEncoder(CrossEncoderModel):
        @property
        def provider_name(self) -> str:
            return "mock"

        async def initialize(self) -> None:
            return None

        async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.5] * len(pairs)

    class _DetKBEmbedder:
        provider = "fake"
        model = "fake"
        dimension = 384
        modalities = frozenset({"text"})

        def __init__(self, emb: _DetEmbeddings) -> None:
            self._emb = emb

        async def embed_documents(self, items: list[Any]) -> list[list[float]]:
            return await self._emb.encode([getattr(i, "text", None) or "" for i in items])

        async def embed_query(
            self, text: str | None, image: bytes | None = None, video: bytes | None = None
        ) -> list[float]:
            return (await self._emb.encode([text or ""]))[0]

    class _HideForbiddenKB(OperationValidatorExtension):
        def __init__(self, forbidden_id: str) -> None:
            super().__init__({})
            self.forbidden_id = forbidden_id

        async def validate_retain(self, ctx: Any) -> ValidationResult:
            return ValidationResult.accept()

        async def validate_recall(self, ctx: Any) -> ValidationResult:
            return ValidationResult.accept()

        async def validate_reflect(self, ctx: Any) -> ValidationResult:
            return ValidationResult.accept()

        async def validate_knowledge_bank_read(self, ctx: KnowledgeBankReadContext) -> ValidationResult:
            if ctx.bank_id == self.forbidden_id:
                return ValidationResult.reject("forbidden kb", status_code=403)
            return ValidationResult.accept()

    monkeypatch.setenv("HINDSIGHT_API_DATABASE_BACKEND", "oracle")
    clear_config_cache()

    emb = _DetEmbeddings()
    mem = MemoryEngine(
        db_url=oracle_db_url,
        memory_llm_provider="mock",
        embeddings=emb,
        cross_encoder=_NoopCrossEncoder(),
        run_migrations=False,
        pool_min_size=1,
        pool_max_size=4,
    )
    await mem.initialize()

    kb_id = f"kb-ora-mcp-{uuid.uuid4().hex[:8]}"
    forbidden_kb = f"kb-forbid-{uuid.uuid4().hex[:8]}"
    mem_id = f"mem-ora-{uuid.uuid4().hex[:8]}"
    try:
        svc = get_knowledge_service(mem)
        monkeypatch.setattr(svc, "_embedder", _DetKBEmbedder(emb))

        mem._tenant_extension = ApiKeyTenantExtension({"api_key": "ora-secret-key"})
        mem._operation_validator = _HideForbiddenKB(forbidden_kb)

        app = create_app(mem, mcp_api_enabled=True, initialize_memory=False)
        async with app.router.lifespan_context(app):
            headers = {"Authorization": "Bearer ora-secret-key"}
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
            ) as http:
                # 1. Auth boundary check: wrong bearer token -> 401 and no credentials leaked
                unauth = await http.post(
                    f"/mcp/kb/{kb_id}/",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    headers={"Authorization": "Bearer wrong-key", "Accept": "application/json, text/event-stream"},
                )
                assert unauth.status_code == 401
                assert "oracle" not in unauth.text.lower() and os.getenv("ORACLE_TEST_DSN", "x") not in unauth.text

                # 2. Multi-bank MCP: create, ingest, cross-kind refusal, validator 403 boundary
                async with streamable_http_client("http://test/mcp/", http_client=http) as (r, w, _):
                    async with ClientSession(r, w) as s:
                        await s.initialize()
                        names = {t.name for t in (await s.list_tools()).tools}
                        assert _KNOWLEDGE_TOOLS <= names

                        created = json.loads(
                            (await s.call_tool("create_knowledge_bank", {"bank_id": kb_id})).content[0].text
                        )
                        assert created["bank_id"] == kb_id
                        ingested = json.loads(
                            (
                                await s.call_tool(
                                    "ingest_knowledge_text",
                                    {
                                        "bank_id": kb_id,
                                        "document_id": "pw",
                                        "text": "The vault password is kept in the red box.",
                                    },
                                )
                            )
                            .content[0]
                            .text
                        )
                        assert ingested["document_ids"] == ["pw"]
                        assert "error" in json.loads(
                            (await s.call_tool("recall", {"bank_id": kb_id, "query": "vault"})).content[0].text
                        )
                        forbid_res = json.loads(
                            (await s.call_tool("search_knowledge_bank", {"bank_id": forbidden_kb, "query": "x"}))
                            .content[0]
                            .text
                        )
                        assert "forbidden kb" in forbid_res["error"]

                # 3. HTTP #6791 operations separation
                assert (await http.get(f"/v1/default/knowledge-banks/{kb_id}/operations")).status_code == 200
                refused_http = await http.get(f"/v1/default/banks/{kb_id}/operations")
                assert (
                    refused_http.status_code == 400
                    and f"/v1/default/knowledge-banks/{kb_id}" in refused_http.json()["detail"]
                )

                # 4. Single-KB MCP (/mcp/kb/{kb_id}/): search, get, delete doc, delete bank
                async with streamable_http_client(f"http://test/mcp/kb/{kb_id}/", http_client=http) as (r, w, _):
                    async with ClientSession(r, w) as s:
                        await s.initialize()
                        kb_names = {t.name for t in (await s.list_tools()).tools}
                        assert kb_names == _KNOWLEDGE_TOOLS - {"list_knowledge_banks", "create_knowledge_bank"}
                        found = (await s.call_tool("search_knowledge_bank", {"query": "password"})).structuredContent
                        assert found is not None
                        assert found["results"][0]["document_id"] == "pw"
                        doc = (await s.call_tool("get_knowledge_document", {"document_id": "pw"})).structuredContent
                        assert doc is not None
                        assert doc["status"] == "ready"
                        assert (
                            await s.call_tool("delete_knowledge_document", {"document_id": "pw"})
                        ).structuredContent == {"deleted": True}
                        assert (await s.call_tool("delete_knowledge_bank", {})).structuredContent == {"deleted": True}

                # 5. Memory bank still works on /banks/{mem_id}/operations
                assert (await http.put(f"/v1/default/banks/{mem_id}", json={})).is_success
                assert (await http.get(f"/v1/default/banks/{mem_id}/operations")).status_code == 200
                await http.delete(f"/v1/default/banks/{mem_id}")
    finally:
        await mem.close()
        clear_config_cache()
