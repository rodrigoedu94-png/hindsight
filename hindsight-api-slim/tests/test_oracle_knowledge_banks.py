"""Unit tests for Oracle AI Database 26ai Knowledge Banks support."""

from __future__ import annotations

import array
import importlib.util
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
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


def _load_migration_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "hindsight_api"
        / "alembic"
        / "versions"
        / "e3c4d5e6f7a8_oracle_knowledge_banks.py"
    )
    spec = importlib.util.spec_from_file_location("e3c4d5e6f7a8_oracle_knowledge_banks", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    assert "privilege = 'INSERT'" in joined
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON" in joined

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
