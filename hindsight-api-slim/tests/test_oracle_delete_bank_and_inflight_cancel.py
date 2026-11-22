"""Tests for Oracle rollback on cancellation, delete_bank concurrency, and in-flight worker cancellation (PI-96)."""

from __future__ import annotations

import asyncio
import io
import json
import uuid
import zipfile
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import oracledb
import pytest
from asyncpg.exceptions import DeadlockDetectedError

from hindsight_api.engine.bank_life import BankDeletedError, bind_job_bank
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.transfer import importer
from hindsight_api.models import RequestContext
from hindsight_api.worker.exceptions import RetryTaskAt
from hindsight_api.worker.poller import ClaimedTask, WorkerPoller


def _ora_error(code: int, message: str) -> oracledb.DatabaseError:
    return oracledb.DatabaseError(SimpleNamespace(code=code, message=message))


def _ora_integrity_error(code: int, message: str) -> oracledb.IntegrityError:
    return oracledb.IntegrityError(SimpleNamespace(code=code, message=message))


class _FakeOracleCursor:
    def __init__(self, executed: list[str], *, fail_on_rollback: bool = False) -> None:
        self._executed = executed
        self._fail_on_rollback = fail_on_rollback
        self.closed = False

    async def execute(self, sql: str, *args, **kwargs) -> None:
        self._executed.append(sql)
        if self._fail_on_rollback and sql.startswith("ROLLBACK TO SAVEPOINT"):
            raise RuntimeError("lost session")

    def close(self) -> None:
        self.closed = True


class _FakeRawOracleConn:
    def __init__(self, *, fail_rollback: bool = False) -> None:
        self.version = "23.26.4.1.0"
        self.current_schema = "ADMIN"
        self.executed: list[str] = []
        self.cursors: list[_FakeOracleCursor] = []
        self.commits = 0
        self.rollbacks = 0
        self._fail_rollback = fail_rollback

    def cursor(self) -> _FakeOracleCursor:
        cur = _FakeOracleCursor(self.executed, fail_on_rollback=self._fail_rollback)
        self.cursors.append(cur)
        return cur

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1
        if self._fail_rollback:
            raise ConnectionError("network drop during rollback")


class _FakeOraclePool:
    def __init__(self, conn: _FakeRawOracleConn) -> None:
        self.conn = conn
        self.released: list[_FakeRawOracleConn] = []
        self.dropped: list[_FakeRawOracleConn] = []
        self.busy = 0
        self.max = 4
        self.opened = 1

    async def acquire(self) -> _FakeRawOracleConn:
        return self.conn

    async def release(self, conn: _FakeRawOracleConn) -> None:
        self.released.append(conn)

    async def drop(self, conn: _FakeRawOracleConn) -> None:
        self.dropped.append(conn)


@pytest.mark.asyncio
async def test_oracle_connection_transaction_rolls_back_savepoint_on_cancel() -> None:
    from hindsight_api.engine.db.oracle import OracleConnection

    raw = _FakeRawOracleConn()
    conn = OracleConnection(raw)

    with pytest.raises(asyncio.CancelledError):
        async with conn.transaction():
            raise asyncio.CancelledError

    assert len(raw.executed) == 2
    assert raw.executed[0].startswith("SAVEPOINT sp_")
    assert raw.executed[1].startswith("ROLLBACK TO SAVEPOINT sp_")
    assert all(c.closed for c in raw.cursors)


@pytest.mark.asyncio
async def test_oracle_connection_transaction_preserves_cancel_if_rollback_fails() -> None:
    from hindsight_api.engine.db.oracle import OracleConnection

    raw = _FakeRawOracleConn(fail_rollback=True)
    conn = OracleConnection(raw)

    with pytest.raises(asyncio.CancelledError):
        async with conn.transaction():
            raise asyncio.CancelledError

    assert all(c.closed for c in raw.cursors)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["acquire", "transaction"])
async def test_oracle_backend_rolls_back_and_drops_broken_conn_on_cancel(mode: str) -> None:
    from hindsight_api.engine.db.oracle import OracleBackend

    backend = OracleBackend()
    backend._default_schema = "ADMIN"

    healthy = _FakeRawOracleConn()
    healthy_pool = _FakeOraclePool(healthy)
    backend._pool = healthy_pool

    with pytest.raises(asyncio.CancelledError):
        async with getattr(backend, mode)():
            raise asyncio.CancelledError

    assert healthy.commits == 0
    assert healthy.rollbacks == 1
    assert healthy_pool.released == [healthy]
    assert healthy_pool.dropped == []

    broken = _FakeRawOracleConn(fail_rollback=True)
    broken_pool = _FakeOraclePool(broken)
    backend._pool = broken_pool

    with pytest.raises(asyncio.CancelledError):
        async with getattr(backend, mode)():
            raise asyncio.CancelledError

    assert broken.rollbacks == 1
    assert broken_pool.dropped == [broken]
    assert broken_pool.released == []


def _make_stub_engine_for_delete() -> MemoryEngine:
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._initialized = True
    engine._operation_validator = None
    engine._tenant_extension = None
    engine._active_operation_canceller = None
    engine._active_bank_operations_canceller = None
    engine._authenticate_tenant = AsyncMock(return_value=None)
    engine._refuse_bank_wide_if_scoped = AsyncMock(return_value=None)
    engine._bank_stats_cache = SimpleNamespace(invalidate=AsyncMock())
    engine._delete_files_quietly = AsyncMock()
    engine._submit_vector_index_maintenance_quietly = AsyncMock()
    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(return_value=SimpleNamespace(enable_auto_consolidation=False))
    )
    return engine


class _StubDeleteConn:
    def __init__(self, *, fail_on_bank_lock: list[BaseException] | None = None) -> None:
        self.fail_on_bank_lock = fail_on_bank_lock if fail_on_bank_lock is not None else []
        self.queries: list[str] = []

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def execute(self, query: str, *args, **kwargs):
        self.queries.append(query)
        return "DELETE 1"

    async def fetchrow(self, query: str, *args, **kwargs):
        self.queries.append(query)
        if "SELECT bank_id FROM" in query and "banks" in query and self.fail_on_bank_lock:
            raise self.fail_on_bank_lock.pop(0)
        return {"bank_id": args[0]}

    async def fetch(self, query: str, *args, **kwargs):
        self.queries.append(query)
        return []

    async def fetchval(self, query: str, *args, **kwargs):
        self.queries.append(query)
        if "DELETE FROM" in query and "banks" in query and "RETURNING internal_id" in query:
            return uuid.uuid4()
        return None


class _StubDeleteBackend:
    _wraps_backend = True

    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = failures
        self.acquires = 0
        self.ops = SimpleNamespace(drop_bank_vector_indexes=AsyncMock())

    @asynccontextmanager
    async def acquire(self):
        self.acquires += 1
        yield _StubDeleteConn(fail_on_bank_lock=self.failures)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deadlock_exc",
    [
        _ora_error(60, "ORA-00060: deadlock detected while waiting for resource"),
        DeadlockDetectedError("deadlock detected"),
    ],
)
async def test_delete_bank_retries_deadlock_and_cancels_inflight_tasks(deadlock_exc: BaseException) -> None:
    engine = _make_stub_engine_for_delete()
    canceller = MagicMock(return_value=2)
    engine.register_worker_cancellers(cancel_bank_operations=canceller)

    backend = _StubDeleteBackend([deadlock_exc])
    engine._backend = backend
    engine._get_backend = AsyncMock(return_value=backend)

    store = SimpleNamespace(
        store_owned_for=MagicMock(return_value=False),
        count_bank_contents=AsyncMock(return_value=SimpleNamespace(memory_units=3, entities=1, documents=1)),
        purge_bank_rows=AsyncMock(return_value=[]),
        bank_memories_of_type=AsyncMock(),
        delete_bank_memories_of_type=AsyncMock(),
    )

    with (
        patch("hindsight_api.engine.retain.bank_utils.bank_kind", AsyncMock(return_value="memory")),
        patch("hindsight_api.engine.retain.bank_utils.drop_bank_vector_indexes", AsyncMock()),
        patch("hindsight_api.engine.memories.get_memories", return_value=store),
        patch("hindsight_api.engine.db_utils.asyncio.sleep", AsyncMock()),
    ):
        result = await engine.delete_bank("b-retry", request_context=RequestContext(internal=True))

    assert result["bank_deleted"] is True
    assert backend.acquires == 3  # 1 failed tx + 1 retried tx + 1 index drop acquire
    canceller.assert_called_once_with("b-retry")


@pytest.mark.asyncio
async def test_delete_bank_does_not_cancel_inflight_on_partial_clear() -> None:
    engine = _make_stub_engine_for_delete()
    canceller = MagicMock(return_value=0)
    engine.register_worker_cancellers(cancel_bank_operations=canceller)

    backend = _StubDeleteBackend([])
    engine._backend = backend
    engine._get_backend = AsyncMock(return_value=backend)

    store = SimpleNamespace(
        store_owned_for=MagicMock(return_value=False),
        count_bank_contents=AsyncMock(return_value=SimpleNamespace(memory_units=1, entities=0, documents=1)),
        purge_bank_rows=AsyncMock(return_value=[]),
        bank_memories_of_type=AsyncMock(return_value=SimpleNamespace(unit_ids=[], count=1)),
        delete_bank_memories_of_type=AsyncMock(),
    )

    with (
        patch("hindsight_api.engine.retain.bank_utils.bank_kind", AsyncMock(return_value="memory")),
        patch("hindsight_api.engine.memories.get_memories", return_value=store),
    ):
        await engine.delete_bank("b-clear", delete_bank_profile=False, request_context=RequestContext(internal=True))
        await engine.delete_bank(
            "b-type", fact_type="world", delete_bank_profile=False, request_context=RequestContext(internal=True)
        )

    canceller.assert_not_called()


@pytest.mark.asyncio
async def test_delete_bank_surfaces_non_retryable_errors() -> None:
    engine = _make_stub_engine_for_delete()
    backend = _StubDeleteBackend([RuntimeError("permission denied")])
    engine._backend = backend
    engine._get_backend = AsyncMock(return_value=backend)

    with (
        patch("hindsight_api.engine.retain.bank_utils.bank_kind", AsyncMock(return_value="memory")),
        pytest.raises(Exception, match="Failed to delete agent data: RuntimeError: permission denied"),
    ):
        await engine.delete_bank("b-err", request_context=RequestContext(internal=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("bank_still_exists", [False, True])
async def test_dispatch_task_handles_ora02291_for_deleted_vs_live_bank(bank_still_exists: bool) -> None:
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._audit_logger = None
    fk_exc = _ora_integrity_error(2291, "ORA-02291: integrity constraint (ADMIN.FK_DOCUMENTS_BANK) violated")
    engine._handle_import_documents = AsyncMock(side_effect=fk_exc)

    bank_id = "pi96-fk-bank"
    bound_internal_id = str(uuid.uuid4())
    # First fetchval reads the initial binding; second checks liveness in _dispatch_task.
    responses = [bound_internal_id, bound_internal_id if bank_still_exists else None]

    class _BankLifeBackend:
        _wraps_backend = True

        @asynccontextmanager
        async def acquire(self):
            val = responses.pop(0)
            yield SimpleNamespace(fetchval=AsyncMock(return_value=val))

    backend = _BankLifeBackend()
    engine._get_backend = AsyncMock(return_value=backend)

    binding = await bind_job_bank(backend, bank_id)
    with (
        binding,
        patch(
            "hindsight_api.engine.memory_engine.get_config",
            return_value=SimpleNamespace(worker_max_retries=2, worker_task_retry_backoff_seconds=60),
        ),
    ):
        task = {"type": "import_documents", "bank_id": bank_id, "operation_id": str(uuid.uuid4())}
        if bank_still_exists:
            with pytest.raises(RetryTaskAt):
                await engine._dispatch_task(task, "import_documents", task["operation_id"], None)
        else:
            await engine._dispatch_task(task, "import_documents", task["operation_id"], None)


def _build_transfer_zip(doc_ids: list[str], *, with_observations: bool = False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "source_bank_id": "src-bank",
                    "archive_type": "documents",
                    "document_count": len(doc_ids),
                    "fact_count": len(doc_ids),
                    "observation_count": 1 if with_observations else 0,
                }
            ),
        )
        zf.writestr("mental_models.json", "[]")
        zf.writestr("knowledge_pages.json", "[]")
        for idx, doc_id in enumerate(doc_ids):
            zf.writestr(
                f"documents/{idx:04d}.json",
                json.dumps(
                    {
                        "id": doc_id,
                        "original_text": f"Document {doc_id}",
                        "tags": [],
                        "chunks": [{"chunk_index": 0, "chunk_text": f"Document {doc_id}"}],
                        "facts": [
                            {
                                "text": f"Fact for {doc_id}",
                                "fact_type": "world",
                                "chunk_index": 0,
                                "entities": [],
                                "tags": [],
                            }
                        ],
                    }
                ),
            )
        if with_observations:
            zf.writestr(
                "observations.json",
                json.dumps(
                    [
                        {
                            "text": "Obs 1",
                            "sources": [{"document_id": doc_ids[0], "fact_index": 0}],
                            "tags": [],
                        }
                    ]
                ),
            )
    return buf.getvalue()


@pytest.mark.asyncio
async def test_import_documents_stops_early_on_check_alive_false(monkeypatch) -> None:
    monkeypatch.setattr(importer, "_DOCUMENT_BATCH_SIZE", 1)
    archive = _build_transfer_zip(["doc-1", "doc-2", "doc-3"])

    imported_batches: list[list[str]] = []
    checks = 0

    async def fake_check_alive() -> bool:
        nonlocal checks
        checks += 1
        return checks == 1

    async def fake_prepare(_emb, _fmt, batch):
        return batch

    async def fake_import_batch(**kwargs):
        prepared = kwargs["prepared"]
        imported_batches.append([target_id for _, target_id in prepared])
        return [SimpleNamespace(unit_ids=[str(uuid.uuid4())], original_ordinals=[0]) for _ in prepared]

    backend = SimpleNamespace(ops=SimpleNamespace())
    store = SimpleNamespace(transfer_document_exists=AsyncMock(return_value=False))

    with (
        patch("hindsight_api.engine.memories.get_memories", return_value=store),
        patch.object(importer, "_prepare_documents", side_effect=fake_prepare),
        patch.object(importer, "_import_document_batch", side_effect=fake_import_batch),
    ):
        result = await importer.import_documents(
            backend=backend,
            embeddings_model=MagicMock(),
            entity_resolver=MagicMock(),
            config=SimpleNamespace(),
            format_date_fn=lambda d: "",
            bank_id="dst-bank",
            archive=archive,
            check_alive=fake_check_alive,
        )

    assert result.cancelled is True
    assert result.documents_imported == 1
    assert imported_batches == [["doc-1"]]


@pytest.mark.asyncio
async def test_import_documents_aborts_with_bank_deleted_error_when_bank_gone() -> None:
    archive = _build_transfer_zip(["doc-1", "doc-2"])
    bank_id = "vanished-bank"
    responses = [str(uuid.uuid4()), None]

    class _BankBackend:
        _wraps_backend = True
        ops = SimpleNamespace()

        @asynccontextmanager
        async def acquire(self):
            val = responses.pop(0)
            yield SimpleNamespace(fetchval=AsyncMock(return_value=val))

    backend = _BankBackend()
    binding = await bind_job_bank(backend, bank_id)
    with binding, pytest.raises(BankDeletedError):
        await importer.import_documents(
            backend=backend,
            embeddings_model=MagicMock(),
            entity_resolver=MagicMock(),
            config=SimpleNamespace(),
            format_date_fn=lambda d: "",
            bank_id=bank_id,
            archive=archive,
        )


@pytest.mark.asyncio
async def test_handle_import_documents_skips_completion_writes_and_cleanup_when_cancelled() -> None:
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._file_storage = SimpleNamespace(delete=AsyncMock())
    engine._run_import_documents = AsyncMock(return_value=importer.ImportResult(documents_imported=1, cancelled=True))
    engine._report_imported_documents = AsyncMock()
    engine._submit_post_insert_maintenance = AsyncMock()

    backend = SimpleNamespace(acquire=MagicMock())
    engine._get_backend = AsyncMock(return_value=backend)

    @asynccontextmanager
    async def fake_retrieve(_key: str):
        yield io.BytesIO(b"zip")

    engine._retrieve_archive = fake_retrieve  # type: ignore[method-assign]

    await engine._handle_import_documents(
        {
            "bank_id": "b-import",
            "storage_key": "banks/b-import/imports/1/transfer.zip",
            "on_conflict": "skip",
            "operation_id": str(uuid.uuid4()),
        }
    )

    backend.acquire.assert_not_called()
    engine._file_storage.delete.assert_not_called()
    engine._report_imported_documents.assert_not_called()
    engine._submit_post_insert_maintenance.assert_not_called()


class _FakePollerBackend:
    backend_type = "postgresql"

    def __init__(self, rows_by_op: dict[str, str] | None = None) -> None:
        self.rows_by_op = rows_by_op or {}
        self.mark_calls: list[str] = []

    @asynccontextmanager
    async def acquire(self):
        rows_by_op = self.rows_by_op
        mark_calls = self.mark_calls

        class _Conn:
            @asynccontextmanager
            async def transaction(self):
                yield self

            async def fetch(self, query: str, *args):
                if "SELECT operation_id, status FROM" in query:
                    ids = args[0]
                    return [
                        {"operation_id": oid, "status": rows_by_op[str(oid)]} for oid in ids if str(oid) in rows_by_op
                    ]
                return []

            async def execute(self, query: str, *args):
                mark_calls.append(query)
                return "UPDATE 0"

        yield _Conn()


@pytest.mark.asyncio
async def test_poller_cancel_active_operation_interrupts_inflight_and_skips_terminal_writes() -> None:
    backend = _FakePollerBackend()
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def slow_executor(_task_dict):
        started.set()
        try:
            await asyncio.sleep(60)
        finally:
            cleaned_up.set()

    poller = WorkerPoller(backend=backend, worker_id="w-cancel", executor=slow_executor)
    poller._mark_all_completed = AsyncMock()
    poller._mark_all_failed = AsyncMock()

    op_id = str(uuid.uuid4())
    await poller.execute_task(
        ClaimedTask(
            operation_id=op_id,
            task_dict={"type": "import_documents", "operation_type": "import_documents", "bank_id": "bank-1"},
            schema=None,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2.0)

    assert poller.cancel_active_operation(op_id) is True
    await asyncio.wait_for(cleaned_up.wait(), timeout=2.0)
    assert await poller.wait_for_active_tasks(timeout=2.0) is True
    assert poller._slot_freed.is_set()
    poller._mark_all_completed.assert_not_called()
    poller._mark_all_failed.assert_not_called()


@pytest.mark.asyncio
async def test_poller_cancel_skips_self_and_folded_and_bank_cancels_only_target() -> None:
    backend = _FakePollerBackend()
    op1, op2, op3 = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    self_cancel_result: list[bool] = []
    started = {op1: asyncio.Event(), op2: asyncio.Event(), op3: asyncio.Event()}
    cancelled_ops: set[str] = set()
    poller_ref: list[WorkerPoller] = []

    async def executor(task_dict):
        oid = task_dict["operation_id"]
        started[oid].set()
        if oid == op1:
            self_cancel_result.append(poller_ref[0].cancel_active_operation(op1))
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled_ops.add(oid)
            raise

    poller = WorkerPoller(backend=backend, worker_id="w-bank", executor=executor)
    poller_ref.append(poller)

    await poller.execute_task(
        ClaimedTask(
            operation_id=op1,
            task_dict={"type": "batch_retain", "operation_id": op1, "bank_id": "bank-a"},
            schema=None,
        )
    )
    # Multi-op folded task on bank-a: single-op cancel defers, bank-level cancel still fires
    await poller.execute_task(
        ClaimedTask(
            operation_id=op2,
            task_dict={"type": "batch_retain", "operation_id": op2, "bank_id": "bank-a"},
            schema=None,
            folded_operation_ids=[str(uuid.uuid4())],
        )
    )
    await poller.execute_task(
        ClaimedTask(
            operation_id=op3,
            task_dict={"type": "batch_retain", "operation_id": op3, "bank_id": "bank-b"},
            schema=None,
        )
    )

    for ev in started.values():
        await asyncio.wait_for(ev.wait(), timeout=2.0)

    assert self_cancel_result == [False]
    assert poller.cancel_active_operation(op2) is False
    assert poller.cancel_active_bank_operations("bank-a") == 2

    # Wait until bank-a tasks drain; bank-b should stay running
    for _ in range(50):
        if op1 in cancelled_ops and op2 in cancelled_ops:
            break
        await asyncio.sleep(0.01)

    assert cancelled_ops == {op1, op2}
    assert op3 in poller._active_tasks

    assert poller.cancel_active_operation(op3) is True
    assert await poller.wait_for_active_tasks(timeout=2.0) is True
    assert cancelled_ops == {op1, op2, op3}


@pytest.mark.asyncio
async def test_poller_db_watcher_cancels_cancelled_and_deleted_and_respects_folded_siblings() -> None:
    op_cancelled = str(uuid.uuid4())
    op_deleted = str(uuid.uuid4())
    op_fold_a = str(uuid.uuid4())
    op_fold_b = str(uuid.uuid4())

    backend = _FakePollerBackend(
        rows_by_op={
            op_cancelled: "cancelled",
            op_fold_a: "cancelled",
            op_fold_b: "processing",
            # op_deleted intentionally absent -> bank/op deleted via CASCADE
        }
    )
    started = {oid: asyncio.Event() for oid in (op_cancelled, op_deleted, op_fold_a)}

    async def wait_executor(task_dict):
        started[task_dict["operation_id"]].set()
        await asyncio.sleep(60)

    poller = WorkerPoller(backend=backend, worker_id="w-watch", executor=wait_executor)
    for oid, folded in ((op_cancelled, []), (op_deleted, []), (op_fold_a, [op_fold_b])):
        await poller.execute_task(
            ClaimedTask(
                operation_id=oid,
                task_dict={"type": "consolidation", "operation_id": oid, "bank_id": "bank-watch"},
                schema=None,
                folded_operation_ids=folded,
            )
        )

    for ev in started.values():
        await asyncio.wait_for(ev.wait(), timeout=2.0)

    assert await poller._check_cancelled_active_tasks() == 2
    await asyncio.sleep(0.05)

    assert op_cancelled not in poller._active_tasks
    assert op_deleted not in poller._active_tasks
    assert op_fold_a in poller._active_tasks

    # Now flip the remaining folded sibling to cancelled too
    backend.rows_by_op[op_fold_b] = "cancelled"
    assert await poller._check_cancelled_active_tasks() == 1
    assert await poller.wait_for_active_tasks(timeout=2.0) is True


@pytest.mark.asyncio
async def test_poller_wall_timeout_and_shutdown_graceful_clean_up_exec_task() -> None:
    backend = _FakePollerBackend()
    started = asyncio.Event()
    exec_cancelled = asyncio.Event()

    async def hung_executor(_task_dict):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            exec_cancelled.set()
            raise

    poller = WorkerPoller(backend=backend, worker_id="w-wall", executor=hung_executor)
    poller._mark_all_failed = AsyncMock()

    op_id = str(uuid.uuid4())
    with patch("hindsight_api.worker.poller._wall_timeout_for", return_value=0.05):
        await poller.execute_task(
            ClaimedTask(
                operation_id=op_id,
                task_dict={"type": "batch_retain", "operation_id": op_id, "bank_id": "bank-wall"},
                schema=None,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2.0)
        await asyncio.wait_for(exec_cancelled.wait(), timeout=2.0)
        assert await poller.wait_for_active_tasks(timeout=2.0) is True

    poller._mark_all_failed.assert_awaited_once()

    # shutdown_graceful cancels remaining active tasks cleanly without stranding exec_task
    started.clear()
    exec_cancelled.clear()
    op_id2 = str(uuid.uuid4())
    with patch("hindsight_api.worker.poller._wall_timeout_for", return_value=None):
        await poller.execute_task(
            ClaimedTask(
                operation_id=op_id2,
                task_dict={"type": "batch_retain", "operation_id": op_id2, "bank_id": "bank-shut"},
                schema=None,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2.0)
        await poller.shutdown_graceful(timeout=0.05)

    assert exec_cancelled.is_set()
    assert await poller.wait_for_active_tasks(timeout=2.0) is True


@pytest.mark.asyncio
async def test_cancel_operation_triggers_registered_worker_canceller() -> None:
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._operation_validator = None
    engine._authenticate_tenant = AsyncMock(return_value=None)
    engine._refuse_bank_wide_if_scoped = AsyncMock(return_value=None)
    engine._maybe_update_parent_operation = AsyncMock()
    canceller = MagicMock(return_value=True)
    engine._active_bank_operations_canceller = None
    engine._active_operation_canceller = None
    engine.register_worker_cancellers(cancel_operation=canceller)

    op_id = str(uuid.uuid4())

    class _CancelConn:
        @asynccontextmanager
        async def transaction(self):
            yield self

        async def fetchrow(self, query: str, *args):
            if "SELECT bank_id, status" in query:
                return {"bank_id": "bank-c", "status": "processing"}
            if "UPDATE" in query:
                return {"status": "cancelled"}
            return None

    class _CancelBackend:
        _wraps_backend = True

        @asynccontextmanager
        async def acquire(self):
            yield _CancelConn()

    engine._get_backend = AsyncMock(return_value=_CancelBackend())

    resp = await engine.cancel_operation("bank-c", op_id, request_context=RequestContext(internal=True))

    assert resp["success"] is True
    canceller.assert_called_once_with(op_id)
