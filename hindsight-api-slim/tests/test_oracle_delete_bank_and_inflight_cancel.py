"""Tests for Oracle rollback on cancellation, delete_bank concurrency, and in-flight worker cancellation (PI-96)."""

from __future__ import annotations

import asyncio
import io
import json
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from asyncpg.exceptions import DeadlockDetectedError

from hindsight_api.engine.bank_life import BankDeletedError, bind_job_bank
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.transfer import importer
from hindsight_api.extensions.operation_validator import OperationValidationError
from hindsight_api.models import RequestContext
from hindsight_api.worker.exceptions import RetryTaskAt
from hindsight_api.worker.poller import ClaimedTask, WorkerPoller

try:
    import oracledb
except ImportError:
    oracledb = None  # type: ignore[assignment]

_requires_oracledb = pytest.mark.skipif(oracledb is None, reason="oracledb optional extra is not installed")


async def _wait_until(predicate, *, timeout: float = 2.0, step: float = 0.005) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(step)
    raise AssertionError("Condition was not met before timeout")


def _ora_error(code: int, message: str):
    assert oracledb is not None
    return oracledb.DatabaseError(SimpleNamespace(code=code, message=message))


def _ora_integrity_error(code: int, message: str):
    assert oracledb is not None
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


@_requires_oracledb
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


@_requires_oracledb
@pytest.mark.asyncio
async def test_oracle_connection_transaction_preserves_cancel_if_rollback_fails() -> None:
    from hindsight_api.engine.db.oracle import OracleConnection

    raw = _FakeRawOracleConn(fail_rollback=True)
    conn = OracleConnection(raw)

    with pytest.raises(asyncio.CancelledError):
        async with conn.transaction():
            raise asyncio.CancelledError

    assert any(sql.startswith("ROLLBACK TO SAVEPOINT sp_") for sql in raw.executed)
    assert all(c.closed for c in raw.cursors)


@_requires_oracledb
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
    engine._finish_bank_delete = None
    engine._is_operation_active = None
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
    "deadlock_kind",
    [
        pytest.param("ora_60", marks=_requires_oracledb),
        "pg_deadlock",
    ],
)
async def test_delete_bank_retries_deadlock_and_cancels_inflight_tasks(deadlock_kind: str) -> None:
    deadlock_exc: BaseException = (
        _ora_error(60, "ORA-00060: deadlock detected while waiting for resource")
        if deadlock_kind == "ora_60"
        else DeadlockDetectedError("deadlock detected")
    )
    engine = _make_stub_engine_for_delete()
    canceller = MagicMock(return_value=2)
    finisher = MagicMock()
    engine.register_worker_cancellers(cancel_bank_operations=canceller, finish_bank_delete=finisher)

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
    canceller.assert_called_once_with("b-retry", None, track_delete_gate=True)
    finisher.assert_called_once_with("b-retry", None)


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


@_requires_oracledb
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

    backend: Any = _BankLifeBackend()
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

    backend: Any = _BankBackend()
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

    poller = WorkerPoller(backend=cast(Any, backend), worker_id="w-cancel", executor=slow_executor)
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
async def test_poller_cancel_skips_self_and_interrupts_folded_and_bank_cancels_only_target() -> None:
    backend = _FakePollerBackend()
    op1, op2, op2_peer, op3 = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
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

    poller = WorkerPoller(backend=cast(Any, backend), worker_id="w-bank", executor=executor)
    poller_ref.append(poller)

    await poller.execute_task(
        ClaimedTask(
            operation_id=op1,
            task_dict={"type": "batch_retain", "operation_id": op1, "bank_id": "bank-a"},
            schema=None,
        )
    )
    # Multi-op folded task on bank-a: cancelling either child or parent interrupts the folded execution
    await poller.execute_task(
        ClaimedTask(
            operation_id=op2,
            task_dict={"type": "batch_retain", "operation_id": op2, "bank_id": "bank-a"},
            schema=None,
            folded_operation_ids=[op2_peer],
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
    assert poller.cancel_active_operation(op2_peer) is True
    await _wait_until(lambda: op2 not in poller._active_tasks)
    assert poller.cancel_active_bank_operations("bank-a") == 1

    await _wait_until(lambda: op1 in cancelled_ops and op2 in cancelled_ops and op1 not in poller._active_tasks)

    assert cancelled_ops == {op1, op2}
    assert op3 in poller._active_tasks

    assert poller.cancel_active_operation(op3) is True
    assert await poller.wait_for_active_tasks(timeout=2.0) is True
    assert cancelled_ops == {op1, op2, op3}


@pytest.mark.asyncio
async def test_poller_db_watcher_cancels_cancelled_deleted_and_partial_folded_runs() -> None:
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

    poller = WorkerPoller(backend=cast(Any, backend), worker_id="w-watch", executor=wait_executor)
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

    # All 3 tasks must be cancelled (including the fold with 1 cancelled member so its content isn't written).
    assert await poller._check_cancelled_active_tasks() == 3
    await _wait_until(
        lambda: (
            op_cancelled not in poller._active_tasks
            and op_deleted not in poller._active_tasks
            and op_fold_a not in poller._active_tasks
        )
    )
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

    poller = WorkerPoller(backend=cast(Any, backend), worker_id="w-wall", executor=hung_executor)
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
    engine._finish_bank_delete = None
    engine._is_operation_active = None
    engine.register_worker_cancellers(cancel_operation=canceller)

    op_id = str(uuid.uuid4())
    claimed_at = datetime(2026, 11, 22, 16, 0, tzinfo=timezone.utc)

    class _CancelConn:
        @asynccontextmanager
        async def transaction(self):
            yield self

        async def fetchrow(self, query: str, *args):
            if "SELECT bank_id, status" in query:
                return {"bank_id": "bank-c", "status": "processing"}
            if "UPDATE" in query:
                return {"status": "cancelled", "worker_id": "worker-1", "claimed_at": claimed_at}
            return None

    class _CancelBackend:
        _wraps_backend = True

        @asynccontextmanager
        async def acquire(self):
            yield _CancelConn()

    engine._get_backend = AsyncMock(return_value=_CancelBackend())

    resp = await engine.cancel_operation("bank-c", op_id, request_context=RequestContext(internal=True))

    assert resp["success"] is True
    canceller.assert_called_once_with(
        op_id,
        None,
        expected_worker_id="worker-1",
        expected_claimed_at=claimed_at,
    )


@_requires_oracledb
@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["cursor_open", "cursor_close"])
async def test_oracle_connection_transaction_preserves_cancel_on_cursor_errors(failure_point: str) -> None:
    from hindsight_api.engine.db.oracle import OracleConnection

    raw = _FakeRawOracleConn()
    original_cursor = raw.cursor
    call_count = 0

    def flaky_cursor():
        nonlocal call_count
        call_count += 1
        if call_count == 2 and failure_point == "cursor_open":
            raise RuntimeError("connection closed before rollback cursor")
        cur = original_cursor()
        if call_count == 2 and failure_point == "cursor_close":

            def _boom_close() -> None:
                cur.closed = True
                raise RuntimeError("close failed")

            cur.close = _boom_close  # type: ignore[method-assign]
        return cur

    raw.cursor = flaky_cursor  # type: ignore[method-assign]
    conn = OracleConnection(raw)

    with pytest.raises(asyncio.CancelledError):
        async with conn.transaction():
            raise asyncio.CancelledError

    if failure_point == "cursor_close":
        assert any(sql.startswith("ROLLBACK TO SAVEPOINT sp_") for sql in raw.executed)


@_requires_oracledb
@pytest.mark.asyncio
async def test_oracle_backend_drop_failure_closes_conn_and_post_commit_release_failure_does_not_raise() -> None:
    from hindsight_api.engine.db.oracle import OracleBackend

    closed = 0

    class _TrackedConn(_FakeRawOracleConn):
        def close(self) -> None:
            nonlocal closed
            closed += 1

    class _FailingPool(_FakeOraclePool):
        def __init__(self, conn: _TrackedConn, *, fail_release: bool = False) -> None:
            super().__init__(conn)
            self._fail_release = fail_release

        async def drop(self, conn: _FakeRawOracleConn) -> None:
            self.dropped.append(conn)
            raise RuntimeError("drop failed")

        async def release(self, conn: _FakeRawOracleConn) -> None:
            self.released.append(conn)
            if self._fail_release:
                raise ConnectionError("release failed after commit")

    # 1) When rollback fails AND pool.drop fails, broken conn is closed and NEVER released back to pool.
    broken_conn = _TrackedConn(fail_rollback=True)
    drop_fail_pool = _FailingPool(broken_conn)
    backend = OracleBackend()
    backend._default_schema = "ADMIN"
    backend._pool = drop_fail_pool

    with pytest.raises(asyncio.CancelledError):
        async with backend.acquire():
            raise asyncio.CancelledError

    assert drop_fail_pool.dropped == [broken_conn]
    assert drop_fail_pool.released == []
    assert closed == 1

    # 2) Post-commit pool.release failure drops/closes without raising to caller.
    closed_before = closed
    post_conn = _TrackedConn()
    release_fail_pool = _FailingPool(post_conn, fail_release=True)
    backend._pool = release_fail_pool

    async with backend.acquire():
        pass

    assert post_conn.commits == 1
    assert release_fail_pool.released == [post_conn]
    assert release_fail_pool.dropped == [post_conn]
    assert closed == closed_before + 1


@pytest.mark.asyncio
async def test_poller_cancel_bank_operations_isolates_tenant_schemas() -> None:
    backend = _FakePollerBackend()
    op_default = str(uuid.uuid4())
    op_tenant_b = str(uuid.uuid4())
    started = {op_default: asyncio.Event(), op_tenant_b: asyncio.Event()}
    cancelled: set[str] = set()

    async def exec_fn(task_dict):
        oid = task_dict["operation_id"]
        started[oid].set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.add(oid)
            raise

    poller = WorkerPoller(backend=cast(Any, backend), worker_id="w-tenant", executor=exec_fn)
    await poller.execute_task(
        ClaimedTask(
            operation_id=op_default,
            task_dict={"type": "batch_retain", "operation_id": op_default, "bank_id": "shared"},
            schema=None,
        )
    )
    await poller.execute_task(
        ClaimedTask(
            operation_id=op_tenant_b,
            task_dict={"type": "batch_retain", "operation_id": op_tenant_b, "bank_id": "shared"},
            schema="tenant_b",
        )
    )
    for ev in started.values():
        await asyncio.wait_for(ev.wait(), timeout=2.0)

    assert poller.cancel_active_bank_operations("shared", schema="tenant_b") == 1
    await _wait_until(lambda: op_tenant_b in cancelled and op_tenant_b not in poller._active_tasks)

    assert cancelled == {op_tenant_b}
    assert op_default in poller._active_tasks

    assert poller.cancel_active_operation(op_default, schema="tenant_b") is False
    assert poller.cancel_active_operation(op_default, schema="public") is True
    assert await poller.wait_for_active_tasks(timeout=2.0) is True


@pytest.mark.asyncio
async def test_failed_delete_bank_releases_gate_and_requeues_interrupted_tasks() -> None:
    op_id = str(uuid.uuid4())
    status_by_id = {op_id: "processing"}

    class _ReconcileBackend:
        _wraps_backend = True

        @asynccontextmanager
        async def acquire(self):
            class _Conn:
                @asynccontextmanager
                async def transaction(self):
                    yield self

                async def fetchval(self, *_args):
                    return 1

                async def fetch(self, query: str, *args):
                    if "SELECT operation_id, status" in query:
                        return [
                            {
                                "operation_id": uuid.UUID(oid),
                                "status": status_by_id[oid],
                                "worker_id": "w-del-fail",
                                "claimed_at": None,
                            }
                            for oid in status_by_id
                        ]
                    return []

                async def execute(self, query: str, *args):
                    if "SET status = 'pending'" in query:
                        status_by_id[str(args[0])] = "pending"
                        return "UPDATE 1"
                    return "UPDATE 0"

            yield _Conn()

    poller_backend = _ReconcileBackend()
    started = asyncio.Event()

    async def worker_exec(_task_dict):
        started.set()
        await asyncio.sleep(60)

    poller = WorkerPoller(backend=cast(Any, poller_backend), worker_id="w-del-fail", executor=worker_exec)
    await poller.execute_task(
        ClaimedTask(
            operation_id=op_id,
            task_dict={"type": "batch_retain", "operation_id": op_id, "bank_id": "bank-keep"},
            schema=None,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2.0)

    engine = _make_stub_engine_for_delete()
    engine.register_worker_cancellers(
        cancel_operation=poller.cancel_active_operation,
        cancel_bank_operations=poller.cancel_active_bank_operations,
        finish_bank_delete=poller.finish_bank_delete,
        is_operation_active=poller.is_operation_active,
    )
    fail_delete_backend = _StubDeleteBackend([RuntimeError("boom in delete")])
    engine._backend = fail_delete_backend
    engine._get_backend = AsyncMock(return_value=fail_delete_backend)

    with (
        patch("hindsight_api.engine.retain.bank_utils.bank_kind", AsyncMock(return_value="memory")),
        pytest.raises(Exception, match="Failed to delete agent data: RuntimeError: boom in delete"),
    ):
        await engine.delete_bank("bank-keep", request_context=RequestContext(internal=True))

    await _wait_until(lambda: not poller._active_tasks and status_by_id[op_id] == "pending")
    assert poller._deleting_banks == {}


@pytest.mark.asyncio
async def test_cancel_operation_skips_pending_hook_checks_attempt_blocks_retry_and_keeps_new_attempt_tracking() -> None:
    op_id = str(uuid.uuid4())
    claimed_t1 = datetime(2026, 11, 22, 16, 0, 1, tzinfo=timezone.utc)
    claimed_t2 = datetime(2026, 11, 22, 16, 0, 2, tzinfo=timezone.utc)

    # 1) WorkerPoller attempt match & cleanup identity check
    backend = _FakePollerBackend()
    started_t2 = asyncio.Event()
    cancelled_t2 = asyncio.Event()

    async def exec_t2(_task_dict):
        started_t2.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled_t2.set()
            raise

    poller = WorkerPoller(backend=cast(Any, backend), worker_id="w-attempt", executor=exec_t2)
    await poller.execute_task(
        ClaimedTask(
            operation_id=op_id,
            task_dict={"type": "import_documents", "operation_id": op_id, "bank_id": "bank-att"},
            schema=None,
            claimed_at=claimed_t2,
        )
    )
    await asyncio.wait_for(started_t2.wait(), timeout=2.0)

    assert poller.is_operation_active(op_id, None) is True
    # Stale cancel from wrong worker or older claimed_at must NOT cancel the newer active attempt
    assert (
        poller.cancel_active_operation(op_id, None, expected_worker_id="w-other", expected_claimed_at=claimed_t2)
        is False
    )
    assert (
        poller.cancel_active_operation(op_id, None, expected_worker_id="w-attempt", expected_claimed_at=claimed_t1)
        is False
    )
    assert poller.is_operation_active(op_id, None) is True

    # Calling _cleanup_task with an unrelated/older ActiveTaskInfo must NOT remove current attempt
    stale_info = poller._active_tasks[op_id]
    bogus_old_info: Any = SimpleNamespace()
    await poller._cleanup_task(op_id, "import_documents", bogus_old_info)
    assert poller._active_tasks.get(op_id) is stale_info

    # 2) MemoryEngine.retry_operation refuses 409 while poller reports active
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._operation_validator = None
    engine._authenticate_tenant = AsyncMock(return_value=None)
    engine._refuse_bank_wide_if_scoped = AsyncMock(return_value=None)
    engine._maybe_update_parent_operation = AsyncMock()
    canceller = MagicMock(return_value=True)
    engine._active_operation_canceller = None
    engine._active_bank_operations_canceller = None
    engine._finish_bank_delete = None
    engine._is_operation_active = None
    engine.register_worker_cancellers(cancel_operation=canceller, is_operation_active=poller.is_operation_active)

    with pytest.raises(OperationValidationError) as exc_info:
        await engine.retry_operation("bank-att", op_id, request_context=RequestContext(internal=True))
    assert exc_info.value.status_code == 409

    # 3) Matching attempt DOES cancel the active operation cleanly
    assert (
        poller.cancel_active_operation(op_id, None, expected_worker_id="w-attempt", expected_claimed_at=claimed_t2)
        is True
    )
    await asyncio.wait_for(cancelled_t2.wait(), timeout=2.0)
    assert await poller.wait_for_active_tasks(timeout=2.0) is True

    # 4) Pending cancel (worker_id is None) must NOT fire in-flight hook
    canceller.reset_mock()

    class _PendingCancelConn:
        @asynccontextmanager
        async def transaction(self):
            yield self

        async def fetchrow(self, query: str, *_args):
            if "SELECT bank_id, status" in query:
                return {"bank_id": "bank-att", "status": "pending"}
            if "UPDATE" in query:
                return {"status": "cancelled", "worker_id": None, "claimed_at": None}
            return None

    class _PendingBackend:
        _wraps_backend = True

        @asynccontextmanager
        async def acquire(self):
            yield _PendingCancelConn()

    engine._get_backend = AsyncMock(return_value=_PendingBackend())
    resp = await engine.cancel_operation("bank-att", op_id, request_context=RequestContext(internal=True))
    assert resp["success"] is True
    canceller.assert_not_called()


@pytest.mark.asyncio
async def test_await_executor_cancellable_cancels_once_and_keeps_tracked_until_flush_finishes(monkeypatch) -> None:
    import hindsight_api.worker.poller as poller_mod

    monkeypatch.setattr(poller_mod, "_CANCEL_DRAIN_TIMEOUT", 0.02)

    backend = _FakePollerBackend()
    started = asyncio.Event()
    allow_flush_finish = asyncio.Event()
    flushed = False
    cancel_signals = 0

    async def slow_cleanup_executor(_task_dict):
        nonlocal flushed, cancel_signals
        started.set()
        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancel_signals += 1
                # Slow finally cleanup (like retain's flush_document_bodies): takes longer than _CANCEL_DRAIN_TIMEOUT.
                await allow_flush_finish.wait()
                flushed = True
                raise

    poller = WorkerPoller(backend=cast(Any, backend), worker_id="w-once", executor=slow_cleanup_executor)
    op_id = str(uuid.uuid4())
    await poller.execute_task(
        ClaimedTask(
            operation_id=op_id,
            task_dict={"type": "batch_retain", "operation_id": op_id, "bank_id": "b-once"},
            schema=None,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2.0)

    assert poller.cancel_active_operation(op_id) is True
    # Wait past _CANCEL_DRAIN_TIMEOUT (0.02s) while cleanup is still blocked: task must stay tracked and NOT be re-cancelled.
    await asyncio.sleep(0.07)
    assert op_id in poller._active_tasks
    assert cancel_signals == 1
    assert flushed is False

    allow_flush_finish.set()
    assert await poller.wait_for_active_tasks(timeout=2.0) is True
    assert flushed is True
    assert cancel_signals == 1


@pytest.mark.asyncio
async def test_poller_folded_retain_individual_cancel_requeues_only_surviving_sibling() -> None:
    op_cancelled = str(uuid.uuid4())
    op_survivor = str(uuid.uuid4())
    statuses = {op_cancelled: "cancelled", op_survivor: "processing"}

    class _FoldBackend:
        backend_type = "postgresql"

        @asynccontextmanager
        async def acquire(self):
            class _Conn:
                @asynccontextmanager
                async def transaction(self):
                    yield self

                async def fetch(self, query: str, *args):
                    if "FROM" in query:
                        return [
                            {
                                "operation_id": oid,
                                "status": statuses[str(oid)],
                                "worker_id": "w-fold",
                                "claimed_at": None,
                            }
                            for oid in args[0]
                            if str(oid) in statuses
                        ]
                    return []

                async def execute(self, query: str, *args):
                    if "SET status = 'pending'" in query:
                        statuses[str(args[0])] = "pending"
                        return "UPDATE 1"
                    return "UPDATE 0"

            yield _Conn()

    started = asyncio.Event()

    async def fold_exec(_task_dict):
        started.set()
        await asyncio.sleep(60)

    poller = WorkerPoller(backend=cast(Any, _FoldBackend()), worker_id="w-fold", executor=fold_exec)
    await poller.execute_task(
        ClaimedTask(
            operation_id=op_cancelled,
            task_dict={"type": "batch_retain", "operation_id": op_cancelled, "bank_id": "b-fold"},
            schema=None,
            folded_operation_ids=[op_survivor],
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2.0)

    assert poller.cancel_active_operation(op_cancelled) is True
    assert await poller.wait_for_active_tasks(timeout=2.0) is True
    await _wait_until(lambda: statuses[op_survivor] == "pending")
    assert statuses[op_cancelled] == "cancelled"
    assert statuses[op_survivor] == "pending"


@pytest.mark.asyncio
async def test_restore_bank_tracks_target_deletes_stops_final_phases_and_cleans_partial_bank() -> None:
    src_bank = "src-restore"
    dst_bank = "dst-restore"
    dst_internal_id = str(uuid.uuid4())
    op_id = str(uuid.uuid4())

    # Part A: target bank delete cancels active import_bank and marks op failed (source row intact).
    statuses = {op_id: "processing"}

    class _TargetDeleteBackend:
        backend_type = "postgresql"
        _wraps_backend = True

        @asynccontextmanager
        async def acquire(self):
            class _Conn:
                @asynccontextmanager
                async def transaction(self):
                    yield self

                async def fetchval(self, query: str, *_args):
                    if "banks" in query:
                        return None  # target bank was deleted
                    return None

                async def fetch(self, *_args):
                    return []

                async def execute(self, query: str, *args):
                    if "SET status = 'failed'" in query:
                        statuses[str(args[0])] = "failed"
                        return "UPDATE 1"
                    return "UPDATE 0"

            yield _Conn()

    started = asyncio.Event()

    async def restore_exec(_task_dict):
        started.set()
        await asyncio.sleep(60)

    poller = WorkerPoller(backend=cast(Any, _TargetDeleteBackend()), worker_id="w-restore", executor=restore_exec)
    poller._maybe_update_parent_operation = AsyncMock()
    await poller.execute_task(
        ClaimedTask(
            operation_id=op_id,
            task_dict={"type": "import_bank", "operation_id": op_id, "bank_id": src_bank, "target_bank_id": dst_bank},
            schema=None,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2.0)

    assert poller.cancel_active_bank_operations(dst_bank, track_delete_gate=True) == 1
    poller.finish_bank_delete(dst_bank)
    assert await poller.wait_for_active_tasks(timeout=2.0) is True
    assert statuses[op_id] == "failed"

    # Part B: cancelled _handle_clone_bank skips completion metadata writes.
    clone_engine = MemoryEngine.__new__(MemoryEngine)
    clone_backend = SimpleNamespace(acquire=MagicMock())
    clone_engine._backend = clone_backend
    clone_engine._file_storage = SimpleNamespace()
    clone_engine._get_backend = AsyncMock(return_value=clone_backend)
    clone_engine._check_op_alive = AsyncMock(return_value=True)
    clone_engine.import_bank_async = AsyncMock(return_value=importer.BankImportResult(bank_id=dst_bank, cancelled=True))

    async def _empty_stream(*_args, **_kwargs):
        if False:
            yield b""

    with patch("hindsight_api.engine.transfer.stream_export_bank", _empty_stream):
        await clone_engine._handle_clone_bank({"bank_id": src_bank, "target_bank_id": dst_bank, "operation_id": op_id})
    clone_backend.acquire.assert_not_called()

    # Part C: import_bank_async cleans up partial restored bank only when internal_id matches.
    meta_store: dict[str, object] = {}

    class _ImportCleanupConn:
        def __init__(self, current_internal: str | None) -> None:
            self._current = current_internal

        def parse_json(self, v):
            return v

        async def fetchval(self, query: str, *_args):
            if "FROM" in query and "banks" in query:
                return self._current
            if "FROM" in query and "async_operations" in query:
                return dict(meta_store)
            return None

        async def execute(self, query: str, *args):
            if "SET result_metadata" in query:
                meta_store.clear()
                meta_store.update(json.loads(args[0]))
            return "UPDATE 1"

    async def fake_cancelled_import_bank(**kwargs):
        await kwargs["on_bank_created"](_ImportCleanupConn(dst_internal_id))
        return importer.BankImportResult(bank_id=dst_bank, cancelled=True)

    current_bank_id_val: list[str | None] = [dst_internal_id]

    class _ImportCleanupBackend:
        _wraps_backend = True

        @asynccontextmanager
        async def acquire(self):
            yield _ImportCleanupConn(current_bank_id_val[0])

    imp_engine = _make_stub_engine_for_delete()
    imp_engine._get_backend = AsyncMock(return_value=_ImportCleanupBackend())
    imp_engine._file_storage = SimpleNamespace()
    imp_engine._config_resolver = SimpleNamespace(resolve_full_config=AsyncMock())
    imp_engine.delete_bank = AsyncMock()
    imp_engine.embeddings = MagicMock()
    imp_engine.entity_resolver = MagicMock()
    imp_engine._format_readable_date = lambda d: ""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"schema_version": 1, "source_bank_id": src_bank, "archive_type": "bank"}),
        )
        zf.writestr("banks.json", json.dumps([{"bank_id": src_bank, "name": src_bank}]))
        zf.writestr("mental_models.json", "[]")
        zf.writestr("directives.json", "[]")
        zf.writestr("webhooks.json", "[]")
        zf.writestr("knowledge_pages.json", "[]")
        zf.writestr("attachments.json", "[]")

    with (
        patch("hindsight_api.engine.transfer.import_bank", side_effect=fake_cancelled_import_bank),
        patch("hindsight_api.engine.retain.bank_utils.get_bank_profile_if_exists", AsyncMock(return_value=None)),
        patch("hindsight_api.engine.retain.bank_utils.validate_new_bank_id"),
    ):
        res = await imp_engine.import_bank_async(
            buf.getvalue(), RequestContext(internal=True), target_bank_id=dst_bank, operation_id=op_id
        )
    assert res.cancelled is True
    imp_engine.delete_bank.assert_awaited_once_with(
        dst_bank, request_context=RequestContext(internal=True), _signal_in_flight=False
    )

    # Replacement bank with different internal_id must NOT be deleted!
    imp_engine.delete_bank.reset_mock()
    current_bank_id_val[0] = str(uuid.uuid4())
    await imp_engine._cleanup_partial_restored_bank(dst_bank, dst_internal_id, RequestContext(internal=True), op_id)
    imp_engine.delete_bank.assert_not_called()

    # Part D: _restore_bank checks cancellation in final restore phase and rolls back transaction.
    rolled_back = False
    probe_calls = 0
    restored_internal_id = dst_internal_id

    async def fake_restore_rows(_conn, table, rows, **_kwargs):
        nonlocal restored_internal_id
        rows_list = list(rows)
        if table == "banks" and rows_list and "internal_id" in rows_list[0]:
            restored_internal_id = str(rows_list[0]["internal_id"])
        return len(rows_list)

    async def late_cancel() -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return probe_calls <= 3  # Alive until final tx step 1

    class _TxConn:
        @asynccontextmanager
        async def transaction(self):
            nonlocal rolled_back
            try:
                yield self
            except BaseException:
                rolled_back = True
                raise

        async def fetchval(self, query: str, *_args):
            if "SELECT 1 FROM" in query:
                return None
            if "SELECT internal_id FROM" in query:
                return restored_internal_id
            return None

    class _RestoreBackend:
        _wraps_backend = True
        ops = SimpleNamespace()

        @asynccontextmanager
        async def acquire(self):
            yield _TxConn()

    fake_store = SimpleNamespace(ensure_bank_storage=AsyncMock())
    with (
        zipfile.ZipFile(io.BytesIO(buf.getvalue()), "r") as zf,
        patch("hindsight_api.engine.memories.get_memories", return_value=fake_store),
        patch("hindsight_api.engine.bank_aliases.refuse_alias_as_bank_id_on_conn", AsyncMock()),
        patch("hindsight_api.engine.retain.bank_utils.create_bank_row_on_conn", AsyncMock(return_value=True)),
        patch("hindsight_api.engine.retain.bank_utils.create_bank_vector_indexes", AsyncMock()),
        patch.object(importer, "_restore_rows", side_effect=fake_restore_rows),
        patch.object(importer, "_regenerate_mental_model_embeddings", AsyncMock(return_value={})),
        patch.object(importer, "_replay_documents", AsyncMock(return_value=importer.ImportResult())),
    ):
        restore_res = await importer._restore_bank(
            zf,
            backend=_RestoreBackend(),
            embeddings_model=MagicMock(),
            entity_resolver=MagicMock(),
            resolve_config=AsyncMock(return_value=SimpleNamespace()),
            format_date_fn=lambda d: "",
            target_bank_id=dst_bank,
            scope=None,
            file_storage=None,
            ops=SimpleNamespace(),
            on_bank_created=None,
            check_alive=late_cancel,
        )
    assert restore_res.cancelled is True
    assert rolled_back is True


@pytest.mark.asyncio
async def test_import_documents_retry_skip_recovers_checkpoint_for_observations(monkeypatch) -> None:
    monkeypatch.setattr(importer, "_DOCUMENT_BATCH_SIZE", 1)
    archive_bytes = _build_transfer_zip(["doc-1"], with_observations=True)
    op_id = str(uuid.uuid4())
    op_row_meta: dict[str, object] = {}
    existing_docs: set[str] = set()
    imported_observations: list[str] = []

    class _CkptConn:
        async def fetchval(self, query: str, *_args):
            if "FROM" in query and "async_operations" in query:
                return json.dumps(op_row_meta)
            return None

        async def execute(self, query: str, *args):
            if "SET result_metadata" in query:
                op_row_meta.clear()
                op_row_meta.update(json.loads(args[0]))
            return "UPDATE 1"

    class _CkptBackend:
        _wraps_backend = True
        ops = SimpleNamespace()

        @asynccontextmanager
        async def acquire(self):
            yield _CkptConn()

    engine = MemoryEngine.__new__(MemoryEngine)
    backend = _CkptBackend()
    engine._get_backend = AsyncMock(return_value=backend)
    engine._ensure_bank_exists = AsyncMock()
    engine._config_resolver = SimpleNamespace(resolve_full_config=AsyncMock(return_value=SimpleNamespace()))
    engine._build_retain_outbox_callback_factory = MagicMock(return_value=None)
    engine._report_imported_documents = AsyncMock()
    engine._submit_post_insert_maintenance = AsyncMock()
    engine._file_storage = SimpleNamespace(delete=AsyncMock())
    engine.embeddings = MagicMock()
    engine.entity_resolver = MagicMock()
    engine._format_readable_date = lambda d: ""

    @asynccontextmanager
    async def retrieve(_key: str):
        yield io.BytesIO(archive_bytes)

    engine._retrieve_archive = retrieve  # type: ignore[method-assign]

    async def fake_prepare(_emb, _fmt, batch):
        return batch

    async def fake_doc_batch(**kwargs):
        prepared = kwargs["prepared"]
        out = []
        for _doc, target_id in prepared:
            existing_docs.add(target_id)
            out.append(SimpleNamespace(unit_ids=[str(uuid.uuid4())], original_ordinals=[0]))
        return out

    async def fake_obs_import(**kwargs):
        resolved = kwargs["resolved"]
        for obs, source_ids in resolved:
            assert len(source_ids) == 1
            imported_observations.append(obs.text)
        return importer._ObservationOutcome(imported=len(resolved))

    fake_store = SimpleNamespace(
        transfer_document_exists=AsyncMock(side_effect=lambda **kw: kw["document_id"] in existing_docs),
        import_transfer_observations=AsyncMock(side_effect=fake_obs_import),
    )

    # Attempt 1: alive for batch check, cancelled right before observations
    alive_checks = 0

    async def cancel_before_obs(_op_id: str) -> bool:
        nonlocal alive_checks
        alive_checks += 1
        return alive_checks == 1

    engine._check_op_alive = cancel_before_obs  # type: ignore[method-assign]

    with (
        patch("hindsight_api.engine.memories.get_memories", return_value=fake_store),
        patch.object(importer, "_prepare_documents", side_effect=fake_prepare),
        patch.object(importer, "_import_document_batch", side_effect=fake_doc_batch),
        patch.object(importer, "_embed_in_batches", AsyncMock(return_value=[[0.1, 0.2]])),
    ):
        await engine._handle_import_documents(
            {"bank_id": "b-skip", "storage_key": "k.zip", "on_conflict": "skip", "operation_id": op_id}
        )
        assert "_import_checkpoint" in op_row_meta
        assert imported_observations == []

        # Attempt 2 (retry): alive all the way through; doc-1 is skipped, observations STILL import via checkpoint!
        engine._check_op_alive = AsyncMock(return_value=True)
        await engine._handle_import_documents(
            {"bank_id": "b-skip", "storage_key": "k.zip", "on_conflict": "skip", "operation_id": op_id}
        )

    assert imported_observations == ["Obs 1"]
    assert "_import_checkpoint" not in op_row_meta
    assert op_row_meta["observations_imported"] == 1
    engine._file_storage.delete.assert_awaited_once_with("k.zip")


@pytest.mark.asyncio
async def test_delete_bank_post_commit_release_failure_preserves_cleanup_metadata() -> None:
    engine = _make_stub_engine_for_delete()
    tx_runs = 0
    dropped_indexes: list[str] = []

    class _PostCommitReleaseBackend:
        _wraps_backend = True
        ops = SimpleNamespace(drop_bank_vector_indexes=AsyncMock())

        @asynccontextmanager
        async def acquire(self):
            nonlocal tx_runs
            tx_runs += 1
            yield _StubDeleteConn()
            if tx_runs == 1:
                raise ConnectionError("pool.release failed after commit")

    backend = _PostCommitReleaseBackend()
    engine._backend = backend
    engine._get_backend = AsyncMock(return_value=backend)

    store = SimpleNamespace(
        store_owned_for=MagicMock(return_value=True),
        count_bank_contents=AsyncMock(return_value=SimpleNamespace(memory_units=2, entities=1, documents=1)),
        purge_bank_rows=AsyncMock(return_value=["legacy/file.txt"]),
        drop_bank_storage=AsyncMock(),
    )

    async def capture_drop(_conn, internal_id, **_kwargs):
        dropped_indexes.append(internal_id)

    with (
        patch("hindsight_api.engine.retain.bank_utils.bank_kind", AsyncMock(return_value="memory")),
        patch("hindsight_api.engine.retain.bank_utils.drop_bank_vector_indexes", side_effect=capture_drop),
        patch("hindsight_api.engine.memories.get_memories", return_value=store),
    ):
        result = await engine.delete_bank("b-post", request_context=RequestContext(internal=True))

    assert tx_runs == 2  # 1 delete_tx (no retry!) + 1 drop_bank_vector_indexes acquire
    assert result["bank_deleted"] is True
    assert len(dropped_indexes) == 1
    engine._delete_files_quietly.assert_awaited_once()
    store.drop_bank_storage.assert_awaited_once_with("b-post")


@pytest.mark.asyncio
async def test_delete_operation_and_purge_expired_clean_import_zip_and_partial_bank() -> None:
    deleted_keys: list[str] = []
    partial_bank_internal = str(uuid.uuid4())
    op_doc = str(uuid.uuid4())
    op_bank = str(uuid.uuid4())
    delete_queue = [
        {
            "operation_id": uuid.UUID(op_doc),
            "operation_type": "import_documents",
            "status": "cancelled",
            "result_metadata": None,
            "task_payload": {"storage_key": "banks/b/imports/doc.zip"},
        },
        {
            "operation_id": uuid.UUID(op_bank),
            "operation_type": "import_bank",
            "status": "failed",
            "result_metadata": {"restored_bank_id": "partial-b", "restored_bank_internal_id": partial_bank_internal},
            "task_payload": json.dumps({"storage_key": "banks/src/imports/bank.zip"}),
        },
    ]

    class _DelOpConn:
        async def fetchrow(self, query: str, *_args):
            if "DELETE FROM" in query:
                return delete_queue.pop(0)
            return None

        async def fetchval(self, query: str, *_args):
            if "banks" in query:
                return partial_bank_internal
            return None

    class _DelOpBackend:
        _wraps_backend = True

        @asynccontextmanager
        async def acquire(self):
            yield _DelOpConn()

    engine = MemoryEngine.__new__(MemoryEngine)
    engine._operation_validator = None
    engine._authenticate_tenant = AsyncMock(return_value=None)
    engine._refuse_bank_wide_if_scoped = AsyncMock(return_value=None)
    engine._get_backend = AsyncMock(return_value=_DelOpBackend())
    engine.delete_bank = AsyncMock()
    engine._file_storage = SimpleNamespace(delete=AsyncMock(side_effect=lambda k: deleted_keys.append(k)))

    await engine.delete_operation("b", op_doc, request_context=RequestContext(internal=True))
    await engine.delete_operation("src", op_bank, request_context=RequestContext(internal=True))

    engine.delete_bank.assert_awaited_once_with(
        "partial-b", request_context=RequestContext(internal=True), _signal_in_flight=False
    )

    purge_conn = SimpleNamespace(
        fetch=AsyncMock(
            return_value=[
                {
                    "operation_type": "import_documents",
                    "status": "cancelled",
                    "result_metadata": None,
                    "task_payload": {"storage_key": "banks/b/imports/expired.zip"},
                },
                {
                    "operation_type": "export_bank",
                    "status": "completed",
                    "result_metadata": {"storage_key": "banks/b/exports/expired.zip"},
                    "task_payload": {},
                },
            ]
        )
    )
    purged = await engine.purge_expired_export_archives(
        purge_conn, "async_operations", datetime.now(timezone.utc), batch_size=10
    )
    assert purged == 2
    assert deleted_keys == [
        "banks/b/imports/doc.zip",
        "banks/src/imports/bank.zip",
        "banks/b/imports/expired.zip",
        "banks/b/exports/expired.zip",
    ]


@pytest.mark.asyncio
async def test_api_delete_operation_cancels_inflight_transfer_end_to_end() -> None:
    import httpx

    from hindsight_api.api import create_app

    bank_id = "bank-api-cancel"
    op_id = str(uuid.uuid4())
    op_state = {
        "operation_id": uuid.UUID(op_id),
        "bank_id": bank_id,
        "operation_type": "import_documents",
        "status": "processing",
        "worker_id": "w-api",
        "claimed_at": datetime(2026, 11, 22, 16, 10, tzinfo=timezone.utc),
        "created_at": datetime(2026, 11, 22, 16, 9, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 11, 22, 16, 10, tzinfo=timezone.utc),
        "completed_at": None,
        "error_message": None,
        "result_metadata": {},
        "retry_count": 0,
        "next_retry_at": None,
    }

    class _ApiConn:
        def parse_json(self, v):
            return v

        @asynccontextmanager
        async def transaction(self):
            yield self

        async def fetchrow(self, query: str, *_args):
            if "SELECT bank_id, status FROM" in query:
                return {"bank_id": op_state["bank_id"], "status": op_state["status"]}
            if "UPDATE" in query and "SET status = 'cancelled'" in query:
                op_state["status"] = "cancelled"
                return {
                    "status": "cancelled",
                    "worker_id": op_state["worker_id"],
                    "claimed_at": op_state["claimed_at"],
                }
            if "SELECT operation_id, operation_type" in query:
                return dict(op_state)
            return None

        async def fetch(self, *_args):
            return []

        async def fetchval(self, *_args):
            return None

        async def execute(self, *_args):
            return "UPDATE 0"

    class _ApiBackend:
        backend_type = "postgresql"
        _wraps_backend = True

        @asynccontextmanager
        async def acquire(self):
            yield _ApiConn()

    started = asyncio.Event()
    interrupted = asyncio.Event()

    async def transfer_exec(_task_dict):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            interrupted.set()
            raise

    backend = _ApiBackend()
    poller = WorkerPoller(backend=cast(Any, backend), worker_id="w-api", executor=transfer_exec)

    engine = _make_stub_engine_for_delete()
    engine._audit_logger = None
    engine._authorize_bank_read = AsyncMock(return_value=None)
    engine._maybe_update_parent_operation = AsyncMock()
    engine._backend = backend
    engine._get_backend = AsyncMock(return_value=backend)
    engine.register_worker_cancellers(
        cancel_operation=poller.cancel_active_operation,
        cancel_bank_operations=poller.cancel_active_bank_operations,
        finish_bank_delete=poller.finish_bank_delete,
        is_operation_active=poller.is_operation_active,
    )

    await poller.execute_task(
        ClaimedTask(
            operation_id=op_id,
            task_dict={"type": "import_documents", "operation_type": "import_documents", "bank_id": bank_id},
            schema=None,
            claimed_at=op_state["claimed_at"],
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2.0)

    app = create_app(engine, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        del_resp = await client.delete(f"/v1/default/banks/{bank_id}/operations/{op_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["success"] is True

        await asyncio.wait_for(interrupted.wait(), timeout=2.0)
        assert await poller.wait_for_active_tasks(timeout=2.0) is True

        get_resp = await client.get(f"/v1/default/banks/{bank_id}/operations/{op_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "cancelled"
