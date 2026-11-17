from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.db import create_database_backend
from hindsight_api.engine.db.oracle import OracleConnection, _rewrite_pg_to_oracle
from hindsight_api.engine.memories.postgres import PostgresMemories


@dataclass(frozen=True)
class _Bounds:
    event_date: datetime | None = None
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None


@dataclass
class _MemoryUnitRow:
    id: uuid.UUID
    bank_id: str
    text: str = ""
    source_memory_ids: list[str] | str = field(default_factory=list)
    proof_count: int = 1
    event_date: datetime | None = None
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)


@dataclass(frozen=True)
class _RecordedUpdate:
    query: str
    merged_text: str
    source_memory_ids: list[uuid.UUID]
    proof_count: int
    observation_id: uuid.UUID
    bank_id: str
    event_date: datetime | None
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime | None


@dataclass(frozen=True)
class _ExecutemanyCall:
    query: str
    rows: list[tuple[uuid.UUID, uuid.UUID]]


@dataclass(frozen=True)
class _StubCursor:
    input_sizes: dict[str, object] = field(default_factory=dict)

    def setinputsizes(self, **kwargs) -> None:
        self.input_sizes.update(kwargs)


def _merge_earliest(existing: datetime | None, incoming: datetime | None) -> datetime | None:
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    ex = existing if existing.tzinfo is not None else existing.replace(tzinfo=timezone.utc)
    inc = incoming if incoming.tzinfo is not None else incoming.replace(tzinfo=timezone.utc)
    return min(ex, inc)


def _merge_latest(existing: datetime | None, incoming: datetime | None) -> datetime | None:
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    ex = existing if existing.tzinfo is not None else existing.replace(tzinfo=timezone.utc)
    inc = incoming if incoming.tzinfo is not None else incoming.replace(tzinfo=timezone.utc)
    return max(ex, inc)


class _FakeOracleConn:
    backend_type = "oracle"

    def __init__(self, rows: dict[str, _MemoryUnitRow]):
        self.rows = rows
        self.live_ids: set[str] | None = None
        self.queries: list[str] = []
        self.rewritten_queries: list[str] = []
        self.executemany_calls: list[_ExecutemanyCall] = []
        self.updates: list[_RecordedUpdate] = []

    def _record(self, query: str) -> str:
        rewritten = _rewrite_pg_to_oracle(query).query
        self.queries.append(query)
        self.rewritten_queries.append(rewritten)
        return rewritten

    async def fetchrow(self, query: str, *args) -> _MemoryUnitRow | None:
        self._record(query)
        obs_id = str(args[0])
        row = self.rows.get(obs_id)
        if row is None or row.bank_id != args[1]:
            return None
        return row

    async def fetch(self, query: str, *args) -> list[_MemoryUnitRow]:
        self._record(query)
        upper = query.upper()
        if "= ANY($1::UUID[])" in upper:
            unit_ids = args[0]
            bank_id = args[1]
            live: list[_MemoryUnitRow] = []
            for uid in unit_ids:
                uid_str = str(uid)
                if self.live_ids is not None:
                    if uid_str in self.live_ids:
                        live.append(_MemoryUnitRow(id=uid, bank_id=bank_id))
                elif uid_str in self.rows and self.rows[uid_str].bank_id == bank_id:
                    live.append(_MemoryUnitRow(id=uid, bank_id=bank_id))
            return live
        if "IN ($2, $3)" in upper:
            bank_id = args[0]
            wanted = [str(args[1]), str(args[2])]
            return [
                self.rows[uid] for uid in sorted(set(wanted)) if uid in self.rows and self.rows[uid].bank_id == bank_id
            ]
        return []

    async def execute_rows_affected(self, query: str, *args) -> int:
        self._record(query)
        call = _RecordedUpdate(
            query=query,
            merged_text=args[0],
            source_memory_ids=args[1],
            proof_count=args[2],
            observation_id=args[3],
            bank_id=args[4],
            event_date=args[5],
            occurred_start=args[6],
            occurred_end=args[7],
            mentioned_at=args[8],
        )
        self.updates.append(call)
        row = self.rows.get(str(call.observation_id))
        if row is None or row.bank_id != call.bank_id:
            return 0
        row.text = call.merged_text
        row.source_memory_ids = [str(s) for s in call.source_memory_ids]
        row.proof_count = call.proof_count
        row.event_date = _merge_earliest(row.event_date, call.event_date)
        row.occurred_start = _merge_earliest(row.occurred_start, call.occurred_start)
        row.occurred_end = _merge_latest(row.occurred_end, call.occurred_end)
        row.mentioned_at = _merge_latest(row.mentioned_at, call.mentioned_at)
        return 1

    async def executemany(self, query: str, args: list[tuple[uuid.UUID, uuid.UUID]]) -> None:
        self._record(query)
        self.executemany_calls.append(_ExecutemanyCall(query=query, rows=list(args)))


@pytest.mark.asyncio
async def test_fold_sources_into_observation_oracle_avoids_pg_sql_and_syncs_junction() -> None:
    obs_id = str(uuid.UUID(int=10))
    src_existing = uuid.UUID(int=1)
    src_new = uuid.UUID(int=2)
    long_text = "x" * 5000
    merged_text = "y" * 5000
    existing_end = datetime(2026, 11, 15, 10, 0, tzinfo=timezone.utc)
    later_end = datetime(2026, 11, 17, 10, 0, tzinfo=timezone.utc)
    incoming_event = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)

    conn = _FakeOracleConn(
        {
            obs_id: _MemoryUnitRow(
                id=uuid.UUID(obs_id),
                bank_id="bank-1",
                text=long_text,
                source_memory_ids=json.dumps([str(src_existing)]),
                occurred_end=existing_end,
            )
        }
    )

    ok = await PostgresMemories({}).fold_sources_into_observation(
        conn=conn,
        fq_table=lambda name: name,
        bank_id="bank-1",
        observation_id=obs_id,
        expected_text=long_text,
        merged_text=merged_text,
        source_memory_ids=[src_existing, src_new],
        bounds=_Bounds(event_date=incoming_event, occurred_end=later_end),
        embeddings=object(),  # type: ignore[arg-type]
    )

    assert ok is True
    assert conn.rows[obs_id].text == merged_text
    assert conn.rows[obs_id].source_memory_ids == [str(src_existing), str(src_new)]
    assert conn.rows[obs_id].proof_count == 2
    assert conn.rows[obs_id].event_date == incoming_event
    assert conn.rows[obs_id].occurred_start is None
    assert conn.rows[obs_id].occurred_end == later_end

    # Only the newly folded source is inserted into observation_sources.
    assert len(conn.executemany_calls) == 1
    assert conn.executemany_calls[0].rows == [(uuid.UUID(obs_id), src_new)]

    # Verify rewritten SQL contains zero PG-only constructs and keeps COALESCE(:N, col) hints.
    for rewritten in conn.rewritten_queries:
        upper = rewritten.upper()
        for forbidden in ("UNNEST", "ARRAY_AGG", "RETURNING", "::UUID[]", "$1"):
            assert forbidden not in upper
    update_rewritten = conn.rewritten_queries[1]
    assert "COALESCE(:6, event_date)" in update_rewritten
    assert "COALESCE(:7, occurred_start)" in update_rewritten
    assert "COALESCE(:8, occurred_end)" in update_rewritten
    assert "COALESCE(:9, mentioned_at)" in update_rewritten
    assert "SYSTIMESTAMP" in update_rewritten


@pytest.mark.asyncio
async def test_fold_sources_into_observation_oracle_aborts_on_text_drift_without_writes() -> None:
    obs_id = str(uuid.UUID(int=11))
    conn = _FakeOracleConn(
        {
            obs_id: _MemoryUnitRow(
                id=uuid.UUID(obs_id),
                bank_id="bank-1",
                text="rewritten concurrently",
                source_memory_ids=[str(uuid.UUID(int=1))],
            )
        }
    )

    ok = await PostgresMemories({}).fold_sources_into_observation(
        conn=conn,
        fq_table=lambda name: name,
        bank_id="bank-1",
        observation_id=obs_id,
        expected_text="stale probe text",
        merged_text="merged",
        source_memory_ids=[uuid.UUID(int=2)],
        bounds=_Bounds(),
        embeddings=object(),  # type: ignore[arg-type]
    )

    assert ok is False
    assert conn.updates == []
    assert conn.executemany_calls == []


@pytest.mark.asyncio
async def test_fold_sources_into_observation_oracle_hints_null_timestamp_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "oracledb",
        SimpleNamespace(
            DB_TYPE_CLOB="CLOB",
            DB_TYPE_TIMESTAMP_TZ="TIMESTAMP_TZ",
            defaults=SimpleNamespace(fetch_lobs=False),
        ),
    )
    obs_id = str(uuid.UUID(int=12))
    conn = _FakeOracleConn(
        {
            obs_id: _MemoryUnitRow(
                id=uuid.UUID(obs_id),
                bank_id="bank-1",
                text="obs",
                source_memory_ids=[],
            )
        }
    )
    await PostgresMemories({}).fold_sources_into_observation(
        conn=conn,
        fq_table=lambda name: name,
        bank_id="bank-1",
        observation_id=obs_id,
        expected_text="obs",
        merged_text="obs+",
        source_memory_ids=[uuid.UUID(int=3)],
        bounds=_Bounds(),
        embeddings=object(),  # type: ignore[arg-type]
    )
    cursor = _StubCursor()
    update_rewritten = conn.rewritten_queries[1]
    OracleConnection._apply_clob_input_sizes(
        cursor,
        update_rewritten,
        {"2": json.dumps([str(uuid.UUID(int=3))]), "6": None, "7": None, "8": None, "9": None},
    )
    assert cursor.input_sizes == {
        "2": "CLOB",
        "6": "TIMESTAMP_TZ",
        "7": "TIMESTAMP_TZ",
        "8": "TIMESTAMP_TZ",
        "9": "TIMESTAMP_TZ",
    }


@pytest.mark.asyncio
async def test_fold_observation_into_twin_oracle_keeps_lock_order_and_normalizes_utc() -> None:
    src_live = uuid.UUID(int=100)
    src_dead = uuid.UUID(int=101)
    src_twin = uuid.UUID(int=102)
    obs_id = str(uuid.UUID(int=200))
    twin_id = str(uuid.UUID(int=201))
    naive_event = datetime(2026, 11, 16, 12, 0)
    twin_event = datetime(2026, 11, 17, 12, 0)

    conn = _FakeOracleConn(
        {
            str(src_live): _MemoryUnitRow(id=src_live, bank_id="bank-1"),
            str(src_twin): _MemoryUnitRow(id=src_twin, bank_id="bank-1"),
            obs_id: _MemoryUnitRow(
                id=uuid.UUID(obs_id),
                bank_id="bank-1",
                text="updated obs",
                source_memory_ids=[str(src_live), str(src_dead)],
                event_date=naive_event,
                occurred_end=datetime(2026, 11, 18, 8, 0),
            ),
            twin_id: _MemoryUnitRow(
                id=uuid.UUID(twin_id),
                bank_id="bank-1",
                text="twin obs",
                source_memory_ids=[str(src_twin)],
                event_date=twin_event,
            ),
        }
    )

    ok = await PostgresMemories({}).fold_observation_into_twin(
        conn=conn,
        fq_table=lambda name: name,
        bank_id="bank-1",
        observation_id=obs_id,
        observation_text="updated obs",
        twin_id=twin_id,
        twin_text="twin obs",
        merged_text="merged twin",
        embeddings=object(),  # type: ignore[arg-type]
    )

    assert ok is True
    assert conn.rows[twin_id].text == "merged twin"
    assert conn.rows[twin_id].source_memory_ids == [str(src_twin), str(src_live)]
    assert conn.rows[twin_id].proof_count == 2
    assert conn.rows[twin_id].event_date == naive_event.replace(tzinfo=timezone.utc)
    assert conn.rows[twin_id].occurred_end == datetime(2026, 11, 18, 8, 0, tzinfo=timezone.utc)
    assert conn.executemany_calls[0].rows == [(uuid.UUID(twin_id), src_live)]

    # Sources locked before observations; both source and observation locks use ORDER BY id FOR UPDATE.
    lock_queries = [q for q in conn.queries if "FOR " in q.upper()]
    assert len(lock_queries) == 2
    assert "ORDER BY ID" in lock_queries[0].upper()
    assert "FOR UPDATE" in lock_queries[0].upper()
    assert "ORDER BY ID" in lock_queries[1].upper()
    assert "FOR UPDATE" in lock_queries[1].upper()

    # Naive Oracle timestamp from obs_row was normalized to UTC before UPDATE bind.
    bound_event = conn.updates[0].event_date
    assert bound_event == naive_event.replace(tzinfo=timezone.utc)
    assert bound_event.tzinfo == timezone.utc

    for rewritten in conn.rewritten_queries:
        upper = rewritten.upper()
        for forbidden in ("UNNEST", "ARRAY_AGG", "RETURNING", "FROM MEMORY_UNITS U", "$1"):
            assert forbidden not in upper


@pytest.mark.asyncio
async def test_fold_observation_into_twin_oracle_returns_false_if_twin_or_obs_drifted() -> None:
    src_live = uuid.UUID(int=300)
    obs_id = str(uuid.UUID(int=301))
    twin_id = str(uuid.UUID(int=302))
    conn = _FakeOracleConn(
        {
            str(src_live): _MemoryUnitRow(id=src_live, bank_id="bank-1"),
            obs_id: _MemoryUnitRow(
                id=uuid.UUID(obs_id),
                bank_id="bank-1",
                text="obs",
                source_memory_ids=[str(src_live)],
            ),
            twin_id: _MemoryUnitRow(
                id=uuid.UUID(twin_id),
                bank_id="bank-1",
                text="twin drifted",
                source_memory_ids=[],
            ),
        }
    )

    ok = await PostgresMemories({}).fold_observation_into_twin(
        conn=conn,
        fq_table=lambda name: name,
        bank_id="bank-1",
        observation_id=obs_id,
        observation_text="obs",
        twin_id=twin_id,
        twin_text="expected twin",
        merged_text="merged",
        embeddings=object(),  # type: ignore[arg-type]
    )

    assert ok is False
    assert conn.updates == []
    assert conn.executemany_calls == []


@pytest.mark.asyncio
async def test_fold_observation_into_twin_oracle_returns_false_if_obs_drifted_before_lock() -> None:
    src_live = uuid.UUID(int=310)
    obs_id = str(uuid.UUID(int=311))
    twin_id = str(uuid.UUID(int=312))
    conn = _FakeOracleConn(
        {
            str(src_live): _MemoryUnitRow(id=src_live, bank_id="bank-1"),
            obs_id: _MemoryUnitRow(
                id=uuid.UUID(obs_id),
                bank_id="bank-1",
                text="obs",
                source_memory_ids=[str(src_live)],
            ),
            twin_id: _MemoryUnitRow(
                id=uuid.UUID(twin_id),
                bank_id="bank-1",
                text="twin obs",
                source_memory_ids=[],
            ),
        }
    )
    orig_fetch = conn.fetch

    async def _drift_during_source_lock(query: str, *args) -> list[_MemoryUnitRow]:
        res = await orig_fetch(query, *args)
        if "= ANY($1::UUID[])" in query.upper():
            conn.rows[obs_id].text = "obs rewritten before observation lock"
        return res

    conn.fetch = _drift_during_source_lock  # type: ignore[method-assign]

    ok = await PostgresMemories({}).fold_observation_into_twin(
        conn=conn,
        fq_table=lambda name: name,
        bank_id="bank-1",
        observation_id=obs_id,
        observation_text="obs",
        twin_id=twin_id,
        twin_text="twin obs",
        merged_text="merged",
        embeddings=object(),  # type: ignore[arg-type]
    )

    assert ok is False
    assert conn.updates == []
    assert conn.executemany_calls == []
    conn = AsyncMock()
    conn.backend_type = "postgresql"
    conn.fetchval = AsyncMock(return_value=uuid.UUID(int=1))

    ok = await PostgresMemories({}).fold_sources_into_observation(
        conn=conn,
        fq_table=lambda name: name,
        bank_id="bank-1",
        observation_id=str(uuid.UUID(int=1)),
        expected_text="old",
        merged_text="new",
        source_memory_ids=[uuid.UUID(int=2)],
        bounds=_Bounds(),
        embeddings=object(),  # type: ignore[arg-type]
    )

    assert ok is True
    assert "unnest(" in conn.fetchval.await_args.args[0].lower()


@pytest.mark.oracle
@pytest.mark.usefixtures("_oracle_backend")
@pytest.mark.asyncio
async def test_oracle_consolidation_folds_live_db(oracle_db_url: str) -> None:
    backend = create_database_backend("oracle")
    await backend.initialize(oracle_db_url, min_size=1, max_size=2)
    bank_id = f"test-fold-{uuid.uuid4().hex[:8]}"
    src1, src2, src3 = (uuid.uuid4() for _ in range(3))
    obs_twin, obs_updated = (uuid.uuid4() for _ in range(2))
    twin_text = "x" * 4500
    obs_text = "y" * 4500
    folded_text = "z" * 4500
    t_early = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
    t_mid = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    t_late = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
    store = PostgresMemories({})

    try:
        async with backend.acquire() as conn:
            await conn.execute("INSERT INTO banks (bank_id, name) VALUES ($1, $2)", bank_id, bank_id)
            for sid in (src1, src2, src3):
                await conn.execute(
                    "INSERT INTO memory_units (id, bank_id, text, fact_type, event_date) VALUES ($1, $2, $3, 'world', $4)",
                    sid,
                    bank_id,
                    f"source {sid}",
                    t_mid,
                )
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, occurred_start, source_memory_ids, proof_count) "
                "VALUES ($1, $2, $3, 'observation', $4, NULL, $5, 1)",
                obs_twin,
                bank_id,
                twin_text,
                t_mid,
                [src1],
            )
            await conn.execute(
                "INSERT INTO observation_sources (observation_id, source_id) VALUES ($1, $2)",
                obs_twin,
                src1,
            )
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, occurred_end, source_memory_ids, proof_count) "
                "VALUES ($1, $2, $3, 'observation', $4, $5, $6, 1)",
                obs_updated,
                bank_id,
                obs_text,
                t_early,
                t_late,
                [src3],
            )
            await conn.execute(
                "INSERT INTO observation_sources (observation_id, source_id) VALUES ($1, $2)",
                obs_updated,
                src3,
            )

        async with backend.transaction() as tx:
            ok1 = await store.fold_sources_into_observation(
                conn=tx,
                fq_table=lambda name: name,
                bank_id=bank_id,
                observation_id=str(obs_twin),
                expected_text=twin_text,
                merged_text=folded_text,
                source_memory_ids=[src1, src2],
                bounds=_Bounds(occurred_start=t_early, occurred_end=t_mid),
                embeddings=object(),  # type: ignore[arg-type]
            )
            assert ok1 is True

            ok2 = await store.fold_observation_into_twin(
                conn=tx,
                fq_table=lambda name: name,
                bank_id=bank_id,
                observation_id=str(obs_updated),
                observation_text=obs_text,
                twin_id=str(obs_twin),
                twin_text=folded_text,
                merged_text=folded_text,
                embeddings=object(),  # type: ignore[arg-type]
            )
            assert ok2 is True

        async with backend.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT text, proof_count, source_memory_ids, occurred_start, occurred_end FROM memory_units WHERE id = $1 AND bank_id = $2",
                obs_twin,
                bank_id,
            )
            assert row is not None
            assert row["text"] == folded_text
            assert row["proof_count"] == 3
            assert [str(s) for s in row["source_memory_ids"]] == [str(src1), str(src2), str(src3)]
            assert row["occurred_start"].replace(tzinfo=timezone.utc) == t_early
            assert row["occurred_end"].replace(tzinfo=timezone.utc) == t_late
            os_rows = await conn.fetch(
                "SELECT source_id FROM observation_sources WHERE observation_id = $1",
                obs_twin,
            )
            assert {str(r["source_id"]) for r in os_rows} == {str(src1), str(src2), str(src3)}
    finally:
        async with backend.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
        await backend.shutdown()
