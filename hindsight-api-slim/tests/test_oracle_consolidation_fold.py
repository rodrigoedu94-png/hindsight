from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.db.oracle import OracleConnection, _rewrite_pg_to_oracle
from hindsight_api.engine.memories.postgres import PostgresMemories


@dataclass(frozen=True)
class _Bounds:
    event_date: datetime | None = None
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None


class _FakeOracleConn:
    backend_type = "oracle"

    def __init__(self, rows: dict[str, dict]):
        self.rows = rows
        self.live_ids: set[str] | None = None
        self.queries: list[str] = []
        self.rewritten_queries: list[str] = []
        self.executemany_calls: list[tuple[str, list[tuple]]] = []
        self.updates: list[tuple[str, tuple]] = []

    def _record(self, query: str) -> str:
        rewritten = _rewrite_pg_to_oracle(query).query
        self.queries.append(query)
        self.rewritten_queries.append(rewritten)
        return rewritten

    async def fetchrow(self, query: str, *args):
        self._record(query)
        obs_id = str(args[0])
        row = self.rows.get(obs_id)
        if row is None or row.get("bank_id") != args[1]:
            return None
        return dict(row)

    async def fetch(self, query: str, *args):
        self._record(query)
        upper = query.upper()
        if "= ANY($1::UUID[])" in upper:
            unit_ids = args[0]
            bank_id = args[1]
            live = []
            for uid in unit_ids:
                uid_str = str(uid)
                if self.live_ids is not None:
                    if uid_str in self.live_ids:
                        live.append({"id": uid})
                elif uid_str in self.rows and self.rows[uid_str].get("bank_id") == bank_id:
                    live.append({"id": uid})
            return live
        if "IN ($2, $3)" in upper:
            bank_id = args[0]
            wanted = [str(args[1]), str(args[2])]
            matched = [
                dict(self.rows[uid])
                for uid in sorted(set(wanted))
                if uid in self.rows and self.rows[uid].get("bank_id") == bank_id
            ]
            return matched
        return []

    async def execute_rows_affected(self, query: str, *args) -> int:
        self._record(query)
        self.updates.append((query, args))
        obs_id = str(args[3])
        bank_id = args[4]
        row = self.rows.get(obs_id)
        if row is None or row.get("bank_id") != bank_id:
            return 0
        row["text"] = args[0]
        row["source_memory_ids"] = [str(s) for s in args[1]]
        row["proof_count"] = args[2]
        return 1

    async def executemany(self, query: str, args: list[tuple]) -> None:
        self._record(query)
        self.executemany_calls.append((query, list(args)))


@pytest.mark.asyncio
async def test_fold_sources_into_observation_oracle_avoids_pg_sql_and_syncs_junction() -> None:
    obs_id = str(uuid.UUID(int=10))
    src_existing = uuid.UUID(int=1)
    src_new = uuid.UUID(int=2)
    long_text = "x" * 5000
    merged_text = "y" * 5000

    conn = _FakeOracleConn(
        {
            obs_id: {
                "id": uuid.UUID(obs_id),
                "bank_id": "bank-1",
                "text": long_text,
                "source_memory_ids": json.dumps([str(src_existing)]),
            }
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
        bounds=_Bounds(event_date=datetime(2026, 11, 17, tzinfo=timezone.utc)),
        embeddings=object(),  # type: ignore[arg-type]
    )

    assert ok is True
    assert conn.rows[obs_id]["text"] == merged_text
    assert conn.rows[obs_id]["source_memory_ids"] == [str(src_existing), str(src_new)]
    assert conn.rows[obs_id]["proof_count"] == 2

    # Only the newly folded source is inserted into observation_sources.
    assert len(conn.executemany_calls) == 1
    assert conn.executemany_calls[0][1] == [(uuid.UUID(obs_id), src_new)]

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
            obs_id: {
                "id": uuid.UUID(obs_id),
                "bank_id": "bank-1",
                "text": "rewritten concurrently",
                "source_memory_ids": [str(uuid.UUID(int=1))],
            }
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
async def test_fold_sources_into_observation_oracle_hints_null_timestamp_params() -> None:
    class _StubCursor:
        def __init__(self) -> None:
            self.input_sizes: dict[str, object] = {}

        def setinputsizes(self, **kwargs) -> None:
            self.input_sizes.update(kwargs)

    import sys
    from types import SimpleNamespace

    monkeypatch = pytest.MonkeyPatch()
    try:
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
                obs_id: {
                    "id": uuid.UUID(obs_id),
                    "bank_id": "bank-1",
                    "text": "obs",
                    "source_memory_ids": [],
                }
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
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_fold_observation_into_twin_oracle_keeps_lock_order_and_normalizes_utc() -> None:
    src_live = uuid.UUID(int=100)
    src_dead = uuid.UUID(int=101)
    src_twin = uuid.UUID(int=102)
    obs_id = str(uuid.UUID(int=200))
    twin_id = str(uuid.UUID(int=201))
    naive_event = datetime(2026, 11, 16, 12, 0)

    conn = _FakeOracleConn(
        {
            str(src_live): {"id": src_live, "bank_id": "bank-1"},
            str(src_twin): {"id": src_twin, "bank_id": "bank-1"},
            obs_id: {
                "id": uuid.UUID(obs_id),
                "bank_id": "bank-1",
                "text": "updated obs",
                "source_memory_ids": [str(src_live), str(src_dead)],
                "event_date": naive_event,
                "occurred_start": None,
                "occurred_end": None,
                "mentioned_at": None,
            },
            twin_id: {
                "id": uuid.UUID(twin_id),
                "bank_id": "bank-1",
                "text": "twin obs",
                "source_memory_ids": [str(src_twin)],
                "event_date": datetime(2026, 11, 17, 12, 0),
                "occurred_start": None,
                "occurred_end": None,
                "mentioned_at": None,
            },
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
    assert conn.rows[twin_id]["text"] == "merged twin"
    assert conn.rows[twin_id]["source_memory_ids"] == [str(src_twin), str(src_live)]
    assert conn.rows[twin_id]["proof_count"] == 2
    assert conn.executemany_calls[0][1] == [(uuid.UUID(twin_id), src_live)]

    # Sources locked before observations; observations locked in ORDER BY id order.
    lock_queries = [q for q in conn.queries if "FOR " in q.upper()]
    assert len(lock_queries) == 2
    assert "FOR SHARE" in lock_queries[0].upper()
    assert "ORDER BY ID" in lock_queries[1].upper()
    assert "FOR UPDATE" in lock_queries[1].upper()

    # Naive Oracle timestamp from obs_row was normalized to UTC before UPDATE bind.
    bound_event = conn.updates[0][1][5]
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
            str(src_live): {"id": src_live, "bank_id": "bank-1"},
            obs_id: {
                "id": uuid.UUID(obs_id),
                "bank_id": "bank-1",
                "text": "obs",
                "source_memory_ids": [str(src_live)],
                "event_date": None,
                "occurred_start": None,
                "occurred_end": None,
                "mentioned_at": None,
            },
            twin_id: {
                "id": uuid.UUID(twin_id),
                "bank_id": "bank-1",
                "text": "twin drifted",
                "source_memory_ids": [],
                "event_date": None,
                "occurred_start": None,
                "occurred_end": None,
                "mentioned_at": None,
            },
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
async def test_postgres_fold_path_remains_default_when_not_oracle() -> None:
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
