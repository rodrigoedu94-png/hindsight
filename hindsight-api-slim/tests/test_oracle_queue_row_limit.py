"""Deterministic checks for version-gated Oracle queue row limiting.

These checks do not connect to Oracle or start a worker. Live lock/concurrency
semantics require a separate, disposable Oracle database.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.oracle import OracleConnection, _rewrite_pg_to_oracle


def _normalized(query: str) -> str:
    return " ".join(query.split())


@pytest.mark.parametrize("limit", ["5", "$2"])
@pytest.mark.parametrize("limit_after_lock", [False, True])
def test_native_limit_is_after_ordering_and_before_locking(limit: str, limit_after_lock: bool) -> None:
    lock = "FOR UPDATE SKIP LOCKED"
    tail = f"{lock} LIMIT {limit}" if limit_after_lock else f"LIMIT {limit} {lock}"
    sql = f"SELECT id FROM async_operations WHERE bank_id = $1 ORDER BY created_at {tail}"

    query = _normalized(_rewrite_pg_to_oracle(sql, supports_for_update_row_limit=True).query)
    expected_limit = ":2" if limit == "$2" else "5"

    assert query == (
        "SELECT id FROM async_operations WHERE bank_id = :1 ORDER BY created_at "
        f"FETCH FIRST {expected_limit} ROWS ONLY FOR UPDATE SKIP LOCKED"
    )
    assert "ROWNUM" not in query


def test_rewriter_defaults_to_legacy_rownum_limit() -> None:
    query = _rewrite_pg_to_oracle(
        "SELECT id FROM async_operations WHERE bank_id = $1 ORDER BY created_at LIMIT $2 FOR UPDATE SKIP LOCKED"
    ).query

    assert "WHERE ROWNUM <= :2 AND bank_id = :1" in query
    assert "FETCH FIRST" not in query
    assert "FOR UPDATE SKIP LOCKED" in query


@pytest.mark.parametrize("offset", ["LIMIT 5 OFFSET 2", "OFFSET 2 LIMIT 5"])
def test_native_lock_row_limit_is_not_used_with_offset(offset: str) -> None:
    query = _rewrite_pg_to_oracle(
        f"SELECT id FROM async_operations WHERE status = 'pending' ORDER BY created_at {offset} FOR UPDATE SKIP LOCKED",
        supports_for_update_row_limit=True,
    ).query

    # Oracle 26ai's FOR UPDATE row limit does not support OFFSET. Keep the
    # existing fallback rather than generating a new, unsupported SQL shape.
    assert "FETCH FIRST" not in query
    assert "ROWNUM <= 5" in query


@pytest.mark.parametrize(
    "sql",
    [
        "WITH candidates AS (SELECT id FROM async_operations WHERE status = 'pending') "
        "SELECT id FROM candidates WHERE id > 0 ORDER BY id LIMIT 5 FOR UPDATE SKIP LOCKED",
        "SELECT DISTINCT id FROM async_operations WHERE status = 'pending' ORDER BY id LIMIT 5 FOR UPDATE SKIP LOCKED",
    ],
)
def test_native_lock_row_limit_rejects_ineligible_query_shapes(sql: str) -> None:
    query = _rewrite_pg_to_oracle(sql, supports_for_update_row_limit=True).query

    assert "FETCH FIRST" not in query
    assert "ROWNUM <= 5" in query


@pytest.mark.parametrize("native_limit", [False, True])
def test_vector_limits_remain_exact(native_limit: bool) -> None:
    query = _rewrite_pg_to_oracle(
        "SELECT id FROM memory_units WHERE bank_id = $2 ORDER BY embedding <=> $1::vector LIMIT 5",
        supports_for_update_row_limit=native_limit,
    ).query

    assert "ORDER BY VECTOR_DISTANCE(embedding, :1, COSINE) FETCH EXACT FIRST 5 ROWS ONLY" in query
    assert "ROWNUM" not in query


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("23.26.3.0.0", False),
        ("23.26.4.0.0", True),
        ("23.26.4.1.0", True),
        ("23.26.5.0.0", True),
        ("24.0.0.0.0", True),
        ("23.9.0.25.07", False),
        (None, False),
        ("", False),
        ("23.26", False),
        ("23.26.4-preview", False),
        ("not-a-version", False),
        (MagicMock(), False),
    ],
)
def test_connection_detects_capability_conservatively(version: object, expected: bool) -> None:
    raw = MagicMock()
    raw.version = version

    conn = OracleConnection(raw)

    assert conn._supports_for_update_row_limit is expected


def test_connection_without_version_uses_legacy_path() -> None:
    raw = MagicMock()
    del raw.version

    assert OracleConnection(raw)._supports_for_update_row_limit is False


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["23.26.3.0.0", "23.26.4.1.0"])
async def test_fetch_applies_detected_capability_to_executed_sql(version: str) -> None:
    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.description = None
    raw = MagicMock()
    raw.version = version
    raw.cursor.return_value = cursor
    conn = OracleConnection(raw)

    with patch("hindsight_api.engine.db.oracle._import_oracledb"):
        rows = await conn.fetch(
            "SELECT id FROM async_operations WHERE status = 'pending' ORDER BY created_at LIMIT 2 FOR UPDATE SKIP LOCKED"
        )

    assert rows == []
    executed = _normalized(cursor.execute.await_args.args[0])
    if version == "23.26.4.1.0":
        assert executed.endswith("ORDER BY created_at FETCH FIRST 2 ROWS ONLY FOR UPDATE SKIP LOCKED")
        assert "ROWNUM" not in executed
    else:
        assert "WHERE ROWNUM <= 2 AND status = 'pending'" in executed
        assert "FETCH FIRST" not in executed
    cursor.close.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["reserved", "shared", "consolidation", "like", "not_like", "fold"])
@pytest.mark.parametrize("already_claimed", [False, True])
async def test_actual_queue_queries_keep_guards_and_native_limit(kind: str, already_claimed: bool) -> None:
    ops = OracleOps()
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])
    table = "HS_QUEUE_TEST.async_operations"
    claimed = ["00000000-0000-0000-0000-000000000001"] if already_claimed else []

    if kind == "reserved":
        await ops._claim_reserved_tasks(conn, table, 3, "retain")
    elif kind == "shared":
        await ops._claim_shared_tasks(conn, table, claimed, 3)
    elif kind == "consolidation":
        await ops._claim_consolidation_plain(conn, table, claimed, 3)
    elif kind == "like":
        await ops._claim_consolidation_like(conn, table, claimed, 3, ["test-%"])
    elif kind == "not_like":
        await ops._claim_consolidation_not_like(conn, table, claimed, 3, ["test-%"])
    else:
        await ops.fetch_foldable_retain_peers(conn, table, "test-bank", "test-key", 3)

    original = conn.fetch.await_args.args[0]
    query = _normalized(_rewrite_pg_to_oracle(original, supports_for_update_row_limit=True).query)

    assert "LIMIT" not in query
    assert "ROWNUM" not in query
    assert "FETCH FIRST :" in query
    assert query.index("ORDER BY") < query.index("FETCH FIRST") < query.index("FOR UPDATE SKIP LOCKED")
    assert "status = 'pending'" in query
    assert "task_payload IS NOT NULL" in query
    assert "SYSTIMESTAMP" in query
    if kind == "fold":
        assert "bank_id = :1" in query
        assert "serialization_key = :2" in query
        assert "ORDER BY created_at, operation_id" in query
    else:
        assert "WHERE bank_peer.bank_id = o.bank_id" in query
        if kind in ("shared", "reserved"):
            assert "WHERE doc_peer.bank_id = o.bank_id" in query
