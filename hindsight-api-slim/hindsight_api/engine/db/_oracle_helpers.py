"""Internal helpers shared by ``oracle.py`` for timeout, cancellation, and session cleanup."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid as _uuid_mod
from typing import Any

logger = logging.getLogger(__name__)

ORACLE_TIMEOUT_CODES = ("DPY-4024", "ORA-03156")


def is_oracle_timeout(exc: BaseException) -> bool:
    """Return whether ``exc`` represents an asyncio or Oracle driver call timeout."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    msg = str(exc)
    return any(code in msg for code in ORACLE_TIMEOUT_CODES)


async def fetch_with_conn_timeout(conn: Any, query: str, params: list[Any], timeout: float) -> list[Any]:
    """Fetch query rows using Oracle ``call_timeout`` when available, or ``asyncio.wait_for`` otherwise."""
    if getattr(conn, "backend_type", None) == "oracle":
        return await conn.fetch(query, *params, timeout=timeout)
    return await asyncio.wait_for(conn.fetch(query, *params), timeout=timeout)


async def fetch_fallback_with_timeout(conn: Any, query: str, params: list[Any], timeout: float) -> list[Any]:
    """Run fallback query; on Oracle, bound it via ``timeout`` and return ``[]`` if that times out too."""
    if getattr(conn, "backend_type", None) != "oracle":
        return await conn.fetch(query, *params)
    try:
        return await conn.fetch(query, *params, timeout=timeout)
    except Exception as exc:
        if is_oracle_timeout(exc):
            return []
        raise


async def run_with_call_scope_or_timeout(conn: Any, fn: Any, args: tuple[Any, ...], timeout: float) -> Any:
    """Invoke ``fn(*args)`` bounded by ``conn._call_scope(timeout)`` if supported, else ``asyncio.wait_for``."""
    call_scope = getattr(conn, "_call_scope", None)
    if callable(call_scope):
        async with call_scope(timeout):
            return await fn(*args)
    return await asyncio.wait_for(fn(*args), timeout=timeout)


def apply_call_timeout(raw_conn: Any, timeout: float | None) -> int | None:
    """Set millisecond ``call_timeout`` on ``raw_conn`` and return its previous value, if any."""
    if timeout is None or not hasattr(raw_conn, "call_timeout"):
        return None
    prev = raw_conn.call_timeout
    raw_conn.call_timeout = max(1, int(timeout * 1000))
    return prev


def restore_call_timeout(wrapper: Any, raw_conn: Any, prev_timeout: int | None) -> None:
    """Restore ``call_timeout`` unless the connection was invalidated by cancellation."""
    if prev_timeout is None or getattr(wrapper, "_closed_on_cancel", False):
        return
    try:
        raw_conn.call_timeout = prev_timeout
    except Exception:
        pass


def mark_conn_cancelled(wrapper: Any, raw_conn: Any) -> None:
    """Mark both wrapper and underlying raw connection unusable after mid-call cancellation."""
    wrapper._closed_on_cancel = True
    try:
        raw_conn._closed_on_cancel = True
    except Exception:
        pass


def is_conn_cancelled(wrapper: Any, raw_conn: Any) -> bool:
    """Return whether wrapper or raw connection was invalidated by cancellation."""
    if getattr(wrapper, "_closed_on_cancel", False):
        return True
    return bool(getattr(raw_conn, "_closed_on_cancel", False))


def session_outcome(wrapper: Any, raw_conn: Any, flags: tuple[bool, bool, bool]) -> str:
    """Compute cleanup mode ('drop', 'release_quiet', or 'release') for an ending session."""
    broken, committed, unwinding = flags
    if broken:
        return "drop"
    if is_conn_cancelled(wrapper, raw_conn):
        return "drop"
    if committed:
        return "release_quiet"
    return "release_quiet" if unwinding else "release"


def apply_remaining_call_timeout(raw_conn: Any, deadline: float | None) -> None:
    """Enforce cumulative query timeout across multi-roundtrip driver calls."""
    if deadline is None:
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Oracle query exceeded timeout")
    if hasattr(raw_conn, "call_timeout"):
        raw_conn.call_timeout = max(1, int(remaining * 1000))


def close_cursor_safely(wrapper: Any, cursor: Any) -> None:
    """Close cursor, suppressing teardown errors if the connection was cancelled mid-call."""
    if not getattr(wrapper, "_closed_on_cancel", False):
        cursor.close()
        return
    try:
        cursor.close()
    except Exception:
        pass


def coerce_uuid_value(val: Any) -> Any:
    """Convert 32-char hex strings or 16-byte RAW values to ``UUID`` when valid."""
    if isinstance(val, str) and len(val) == 32:
        return _safe_uuid(hex=val, fallback=val)
    if isinstance(val, bytes) and len(val) == 16:
        return _safe_uuid(bytes=val, fallback=val)
    return val


def _safe_uuid(*, fallback: Any, **kwargs: Any) -> Any:
    try:
        return _uuid_mod.UUID(**kwargs)
    except Exception:
        return fallback


async def execute_ignoring_dup(cursor: Any, query: str, params: dict[str, Any]) -> bool:
    """Execute a dup-ignoring INSERT; return False when ORA-00001 suppresses the row."""
    try:
        await cursor.execute(query, params)
        return True
    except Exception as e:
        if "ORA-00001" in str(e):
            return False
        raise


async def _fetch_batches_within_deadline(raw_conn: Any, fetchmany: Any, deadline: float) -> list[Any]:
    rows: list[Any] = []
    while True:
        apply_remaining_call_timeout(raw_conn, deadline)
        batch = await fetchmany()
        if not batch:
            return rows
        rows.extend(batch)


async def fetch_rows_within_deadline(raw_conn: Any, cursor: Any, deadline: float | None) -> list[Any]:
    """Fetch all cursor rows while enforcing ``deadline`` across multi-roundtrip batches."""
    if deadline is None:
        return list(await cursor.fetchall())
    fetchmany = getattr(cursor, "fetchmany", None)
    if callable(fetchmany):
        return await _fetch_batches_within_deadline(raw_conn, fetchmany, deadline)
    if time.monotonic() >= deadline:
        raise TimeoutError("Oracle query exceeded timeout")
    return list(await cursor.fetchall())


async def _release_or_drop_on_error(release_or_drop: Any, pool: Any, conn: Any) -> None:
    try:
        await release_or_drop(pool, conn, broken=False)
    except BaseException as rel_err:
        logger.warning(f"Oracle pool.release failed; discarding connection: {rel_err!r}")
        await release_or_drop(pool, conn, broken=True)
        if isinstance(rel_err, asyncio.CancelledError):
            raise


async def finalize_session_conn(release_or_drop: Any, pool: Any, conn: Any, outcome: str) -> None:
    """Drop broken connections or release healthy ones back to the pool."""
    if outcome == "drop":
        await release_or_drop(pool, conn, broken=True)
        return
    if outcome == "release_quiet":
        await _release_or_drop_on_error(release_or_drop, pool, conn)
        return
    await release_or_drop(pool, conn, broken=False)
