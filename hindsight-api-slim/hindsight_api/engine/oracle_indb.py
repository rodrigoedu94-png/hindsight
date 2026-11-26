"""Shared async Oracle AI Database 26ai helper for in-database ONNX models.

Both :class:`OracleInDbEmbeddings` (``VECTOR_EMBEDDING``) and
:class:`OracleInDbCrossEncoder` (``PREDICTION``) share this pool and query
helpers so they reuse one async connection pool per ``(event_loop, dsn)`` and
validate SQL model identifiers identically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from .db.oracle import _import_oracledb, _oracle_connect_params

logger = logging.getLogger(__name__)

_ORACLE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$#]{0,127}$")
_loop_pools: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, Any]] = weakref.WeakKeyDictionary()
_pool_init_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()


def validate_oracle_model_name(name: str) -> str:
    """Validate and normalize an Oracle mining/ONNX model identifier.

    ``VECTOR_EMBEDDING`` and ``PREDICTION`` require the model name as a SQL
    identifier rather than a bind variable, so strict identifier validation is
    mandatory before interpolating it into SQL.
    """
    cleaned = name.strip()
    if not _ORACLE_IDENT_RE.fullmatch(cleaned):
        raise ValueError(f"Invalid Oracle ONNX model name {name!r}: must match ^[A-Za-z_][A-Za-z0-9_$#]{{0,127}}$")
    return cleaned.upper()


class OracleInDbClient:
    """Lightweight per-loop async Oracle client for in-DB ONNX SQL calls."""

    def __init__(
        self,
        database_url: str,
        *,
        schema: str | None = None,
        max_concurrent: int = 4,
    ) -> None:
        self.database_url = database_url
        self.schema = schema if schema and schema.lower() != "public" else None
        self.max_concurrent = max(1, max_concurrent)

    async def _get_pool(self) -> Any:
        loop = asyncio.get_running_loop()
        pools = _loop_pools.setdefault(loop, {})
        if self.database_url in pools:
            return pools[self.database_url]

        lock = _pool_init_locks.setdefault(loop, asyncio.Lock())
        async with lock:
            if self.database_url in pools:
                return pools[self.database_url]
            oracledb = _import_oracledb()
            kwargs = _oracle_connect_params(self.database_url)
            pool = oracledb.create_pool_async(
                **kwargs,
                min=1,
                max=max(4, self.max_concurrent),
                stmtcachesize=50,
            )
            pools[self.database_url] = pool
            logger.info("OracleInDbClient: created async pool (max=%s)", max(4, self.max_concurrent))
            return pool

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        """Acquire a pooled async connection with ``current_schema`` applied."""
        pool = await self._get_pool()
        conn = await pool.acquire()
        try:
            if self.schema and hasattr(conn, "current_schema") and conn.current_schema != self.schema:
                conn.current_schema = self.schema
            yield conn
        finally:
            await pool.release(conn)

    async def verify_mining_model(self, model_name: str, expected_function: str) -> None:
        """Confirm *model_name* is loaded in Oracle with *expected_function*."""
        async with self.acquire() as conn:
            cur = conn.cursor()
            try:
                await cur.execute(
                    "SELECT mining_function FROM user_mining_models WHERE UPPER(model_name) = :m",
                    {"m": model_name.upper()},
                )
                row = await cur.fetchone()
                if row is None:
                    await cur.execute(
                        "SELECT mining_function FROM all_mining_models WHERE UPPER(model_name) = :m",
                        {"m": model_name.upper()},
                    )
                    row = await cur.fetchone()
            finally:
                cur.close()

        if row is None:
            raise RuntimeError(
                f"Oracle ONNX model {model_name!r} is not loaded in this database schema "
                f"(checked user_mining_models and all_mining_models). Load it first via DBMS_VECTOR.LOAD_ONNX_MODEL."
            )
        actual_fn = str(row[0]).upper()
        if actual_fn != expected_function.upper():
            raise RuntimeError(
                f"Oracle ONNX model {model_name!r} has mining_function={actual_fn!r}, expected {expected_function!r}."
            )

    async def fetch_single(self, sql: str, params: dict[str, Any]) -> Any:
        """Execute a single-row SQL query and return the first column."""
        async with self.acquire() as conn:
            cur = conn.cursor()
            try:
                await cur.execute(sql, params)
                row = await cur.fetchone()
                return row[0] if row is not None else None
            finally:
                cur.close()

    async def fetch_json_table_batch(self, sql: str, items: list[Any]) -> list[Any]:
        """Execute a ``JSON_TABLE(:payload, ...)`` batch query and return ordered column-1 values."""
        payload = json.dumps(items, ensure_ascii=False)
        oracledb = _import_oracledb()
        async with self.acquire() as conn:
            cur = conn.cursor()
            try:
                if len(payload.encode("utf-8")) > 32000:
                    cur.setinputsizes(payload=oracledb.DB_TYPE_CLOB)
                await cur.execute(sql, {"payload": payload})
                rows = await cur.fetchall()
                return [row[1] for row in rows]
            finally:
                cur.close()
