"""Oracle-specific memory store operations."""

from __future__ import annotations

from typing import Any


def is_oracle_conn(conn: Any) -> bool:
    """Return True when ``conn`` targets the Oracle backend."""
    return getattr(conn, "backend_type", "postgresql") == "oracle"


__all__ = ["is_oracle_conn"]
