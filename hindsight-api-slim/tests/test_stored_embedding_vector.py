"""Recall dedup reads stored embeddings back from either backend (Notion: Caderno de Memórias em AI §4).

Postgres returns pgvector text; Oracle's rewriter drops ``embedding::text`` and python-oracledb hands
the VECTOR column back as an ``array.array``. Parsing the latter as text raised, so dedup's cosine
pre-filter was silently off on Oracle.
"""

import array
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from hindsight_api.engine import memory_engine
from hindsight_api.engine.memory_engine import _stored_embedding_vector

EXPECTED = np.array([0.25, -0.5, 1.0], dtype=np.float32)


@pytest.mark.parametrize(
    "stored",
    [
        "[0.25,-0.5,1]",
        " [0.25, -0.5, 1.0] ",
        array.array("f", [0.25, -0.5, 1.0]),
        array.array("d", [0.25, -0.5, 1.0]),
        [0.25, -0.5, 1.0],
        np.array([0.25, -0.5, 1.0], dtype=np.float64),
    ],
    ids=["pgvector-text", "pgvector-text-spaced", "oracle-float32", "oracle-float64", "list", "numpy"],
)
def test_stored_embedding_vector_reads_every_backend_shape(stored):
    vector = _stored_embedding_vector(stored)
    assert vector.dtype == np.float32
    np.testing.assert_allclose(vector, EXPECTED)


@pytest.mark.asyncio
async def test_candidate_embeddings_returns_vectors_for_oracle_rows(monkeypatch):
    rows = [
        {"id": "u1", "embedding": array.array("f", [0.25, -0.5, 1.0]), "fact_type": "world"},
        {"id": "u2", "embedding": None, "fact_type": "world"},
    ]
    store = MagicMock()
    store.unit_embeddings = AsyncMock(return_value=rows)
    monkeypatch.setattr("hindsight_api.engine.memories.get_memories", lambda: store)

    class _Conn:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(memory_engine, "acquire_with_retry", lambda backend: _Conn())
    engine = MagicMock()
    engine._get_backend = AsyncMock(return_value=MagicMock())

    vectors = await memory_engine.MemoryEngine._candidate_embeddings(engine, "bank-1", ["u1", "u2"])

    assert vectors is not None and set(vectors) == {"u1"}
    np.testing.assert_allclose(vectors["u1"], EXPECTED)
