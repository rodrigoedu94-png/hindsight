"""Unit tests for Oracle 26ai in-database ONNX embeddings and reranker providers."""

from __future__ import annotations

import array
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.cross_encoder import OracleInDbCrossEncoder, create_cross_encoder_from_env
from hindsight_api.engine.embeddings import OracleInDbEmbeddings, create_embeddings_from_env
from hindsight_api.engine.oracle_indb import validate_oracle_model_name


def test_validate_oracle_model_name():
    assert validate_oracle_model_name("hs_embed_1") == "HS_EMBED_1"
    assert validate_oracle_model_name("  HS_RERANK_1 ") == "HS_RERANK_1"
    for bad in ("", "1bad", "hs-embed-1", "HS_EMBED_1; DROP TABLE", "a b"):
        with pytest.raises(ValueError):
            validate_oracle_model_name(bad)


def test_factories_select_oracle_providers(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_DATABASE_BACKEND", "oracle")
    monkeypatch.setenv("HINDSIGHT_API_DATABASE_URL", "oracle://u:p@localhost:1521/FREEPDB1")
    monkeypatch.setenv("HINDSIGHT_API_DATABASE_SCHEMA", "HS26_ONNX")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "oracle")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ORACLE_MODEL", "HS_EMBED_1")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "oracle")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_ORACLE_MODEL", "HS_RERANK_1")

    cfg = HindsightConfig.from_env()
    with patch("hindsight_api.config.get_config", return_value=cfg):
        emb = create_embeddings_from_env()
        ce = create_cross_encoder_from_env()

    assert isinstance(emb, OracleInDbEmbeddings)
    assert emb.provider_name == "oracle"
    assert emb.model_name == "HS_EMBED_1"
    assert isinstance(ce, OracleInDbCrossEncoder)
    assert ce.provider_name == "oracle"
    assert ce.model_name == "HS_RERANK_1"
    assert ce.retry_policy is not None


@pytest.mark.asyncio
async def test_oracle_indb_embeddings_batch_and_single():
    emb = OracleInDbEmbeddings(
        database_url="oracle://u:p@localhost:1521/FREEPDB1", model_name="HS_EMBED_1", batch_size=2
    )
    emb._client = AsyncMock()
    emb._client.verify_mining_model = AsyncMock(return_value=None)
    emb._client.fetch_single = AsyncMock(return_value=array.array("f", [0.1, 0.2, 0.3]))
    emb._client.fetch_json_table_batch = AsyncMock(
        return_value=[array.array("f", [0.4, 0.5, 0.6]), array.array("f", [0.7, 0.8, 0.9])]
    )

    await emb.initialize()
    assert emb.dimension == 3

    out1 = await emb.encode(["hello"])
    assert len(out1) == 1 and pytest.approx(out1[0]) == [0.1, 0.2, 0.3]

    out3 = await emb.encode(["a", "b", "c"])
    assert len(out3) == 3
    assert pytest.approx(out3[0]) == [0.4, 0.5, 0.6]
    assert pytest.approx(out3[2]) == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_oracle_indb_cross_encoder_batch_and_single():
    ce = OracleInDbCrossEncoder(
        database_url="oracle://u:p@localhost:1521/FREEPDB1",
        model_name="HS_RERANK_1",
        batch_size=2,
        max_chars=10,
    )
    ce._client = AsyncMock()
    ce._client.verify_mining_model = AsyncMock(return_value=None)
    ce._client.fetch_single = AsyncMock(return_value=1.25)
    ce._client.fetch_json_table_batch = AsyncMock(return_value=[2.5, -3.5])

    await ce.initialize()
    scores = await ce.predict([("query", "doc1 long text"), ("query", "doc2"), ("query", "doc3")])
    assert scores == [2.5, -3.5, 1.25]
    # Verify max_chars truncation applied to JSON_TABLE payload
    sent_items = ce._client.fetch_json_table_batch.call_args.args[1]
    assert sent_items[0]["d"] == "doc1 long "


def test_array_embedding_conversion():
    arr = array.array("f", [0.25, -0.5, 0.75])
    np_vec = np.asarray(arr, dtype=np.float32)
    assert np_vec.dtype == np.float32
    assert pytest.approx(np_vec.tolist()) == [0.25, -0.5, 0.75]
