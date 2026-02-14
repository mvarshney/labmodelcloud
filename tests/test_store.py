import pytest
import asyncio
from unittest.mock import patch

from src.errors import ModelNotFoundError
from src.models.store import ModelStore


@pytest.mark.asyncio
async def test_load_model(tmp_model_store):
    store = ModelStore(store_path=tmp_model_store)
    with patch("src.models.store.SIMULATED_DOWNLOAD_LATENCY", 0.0):
        model = await store.load_model("test-model-v0")
    assert model.model_id == "test-model-v0"
    assert model.weights.shape == (64, 64)


@pytest.mark.asyncio
async def test_load_missing_model(tmp_model_store):
    store = ModelStore(store_path=tmp_model_store)
    with pytest.raises(ModelNotFoundError, match="nonexistent"):
        await store.load_model("nonexistent")


@pytest.mark.asyncio
async def test_predict_after_load(tmp_model_store):
    store = ModelStore(store_path=tmp_model_store)
    with patch("src.models.store.SIMULATED_DOWNLOAD_LATENCY", 0.0):
        model = await store.load_model("test-model-v0")
    # Input with matching dimensions
    result = model.predict([[1.0] * 64])
    assert len(result) == 1
    assert len(result[0]) == 64


@pytest.mark.asyncio
async def test_predict_with_padding(tmp_model_store):
    store = ModelStore(store_path=tmp_model_store)
    with patch("src.models.store.SIMULATED_DOWNLOAD_LATENCY", 0.0):
        model = await store.load_model("test-model-v0")
    # Input shorter than weight matrix — should be padded
    result = model.predict([[1.0, 2.0, 3.0]])
    assert len(result) == 1
    assert len(result[0]) == 64
