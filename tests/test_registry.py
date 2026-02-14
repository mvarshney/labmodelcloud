import pytest

from src.errors import ModelNotFoundError
from src.models.registry import ModelRegistry


def test_load_manifest(tmp_model_store):
    registry = ModelRegistry(manifest_path=tmp_model_store / "manifest.json")
    assert len(registry) == 3


def test_validate_existing_model(tmp_model_store):
    registry = ModelRegistry(manifest_path=tmp_model_store / "manifest.json")
    info = registry.validate("test-model-v0")
    assert info["model_id"] == "test-model-v0"
    assert info["family"] == "test-model"


def test_validate_missing_model(tmp_model_store):
    registry = ModelRegistry(manifest_path=tmp_model_store / "manifest.json")
    with pytest.raises(ModelNotFoundError, match="nonexistent"):
        registry.validate("nonexistent")


def test_contains(tmp_model_store):
    registry = ModelRegistry(manifest_path=tmp_model_store / "manifest.json")
    assert "test-model-v0" in registry
    assert "missing" not in registry


def test_list_models(tmp_model_store):
    registry = ModelRegistry(manifest_path=tmp_model_store / "manifest.json")
    models = registry.list_models()
    assert len(models) == 3
    assert all("model_id" in m for m in models)


def test_missing_manifest(tmp_path):
    registry = ModelRegistry(manifest_path=tmp_path / "nonexistent.json")
    assert len(registry) == 0
