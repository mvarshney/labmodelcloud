from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import MODEL_MANIFEST_PATH
from src.errors import ModelNotFoundError

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Validates and lists available model IDs from the manifest file."""

    def __init__(self, manifest_path: Path | str | None = None):
        self._manifest_path = Path(manifest_path or MODEL_MANIFEST_PATH)
        self._models: dict[str, dict] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            logger.warning("Manifest not found at %s — registry is empty", self._manifest_path)
            return
        with open(self._manifest_path) as f:
            data = json.load(f)
        self._models = {m["model_id"]: m for m in data["models"]}
        logger.info("Loaded %d models from manifest", len(self._models))

    def validate(self, model_id: str) -> dict:
        """Return model metadata if valid, otherwise raise ModelNotFoundError."""
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        return self._models[model_id]

    def list_models(self) -> list[dict]:
        """Return metadata for all registered models."""
        return list(self._models.values())

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models

    def __len__(self) -> int:
        return len(self._models)
