from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.config import MODEL_STORE_PATH, SIMULATED_DOWNLOAD_LATENCY
from src.errors import CudaOutOfMemoryError, ModelNotFoundError
from src.models.base import SimulatedModel

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


def _load_weights_sync(weights_path: Path) -> np.ndarray:
    """CPU-bound numpy deserialization — runs in a thread."""
    return np.load(weights_path)


class ModelStore:
    """Async weight fetcher that simulates S3 download latency."""

    def __init__(self, store_path: Path | str | None = None):
        self._store_path = Path(store_path or MODEL_STORE_PATH)

    async def load_model(self, model_id: str) -> SimulatedModel:
        """Load a model's weights from disk with simulated network latency.

        Returns a SimulatedModel ready for inference.
        Raises ModelNotFoundError if the artifact doesn't exist.
        Raises CudaOutOfMemoryError if loading fails due to memory issues.
        """
        model_dir = self._store_path / model_id
        weights_path = model_dir / "weights.npy"

        if not weights_path.exists():
            raise ModelNotFoundError(model_id)

        start = time.perf_counter()

        # Simulate S3/GCS download latency
        await asyncio.sleep(SIMULATED_DOWNLOAD_LATENCY)

        # Deserialize numpy array in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        try:
            weights = await loop.run_in_executor(_executor, _load_weights_sync, weights_path)
        except MemoryError as e:
            raise CudaOutOfMemoryError(model_id, str(e)) from e

        elapsed = time.perf_counter() - start
        model = SimulatedModel(model_id=model_id, weights=weights)
        logger.info(
            "Loaded %s in %.3fs (%.1f MB)",
            model_id, elapsed, model.memory_bytes / 1e6,
        )
        return model
