from __future__ import annotations

import logging
import time

from ray import serve

from src.config import MAX_MODELS_PER_REPLICA
from src.metrics.collectors import model_cache_miss_total, model_load_duration_seconds
from src.models.base import SimulatedModel
from src.models.store import ModelStore

logger = logging.getLogger("ray.serve")


@serve.deployment(name="MultiplexedModelWorker")
class MultiplexedModelWorker:
    """Inference worker that dynamically loads/evicts model weights.

    Uses Ray Serve's @serve.multiplexed to manage an LRU cache of models.
    When max_num_models_per_replica is exceeded, the least-recently-used
    model is evicted automatically.
    """

    def __init__(self):
        self._store = ModelStore()
        self._healthy = True
        logger.info(
            "Worker initialized (max_models_per_replica=%d)",
            MAX_MODELS_PER_REPLICA,
        )

    @serve.multiplexed(max_num_models_per_replica=MAX_MODELS_PER_REPLICA)
    async def get_model(self, model_id: str) -> SimulatedModel:
        """Load a model from the store. Called on cache miss."""
        logger.info("Cache MISS for %s — loading from store", model_id)
        model_cache_miss_total.inc(tags={"model_id": model_id})

        start = time.perf_counter()
        try:
            model = await self._store.load_model(model_id)
        except Exception:
            self._healthy = False
            logger.exception("Failed to load model %s — marking worker unhealthy", model_id)
            raise
        elapsed = time.perf_counter() - start
        model_load_duration_seconds.observe(elapsed, tags={"model_id": model_id})

        logger.info("Loaded %s in %.3fs", model_id, elapsed)
        return model

    async def predict(self, input_data: list[list[float]]) -> dict:
        """Run inference using the multiplexed model for this request."""
        model_id = serve.get_multiplexed_model_id()
        model: SimulatedModel = await self.get_model(model_id)
        result = model.predict(input_data)
        return {
            "model_id": model.model_id,
            "shape": [len(result), len(result[0]) if result else 0],
            "predictions": result,
        }

    def check_health(self):
        if not self._healthy:
            raise RuntimeError("Worker is unhealthy due to a failed model load")
