import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve
from ray.serve.handle import DeploymentHandle

from src.errors import ModelNotFoundError
from src.metrics.collectors import inference_request_latency_seconds
from src.models.registry import ModelRegistry

logger = logging.getLogger("ray.serve")

app = FastAPI(title="Model Multiplexing Gateway", version="1.0.0")


class PredictRequest(BaseModel):
    input: list[list[float]]


class PredictResponse(BaseModel):
    model_id: str
    shape: list[int]
    predictions: list[list[float]]


@serve.deployment(name="Ingress")
@serve.ingress(app)
class Ingress:
    """FastAPI ingress that validates model IDs and routes to workers."""

    def __init__(self, worker_handle: DeploymentHandle):
        self._worker = worker_handle
        self._registry = ModelRegistry()
        logger.info("Ingress initialized with %d registered models", len(self._registry))

    @app.get("/v1/models")
    async def list_models(self) -> list[dict]:
        """List all available models."""
        return self._registry.list_models()

    @app.get("/v1/models/{model_id}")
    async def get_model_info(self, model_id: str) -> dict:
        """Get metadata for a specific model."""
        try:
            return self._registry.validate(model_id)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/v1/predict/{model_id}", response_model=PredictResponse)
    async def predict(self, model_id: str, body: PredictRequest) -> PredictResponse:
        """Run inference on the specified model."""
        # Validate model_id
        try:
            self._registry.validate(model_id)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        start = time.perf_counter()

        # Route to worker with multiplexed_model_id
        result = await self._worker.options(
            multiplexed_model_id=model_id,
        ).predict.remote(body.input)

        elapsed = time.perf_counter() - start
        inference_request_latency_seconds.observe(elapsed, tags={"model_id": model_id})

        logger.info("Inference %s completed in %.3fs", model_id, elapsed)
        return PredictResponse(**result)

    @app.get("/health")
    async def health(self) -> dict:
        return {"status": "healthy"}
