"""API routes for Gateway service."""

from fastapi import APIRouter, HTTPException, Depends, Response
from typing import Optional

from ..models.schemas import PredictRequest, PredictResponse
from ..services.triton_client import TritonInferenceClient
from ..services.router import WeightedRouter
from ..metrics.prometheus import MetricsCollector, get_metrics, get_content_type

router = APIRouter()

# Global instances (initialized in main.py)
_triton_client: Optional[TritonInferenceClient] = None
_router: Optional[WeightedRouter] = None
_metrics: Optional[MetricsCollector] = None


def get_triton_client() -> TritonInferenceClient:
    """Dependency to get Triton client."""
    if _triton_client is None:
        raise RuntimeError("Triton client not initialized")
    return _triton_client


def get_router() -> WeightedRouter:
    """Dependency to get router."""
    if _router is None:
        raise RuntimeError("Router not initialized")
    return _router


def get_metrics_collector() -> MetricsCollector:
    """Dependency to get metrics collector."""
    if _metrics is None:
        raise RuntimeError("Metrics collector not initialized")
    return _metrics


def init_gateway(
    triton_client: TritonInferenceClient,
    router: WeightedRouter,
    metrics: MetricsCollector
):
    """Initialize global gateway components."""
    global _triton_client, _router, _metrics
    _triton_client = triton_client
    _router = router
    _metrics = metrics


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    triton_client: TritonInferenceClient = Depends(get_triton_client),
    router: WeightedRouter = Depends(get_router),
    metrics: MetricsCollector = Depends(get_metrics_collector),
) -> PredictResponse:
    """
    Inference endpoint.

    This is the main endpoint that demonstrates:
    1. Weighted routing for traffic splitting
    2. Triton gRPC communication
    3. Dynamic batching (transparent to client)
    4. Prometheus metrics collection

    The request contains N (user, item) pairs and returns N scores.
    """
    try:
        # Validate input
        if len(request.user_ids) != len(request.item_ids):
            raise HTTPException(
                status_code=400,
                detail="user_ids and item_ids must have the same length"
            )

        if len(request.user_ids) == 0:
            raise HTTPException(
                status_code=400,
                detail="user_ids and item_ids cannot be empty"
            )

        # Step 1: Select model based on routing weights
        model_name = router.select_model()
        if model_name is None:
            raise HTTPException(
                status_code=503,
                detail="No models available for routing"
            )

        # Step 2: Send inference request to Triton
        # This is where dynamic batching happens (on Triton side)!
        with metrics.measure_latency(model_name):
            scores, batch_size = await triton_client.infer(
                model_name=model_name,
                user_ids=request.user_ids,
                item_ids=request.item_ids
            )

        # Step 3: Record metrics
        metrics.record_success(model_name, batch_size)

        # Step 4: Return response
        return PredictResponse(
            scores=scores,
            model_name=model_name,
            batch_size=batch_size
        )

    except HTTPException:
        raise
    except Exception as e:
        # Record error metric
        model_name = router.select_model() or "unknown"
        metrics.record_error(model_name)

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )


@router.get("/routing/weights")
async def get_routing_weights(
    router: WeightedRouter = Depends(get_router),
) -> dict:
    """
    Get current routing weights.

    Useful for debugging and verification.
    """
    return {
        "weights": router.get_weights()
    }


@router.get("/metrics")
async def metrics_endpoint(
    metrics: MetricsCollector = Depends(get_metrics_collector),
) -> Response:
    """
    Prometheus metrics endpoint.

    This endpoint is scraped by Prometheus to collect metrics.
    """
    return Response(
        content=get_metrics(),
        media_type=get_content_type()
    )
