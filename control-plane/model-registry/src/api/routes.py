"""API routes for Model Registry service."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from ..models.schemas import (
    ModelRegisterRequest,
    ModelInfo,
    ModelListResponse,
    ModelStatus,
)
from ..services.registry import ModelRegistry

router = APIRouter()

# Global registry instance (will be initialized in main.py)
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Dependency to get the registry instance."""
    if _registry is None:
        raise RuntimeError("Registry not initialized")
    return _registry


def init_registry(registry: ModelRegistry):
    """Initialize the global registry instance."""
    global _registry
    _registry = registry


@router.post("/models/register", response_model=ModelInfo, status_code=201)
async def register_model(
    request: ModelRegisterRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> ModelInfo:
    """
    Register a new model.

    This endpoint registers a model in the registry and logs metadata to MLflow.
    """
    try:
        return registry.register_model(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    name: Optional[str] = None,
    registry: ModelRegistry = Depends(get_registry),
) -> ModelListResponse:
    """
    List all registered models.

    Optionally filter by model name.
    """
    try:
        models = registry.list_models(name=name)
        return ModelListResponse(models=models, total=len(models))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_registry),
) -> ModelInfo:
    """
    Get details of a specific model.
    """
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model


@router.put("/models/{model_id}/status", response_model=ModelInfo)
async def update_model_status(
    model_id: str,
    status: ModelStatus,
    registry: ModelRegistry = Depends(get_registry),
) -> ModelInfo:
    """
    Update model status.

    Used by deployment service to update status during deployment.
    """
    try:
        return registry.update_model_status(model_id, status)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update status: {str(e)}")


@router.delete("/models/{model_id}", status_code=204)
async def delete_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_registry),
):
    """
    Delete a model from the registry.
    """
    success = registry.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
