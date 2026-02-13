"""API routes for Deployment Service."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Optional

from ..models.schemas import (
    DeployRequest,
    UnloadRequest,
    DeploymentInfo,
    TritonModelStatus,
    ListModelsResponse,
)
from ..services.deployer import ModelDeployer

router = APIRouter()

# Global deployer instance (initialized in main.py)
_deployer: Optional[ModelDeployer] = None


def get_deployer() -> ModelDeployer:
    """Dependency to get deployer instance."""
    if _deployer is None:
        raise RuntimeError("Deployer not initialized")
    return _deployer


def init_deployer(deployer: ModelDeployer):
    """Initialize the global deployer instance."""
    global _deployer
    _deployer = deployer


@router.post("/deploy", response_model=DeploymentInfo, status_code=202)
async def deploy_model(
    request: DeployRequest,
    deployer: ModelDeployer = Depends(get_deployer),
) -> DeploymentInfo:
    """
    Deploy a model to Triton.

    This endpoint:
    1. Downloads the model from S3/MinIO
    2. Copies it to Triton model repository
    3. Loads it in Triton
    4. Runs a smoke test
    5. Updates the routing configuration

    Returns immediately with deployment info (status: pending).
    Check /deployments/{deployment_id} for status updates.
    """
    try:
        deployment = await deployer.deploy_model(
            model_id=request.model_id,
            triton_model_name=request.triton_model_name,
            force=request.force,
        )
        return deployment
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start deployment: {str(e)}"
        )


@router.get("/deployments/{deployment_id}", response_model=DeploymentInfo)
async def get_deployment(
    deployment_id: str,
    deployer: ModelDeployer = Depends(get_deployer),
) -> DeploymentInfo:
    """
    Get deployment status.

    Use this endpoint to check the status of a deployment.
    """
    deployment = deployer.get_deployment(deployment_id)
    if deployment is None:
        raise HTTPException(
            status_code=404,
            detail=f"Deployment {deployment_id} not found"
        )
    return deployment


@router.post("/unload", status_code=200)
async def unload_model(
    request: UnloadRequest,
    deployer: ModelDeployer = Depends(get_deployer),
) -> dict:
    """
    Unload a model from Triton.

    This removes the model from memory but keeps it in the model repository.
    """
    success = deployer.unload_model(request.triton_model_name)
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model {request.triton_model_name}"
        )

    return {
        "status": "success",
        "message": f"Model {request.triton_model_name} unloaded"
    }


@router.get("/models", response_model=ListModelsResponse)
async def list_models(
    deployer: ModelDeployer = Depends(get_deployer),
) -> ListModelsResponse:
    """
    List all models currently loaded in Triton.
    """
    try:
        models_data = deployer.list_triton_models()
        models = [
            TritonModelStatus(
                name=m["name"],
                version=m["version"],
                state=m["state"],
                ready=m["ready"]
            )
            for m in models_data
        ]
        return ListModelsResponse(models=models, total=len(models))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/triton/health")
async def triton_health(
    deployer: ModelDeployer = Depends(get_deployer),
) -> dict:
    """
    Check Triton server health.
    """
    is_live = deployer.triton_client.is_server_live()
    is_ready = deployer.triton_client.is_server_ready()

    return {
        "triton_live": is_live,
        "triton_ready": is_ready,
        "status": "healthy" if (is_live and is_ready) else "unhealthy"
    }
