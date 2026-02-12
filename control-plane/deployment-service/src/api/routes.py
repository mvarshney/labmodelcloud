from fastapi import APIRouter, HTTPException
from typing import List
from ..models.schemas import ModelDeploymentRequest, ModelDeploymentResponse
from ..services.deployer import Deployer

router = APIRouter()
deployer = Deployer()

@router.post("/deploy", response_model=ModelDeploymentResponse)
async def deploy_model(request: ModelDeploymentRequest):
    try:
        response = await deployer.deploy_model(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{model_name}", response_model=str)
async def get_model_status(model_name: str):
    try:
        status = await deployer.get_model_status(model_name)
        return status
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))