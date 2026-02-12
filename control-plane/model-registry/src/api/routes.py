from fastapi import APIRouter, HTTPException
from typing import List
from ..models.schemas import ModelMetadata
from ..services.registry import ModelRegistry

router = APIRouter()
registry = ModelRegistry()

@router.post("/models/", response_model=ModelMetadata)
async def register_model(model: ModelMetadata):
    try:
        return await registry.register_model(model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models/", response_model=List[ModelMetadata])
async def list_models():
    return await registry.list_models()

@router.get("/models/{model_id}", response_model=ModelMetadata)
async def get_model(model_id: str):
    model = await registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.delete("/models/{model_id}", response_model=dict)
async def delete_model(model_id: str):
    success = await registry.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"detail": "Model deleted successfully"}