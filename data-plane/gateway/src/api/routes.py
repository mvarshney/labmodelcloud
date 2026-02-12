from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class PredictRequest(BaseModel):
    tensor: list

class PredictResponse(BaseModel):
    score: float

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # Placeholder for prediction logic
    return PredictResponse(score=0.0)  # Replace with actual prediction logic

@router.get("/health")
async def health_check():
    return {"status": "healthy"}