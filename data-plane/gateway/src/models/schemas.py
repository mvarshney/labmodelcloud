from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    tensor: List[List[float]]  # Shape: [batch_size, features]

class PredictResponse(BaseModel):
    score: float

class ModelConfig(BaseModel):
    model_name: str
    version: str
    weights: Optional[float] = None  # For weighted routing configuration

class RoutingConfig(BaseModel):
    models: List[ModelConfig]