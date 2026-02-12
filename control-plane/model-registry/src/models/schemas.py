from pydantic import BaseModel
from typing import List, Optional

class ModelMetadata(BaseModel):
    name: str
    version: str
    description: Optional[str] = None
    tags: List[str] = []

class ModelRegistrationRequest(BaseModel):
    metadata: ModelMetadata

class ModelRegistrationResponse(BaseModel):
    success: bool
    message: str
    model_id: Optional[str] = None

class ModelUpdateRequest(BaseModel):
    model_id: str
    metadata: ModelMetadata

class ModelUpdateResponse(BaseModel):
    success: bool
    message: str

class ModelDiscoveryResponse(BaseModel):
    models: List[ModelMetadata]