from pydantic import BaseModel
from typing import List, Optional

class ModelSchema(BaseModel):
    name: str
    version: str
    description: Optional[str] = None
    input_shape: List[int]
    output_shape: List[int]

class DeploymentSchema(BaseModel):
    model: ModelSchema
    deployment_status: str
    last_deployed: Optional[str] = None
    error_message: Optional[str] = None