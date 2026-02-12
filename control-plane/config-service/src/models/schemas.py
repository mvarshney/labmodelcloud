from pydantic import BaseModel
from typing import List, Optional

class RoutingConfig(BaseModel):
    model_name: str
    weight: float

class ConfigResponse(BaseModel):
    routing: List[RoutingConfig]

class UpdateConfigRequest(BaseModel):
    routing: List[RoutingConfig]