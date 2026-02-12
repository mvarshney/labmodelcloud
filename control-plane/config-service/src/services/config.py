from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

class Config(BaseModel):
    model_weights: Dict[str, float]

class RoutingConfig(BaseModel):
    weights: Dict[str, float]

@app.get("/config", response_model=RoutingConfig)
async def get_config() -> RoutingConfig:
    # Logic to retrieve the current routing configuration
    return RoutingConfig(weights={"recommendation_v1": 0.5, "recommendation_v2": 0.5})

@app.post("/config", response_model=RoutingConfig)
async def update_config(config: RoutingConfig) -> RoutingConfig:
    # Logic to update the routing configuration
    return config