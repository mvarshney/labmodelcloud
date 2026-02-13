"""Pydantic models for Gateway API."""

from pydantic import BaseModel, Field
from typing import List


class PredictRequest(BaseModel):
    """
    Inference request.

    Clients send user_id and item_id pairs to get recommendation scores.
    """

    user_ids: List[int] = Field(..., description="List of user IDs")
    item_ids: List[int] = Field(..., description="List of item IDs")

    class Config:
        json_schema_extra = {
            "example": {
                "user_ids": [123, 456, 789],
                "item_ids": [1001, 2002, 3003]
            }
        }


class PredictResponse(BaseModel):
    """
    Inference response.

    Contains recommendation scores for each (user, item) pair.
    """

    scores: List[float] = Field(..., description="Recommendation scores (0-1)")
    model_name: str = Field(..., description="Model that served this request")
    batch_size: int = Field(..., description="Batch size used in Triton")

    class Config:
        json_schema_extra = {
            "example": {
                "scores": [0.87, 0.54, 0.92],
                "model_name": "recommendation_v1",
                "batch_size": 3
            }
        }
