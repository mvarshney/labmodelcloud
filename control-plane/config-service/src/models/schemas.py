"""Pydantic models for Config Service API."""

from pydantic import BaseModel, Field, field_validator
from typing import Dict


class RoutingConfig(BaseModel):
    """
    Routing configuration for traffic splitting.

    Maps model names to routing weights. Weights don't need to sum to 1;
    they represent relative proportions.
    """

    weights: Dict[str, float] = Field(
        ...,
        description="Model name to weight mapping (weights are relative)"
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that weights are positive and sum > 0."""
        if not v:
            raise ValueError("Weights cannot be empty")

        for model_name, weight in v.items():
            if weight < 0:
                raise ValueError(f"Weight for {model_name} must be non-negative")

        total_weight = sum(v.values())
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "weights": {
                    "recommendation_v1": 0.7,
                    "recommendation_v2": 0.3
                }
            }
        }


class RoutingConfigResponse(BaseModel):
    """Response containing current routing configuration."""

    config: RoutingConfig = Field(..., description="Current routing configuration")
    normalized_weights: Dict[str, float] = Field(
        ...,
        description="Normalized weights (sum to 1.0)"
    )
    version: int = Field(..., description="Config version number")

    class Config:
        json_schema_extra = {
            "example": {
                "config": {
                    "weights": {
                        "recommendation_v1": 0.7,
                        "recommendation_v2": 0.3
                    }
                },
                "normalized_weights": {
                    "recommendation_v1": 0.7,
                    "recommendation_v2": 0.3
                },
                "version": 5
            }
        }
