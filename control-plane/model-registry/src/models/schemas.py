"""Pydantic models for Model Registry API."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ModelFramework(str, Enum):
    """Supported model frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"


class ModelStatus(str, Enum):
    """Model registration status."""
    REGISTERED = "registered"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"


class ModelRegisterRequest(BaseModel):
    """Request to register a new model."""

    name: str = Field(..., description="Model name (e.g., 'recommendation_v1')")
    version: str = Field(..., description="Model version (e.g., '1', 'v2.0')")
    framework: ModelFramework = Field(..., description="Model framework")
    s3_path: str = Field(..., description="S3/MinIO path to model artifact")
    description: Optional[str] = Field(None, description="Model description")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., metrics, training info)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "recommendation_v1",
                "version": "1",
                "framework": "pytorch",
                "s3_path": "s3://models/recommendation_v1/1/model.pt",
                "description": "Initial recommendation model",
                "metadata": {
                    "accuracy": 0.87,
                    "training_date": "2024-01-15",
                    "dataset": "user_interactions_v1"
                }
            }
        }


class ModelInfo(BaseModel):
    """Model information returned by the registry."""

    model_id: str = Field(..., description="Unique model ID")
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    framework: ModelFramework = Field(..., description="Model framework")
    s3_path: str = Field(..., description="S3/MinIO path to model artifact")
    status: ModelStatus = Field(..., description="Current status")
    description: Optional[str] = Field(None, description="Model description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    registered_at: datetime = Field(..., description="Registration timestamp")
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "rec_v1_1",
                "name": "recommendation_v1",
                "version": "1",
                "framework": "pytorch",
                "s3_path": "s3://models/recommendation_v1/1/model.pt",
                "status": "registered",
                "description": "Initial recommendation model",
                "metadata": {"accuracy": 0.87},
                "registered_at": "2024-01-15T10:30:00Z",
                "mlflow_run_id": "abc123"
            }
        }


class ModelListResponse(BaseModel):
    """Response containing list of models."""

    models: list[ModelInfo] = Field(..., description="List of registered models")
    total: int = Field(..., description="Total number of models")
