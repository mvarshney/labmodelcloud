"""Pydantic models for Deployment Service API."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DeploymentStatus(str, Enum):
    """Deployment status."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"


class DeployRequest(BaseModel):
    """Request to deploy a model."""

    model_id: str = Field(..., description="Model ID from registry")
    triton_model_name: str = Field(..., description="Name in Triton (e.g., 'recommendation_v1')")
    force: bool = Field(False, description="Force redeployment if already loaded")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "recommendation_v1_1",
                "triton_model_name": "recommendation_v1",
                "force": False
            }
        }


class UnloadRequest(BaseModel):
    """Request to unload a model from Triton."""

    triton_model_name: str = Field(..., description="Model name in Triton")

    class Config:
        json_schema_extra = {
            "example": {
                "triton_model_name": "recommendation_v1"
            }
        }


class DeploymentInfo(BaseModel):
    """Deployment information."""

    deployment_id: str = Field(..., description="Unique deployment ID")
    model_id: str = Field(..., description="Model ID from registry")
    triton_model_name: str = Field(..., description="Name in Triton")
    status: DeploymentStatus = Field(..., description="Current deployment status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    started_at: datetime = Field(..., description="Deployment start time")
    completed_at: Optional[datetime] = Field(None, description="Deployment completion time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "deployment_id": "deploy_123",
                "model_id": "recommendation_v1_1",
                "triton_model_name": "recommendation_v1",
                "status": "completed",
                "error_message": None,
                "started_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:30:15Z",
                "metadata": {"smoke_test_passed": True}
            }
        }


class TritonModelStatus(BaseModel):
    """Triton model status information."""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    state: str = Field(..., description="Model state (READY, LOADING, etc.)")
    ready: bool = Field(..., description="Whether model is ready for inference")


class ListModelsResponse(BaseModel):
    """Response with list of models in Triton."""

    models: list[TritonModelStatus] = Field(..., description="List of models")
    total: int = Field(..., description="Total number of models")
