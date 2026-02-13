"""Model Registry service with MLflow integration."""

import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from typing import List, Optional
import os

from ..models.schemas import (
    ModelRegisterRequest,
    ModelInfo,
    ModelStatus,
    ModelFramework,
)


class ModelRegistry:
    """
    Model Registry service.

    Manages model metadata and integrates with MLflow for tracking.
    """

    def __init__(self, mlflow_tracking_uri: str):
        """
        Initialize the registry.

        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.client = MlflowClient(tracking_uri=mlflow_tracking_uri)

        # In-memory storage for simplicity (use DB in production)
        self._models: dict[str, ModelInfo] = {}

    def register_model(self, request: ModelRegisterRequest) -> ModelInfo:
        """
        Register a new model.

        Args:
            request: Model registration request

        Returns:
            ModelInfo with registration details
        """
        # Generate unique model ID
        model_id = f"{request.name}_{request.version}".replace(".", "_")

        # Check if already registered
        if model_id in self._models:
            raise ValueError(f"Model {model_id} already registered")

        # Create MLflow experiment if it doesn't exist
        experiment_name = f"model-registry/{request.name}"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"Warning: Could not create MLflow experiment: {e}")
            experiment_id = None

        # Log model registration to MLflow
        mlflow_run_id = None
        try:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                mlflow_run_id = run.info.run_id

                # Log parameters
                mlflow.log_param("model_name", request.name)
                mlflow.log_param("version", request.version)
                mlflow.log_param("framework", request.framework.value)
                mlflow.log_param("s3_path", request.s3_path)

                # Log metadata as tags
                for key, value in request.metadata.items():
                    mlflow.set_tag(f"metadata.{key}", str(value))

                # Log description
                if request.description:
                    mlflow.set_tag("description", request.description)

        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")

        # Create model info
        model_info = ModelInfo(
            model_id=model_id,
            name=request.name,
            version=request.version,
            framework=request.framework,
            s3_path=request.s3_path,
            status=ModelStatus.REGISTERED,
            description=request.description,
            metadata=request.metadata,
            registered_at=datetime.utcnow(),
            mlflow_run_id=mlflow_run_id,
        )

        # Store in registry
        self._models[model_id] = model_info

        print(f"✓ Registered model: {model_id}")
        return model_info

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get model by ID.

        Args:
            model_id: Model ID

        Returns:
            ModelInfo if found, None otherwise
        """
        return self._models.get(model_id)

    def list_models(self, name: Optional[str] = None) -> List[ModelInfo]:
        """
        List all registered models.

        Args:
            name: Optional filter by model name

        Returns:
            List of ModelInfo
        """
        models = list(self._models.values())

        if name:
            models = [m for m in models if m.name == name]

        # Sort by registration time (newest first)
        models.sort(key=lambda m: m.registered_at, reverse=True)

        return models

    def update_model_status(self, model_id: str, status: ModelStatus) -> ModelInfo:
        """
        Update model status.

        Args:
            model_id: Model ID
            status: New status

        Returns:
            Updated ModelInfo

        Raises:
            ValueError: If model not found
        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found")

        self._models[model_id].status = status

        # Log status change to MLflow
        if self._models[model_id].mlflow_run_id:
            try:
                self.client.set_tag(
                    self._models[model_id].mlflow_run_id,
                    "status",
                    status.value
                )
            except Exception as e:
                print(f"Warning: Could not update MLflow: {e}")

        return self._models[model_id]

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_id: Model ID

        Returns:
            True if deleted, False if not found
        """
        if model_id in self._models:
            del self._models[model_id]
            print(f"✓ Deleted model: {model_id}")
            return True
        return False
