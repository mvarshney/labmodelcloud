# File: /ml-model-serving-platform/ml-model-serving-platform/control-plane/model-registry/src/services/registry.py

from typing import List, Dict
from pydantic import BaseModel
import mlflow

class ModelMetadata(BaseModel):
    name: str
    version: str
    description: str
    tags: Dict[str, str]

class ModelRegistry:
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)

    def register_model(self, model_metadata: ModelMetadata) -> str:
        with mlflow.start_run():
            model_uri = f"models:/{model_metadata.name}/{model_metadata.version}"
            mlflow.log_param("description", model_metadata.description)
            for key, value in model_metadata.tags.items():
                mlflow.set_tag(key, value)
            return model_uri

    def get_model(self, model_name: str, model_version: str) -> ModelMetadata:
        model_info = mlflow.get_model_version(model_name, model_version)
        return ModelMetadata(
            name=model_info.name,
            version=model_info.version,
            description=model_info.description,
            tags=model_info.tags
        )

    def list_models(self) -> List[ModelMetadata]:
        models = mlflow.list_registered_models()
        return [ModelMetadata(
            name=model.name,
            version=model.latest_version,
            description=model.description,
            tags=model.tags
        ) for model in models]