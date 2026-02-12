# Deployer Service Implementation

from typing import List
import requests

class Deployer:
    def __init__(self, triton_url: str):
        self.triton_url = triton_url

    def deploy_model(self, model_name: str, model_version: str, model_path: str) -> bool:
        """Deploy a model to Triton Inference Server."""
        payload = {
            "model_name": model_name,
            "model_version": model_version,
            "model_path": model_path
        }
        response = requests.post(f"{self.triton_url}/v2/repository/models/{model_name}/versions/{model_version}", json=payload)
        return response.status_code == 200

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from Triton Inference Server."""
        response = requests.delete(f"{self.triton_url}/v2/repository/models/{model_name}")
        return response.status_code == 200

    def list_models(self) -> List[str]:
        """List all models in Triton Inference Server."""
        response = requests.get(f"{self.triton_url}/v2/repository/models")
        if response.status_code == 200:
            return response.json().get("models", [])
        return []