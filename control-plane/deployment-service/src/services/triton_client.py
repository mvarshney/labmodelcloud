"""Triton client for model management operations."""

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import numpy as np
from typing import List, Optional
import time


class TritonManagementClient:
    """
    Client for Triton model management operations.

    Handles loading, unloading, and querying models via Triton's HTTP API.
    """

    def __init__(self, triton_url: str = "triton:8000"):
        """
        Initialize Triton client.

        Args:
            triton_url: Triton server URL (host:port)
        """
        self.triton_url = triton_url
        self.client = httpclient.InferenceServerClient(url=triton_url, verbose=False)

    def is_server_ready(self) -> bool:
        """Check if Triton server is ready."""
        try:
            return self.client.is_server_ready()
        except Exception as e:
            print(f"Triton server not ready: {e}")
            return False

    def is_server_live(self) -> bool:
        """Check if Triton server is live."""
        try:
            return self.client.is_server_live()
        except Exception as e:
            print(f"Triton server not live: {e}")
            return False

    def list_models(self) -> List[dict]:
        """
        List all models in Triton.

        Returns:
            List of model info dictionaries
        """
        try:
            model_repository = self.client.get_model_repository_index()
            return [
                {
                    "name": model.name,
                    "version": model.version,
                    "state": model.state,
                    "ready": model.state == "READY"
                }
                for model in model_repository.models
            ]
        except InferenceServerException as e:
            print(f"Error listing models: {e}")
            return []

    def is_model_ready(self, model_name: str, model_version: str = "1") -> bool:
        """
        Check if a specific model is ready.

        Args:
            model_name: Model name
            model_version: Model version

        Returns:
            True if model is ready, False otherwise
        """
        try:
            return self.client.is_model_ready(model_name, model_version)
        except Exception as e:
            print(f"Error checking model readiness: {e}")
            return False

    def load_model(self, model_name: str) -> bool:
        """
        Load a model into Triton.

        Args:
            model_name: Model name to load

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.load_model(model_name)
            print(f"✓ Load request sent for model: {model_name}")

            # Wait for model to be ready (max 30 seconds)
            for i in range(30):
                if self.is_model_ready(model_name):
                    print(f"✓ Model {model_name} is ready")
                    return True
                time.sleep(1)

            print(f"⚠ Model {model_name} did not become ready in 30 seconds")
            return False

        except InferenceServerException as e:
            print(f"✗ Failed to load model {model_name}: {e}")
            return False

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from Triton.

        Args:
            model_name: Model name to unload

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.unload_model(model_name)
            print(f"✓ Unload request sent for model: {model_name}")

            # Wait for model to be unloaded (max 10 seconds)
            for i in range(10):
                if not self.is_model_ready(model_name):
                    print(f"✓ Model {model_name} is unloaded")
                    return True
                time.sleep(1)

            print(f"⚠ Model {model_name} did not unload in 10 seconds")
            return False

        except InferenceServerException as e:
            print(f"✗ Failed to unload model {model_name}: {e}")
            return False

    def get_model_metadata(self, model_name: str, model_version: str = "1") -> Optional[dict]:
        """
        Get model metadata.

        Args:
            model_name: Model name
            model_version: Model version

        Returns:
            Model metadata dictionary or None
        """
        try:
            metadata = self.client.get_model_metadata(model_name, model_version)
            return {
                "name": metadata.name,
                "versions": metadata.versions,
                "platform": metadata.platform,
                "inputs": [
                    {
                        "name": inp.name,
                        "datatype": inp.datatype,
                        "shape": inp.shape
                    }
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "datatype": out.datatype,
                        "shape": out.shape
                    }
                    for out in metadata.outputs
                ]
            }
        except InferenceServerException as e:
            print(f"Error getting metadata for {model_name}: {e}")
            return None

    def smoke_test(self, model_name: str, model_version: str = "1") -> bool:
        """
        Run a simple smoke test on the model.

        Sends a test inference request to verify the model is working.

        Args:
            model_name: Model name
            model_version: Model version

        Returns:
            True if test passed, False otherwise
        """
        try:
            # Create test inputs (assuming recommendation model)
            user_ids = np.array([1, 2, 3, 4], dtype=np.int64)
            item_ids = np.array([100, 200, 300, 400], dtype=np.int64)

            # Create input objects
            inputs = [
                httpclient.InferInput("user_ids", user_ids.shape, "INT64"),
                httpclient.InferInput("item_ids", item_ids.shape, "INT64"),
            ]

            inputs[0].set_data_from_numpy(user_ids)
            inputs[1].set_data_from_numpy(item_ids)

            # Create output request
            outputs = [httpclient.InferRequestedOutput("scores")]

            # Send inference request
            result = self.client.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                outputs=outputs
            )

            # Get output
            scores = result.as_numpy("scores")

            # Basic validation
            assert scores.shape[0] == 4, "Unexpected output shape"
            assert scores.shape[1] == 1, "Unexpected output shape"
            assert all(0 <= s <= 1 for s in scores), "Scores should be between 0 and 1"

            print(f"✓ Smoke test passed for {model_name}")
            print(f"  Sample scores: {scores[:3].squeeze().tolist()}")
            return True

        except Exception as e:
            print(f"✗ Smoke test failed for {model_name}: {e}")
            return False
