"""
End-to-end model deployment script.

This script:
1. Registers a model in the Model Registry
2. Deploys it to Triton via the Deployment Service
3. Updates routing configuration

Usage:
    python deploy_model.py --model-name recommendation_v1 --version 1
"""

import httpx
import time
import sys
from typing import Optional


class ModelDeploymentClient:
    """Client for end-to-end model deployment."""

    def __init__(
        self,
        registry_url: str = "http://localhost:8001",
        deployment_url: str = "http://localhost:8003",
        config_url: str = "http://localhost:8002",
    ):
        """Initialize deployment client."""
        self.registry_url = registry_url
        self.deployment_url = deployment_url
        self.config_url = config_url

    def register_model(
        self,
        name: str,
        version: str,
        s3_path: str,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Register model in the registry.

        Args:
            name: Model name (e.g., 'recommendation_v1')
            version: Model version (e.g., '1')
            s3_path: S3 path to model artifact
            description: Model description
            metadata: Additional metadata

        Returns:
            Model registration response
        """
        print(f"\n[1/4] Registering model {name} v{version}...")

        payload = {
            "name": name,
            "version": version,
            "framework": "pytorch",
            "s3_path": s3_path,
            "description": description or f"Model {name} version {version}",
            "metadata": metadata or {},
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    f"{self.registry_url}/api/v1/models/register",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                print(f"✓ Model registered successfully")
                print(f"  Model ID: {result['model_id']}")
                return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                print(f"⚠ Model already registered, continuing...")
                # Get existing model info
                model_id = f"{name}_{version}".replace(".", "_")
                response = client.get(
                    f"{self.registry_url}/api/v1/models/{model_id}"
                )
                return response.json()
            else:
                print(f"✗ Registration failed: {e}")
                raise
        except Exception as e:
            print(f"✗ Registration failed: {e}")
            raise

    def deploy_model(self, model_id: str, triton_model_name: str) -> dict:
        """
        Deploy model to Triton.

        Args:
            model_id: Model ID from registry
            triton_model_name: Name to use in Triton

        Returns:
            Deployment response
        """
        print(f"\n[2/4] Deploying model to Triton...")

        payload = {
            "model_id": model_id,
            "triton_model_name": triton_model_name,
            "force": False,
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.deployment_url}/api/v1/deploy",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                deployment_id = result["deployment_id"]
                print(f"✓ Deployment started")
                print(f"  Deployment ID: {deployment_id}")

                # Wait for deployment to complete
                self._wait_for_deployment(deployment_id)

                return result

        except Exception as e:
            print(f"✗ Deployment failed: {e}")
            raise

    def _wait_for_deployment(self, deployment_id: str, max_wait: int = 120):
        """Wait for deployment to complete."""
        print(f"  Waiting for deployment to complete...")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(
                        f"{self.deployment_url}/api/v1/deployments/{deployment_id}"
                    )
                    response.raise_for_status()
                    result = response.json()

                    status = result["status"]
                    print(f"  Status: {status}")

                    if status == "completed":
                        print(f"✓ Deployment completed successfully")
                        return
                    elif status == "failed":
                        error = result.get("error_message", "Unknown error")
                        print(f"✗ Deployment failed: {error}")
                        raise Exception(f"Deployment failed: {error}")

                    time.sleep(5)

            except httpx.HTTPError as e:
                print(f"  Error checking status: {e}")
                time.sleep(5)

        print(f"✗ Deployment timed out after {max_wait}s")
        raise Exception("Deployment timeout")

    def update_routing_config(self, weights: dict) -> dict:
        """
        Update routing configuration.

        Args:
            weights: Model name to weight mapping

        Returns:
            Config update response
        """
        print(f"\n[3/4] Updating routing configuration...")
        print(f"  Weights: {weights}")

        payload = {"weights": weights}

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.put(
                    f"{self.config_url}/api/v1/config/routing",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                print(f"✓ Routing config updated")
                print(f"  Version: {result['version']}")
                print(f"  Normalized weights: {result['normalized_weights']}")
                return result

        except Exception as e:
            print(f"✗ Config update failed: {e}")
            raise

    def verify_deployment(self, model_name: str):
        """Verify model is loaded in Triton."""
        print(f"\n[4/4] Verifying deployment...")

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.deployment_url}/api/v1/models"
                )
                response.raise_for_status()
                result = response.json()

                models = result["models"]
                model_found = any(
                    m["name"] == model_name and m["ready"]
                    for m in models
                )

                if model_found:
                    print(f"✓ Model {model_name} is loaded and ready")
                else:
                    print(f"⚠ Model {model_name} not found or not ready")

        except Exception as e:
            print(f"⚠ Verification failed: {e}")


def main():
    """Main deployment workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deploy a model end-to-end"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name (e.g., recommendation_v1)"
    )
    parser.add_argument(
        "--version",
        default="1",
        help="Model version"
    )
    parser.add_argument(
        "--s3-bucket",
        default="models",
        help="S3 bucket name"
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Routing weight for this model"
    )
    parser.add_argument(
        "--registry-url",
        default="http://localhost:8001",
        help="Model Registry URL"
    )
    parser.add_argument(
        "--deployment-url",
        default="http://localhost:8003",
        help="Deployment Service URL"
    )
    parser.add_argument(
        "--config-url",
        default="http://localhost:8002",
        help="Config Service URL"
    )

    args = parser.parse_args()

    # Initialize client
    client = ModelDeploymentClient(
        registry_url=args.registry_url,
        deployment_url=args.deployment_url,
        config_url=args.config_url,
    )

    # Construct S3 path
    s3_path = f"{args.s3_bucket}/{args.model_name}/{args.version}/model.pt"

    print("=" * 60)
    print(f"Deploying Model: {args.model_name} v{args.version}")
    print("=" * 60)

    try:
        # Step 1: Register model
        model_info = client.register_model(
            name=args.model_name,
            version=args.version,
            s3_path=s3_path,
        )
        model_id = model_info["model_id"]

        # Step 2: Deploy to Triton
        client.deploy_model(
            model_id=model_id,
            triton_model_name=args.model_name
        )

        # Step 3: Update routing config
        client.update_routing_config(
            weights={args.model_name: args.weight}
        )

        # Step 4: Verify
        client.verify_deployment(args.model_name)

        print("\n" + "=" * 60)
        print("✓ Deployment Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Send inference requests to http://localhost:8000/api/v1/predict")
        print("2. View metrics at http://localhost:8000/api/v1/metrics")
        print("3. View Grafana dashboard at http://localhost:3000")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Deployment Failed: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
