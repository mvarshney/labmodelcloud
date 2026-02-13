"""Model deployment orchestration service."""

import boto3
from botocore.exceptions import ClientError
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import uuid
import httpx

from ..models.schemas import DeploymentInfo, DeploymentStatus
from .triton_client import TritonManagementClient


class ModelDeployer:
    """
    Orchestrates model deployment to Triton.

    Steps:
    1. Download model from MinIO/S3
    2. Copy to Triton model repository
    3. Load model in Triton
    4. Run smoke test
    5. Update routing configuration
    """

    def __init__(
        self,
        s3_endpoint: str,
        s3_access_key: str,
        s3_secret_key: str,
        s3_bucket: str,
        triton_url: str,
        triton_model_repo: str,
        model_registry_url: str,
        config_service_url: str,
    ):
        """
        Initialize deployer.

        Args:
            s3_endpoint: MinIO/S3 endpoint
            s3_access_key: S3 access key
            s3_secret_key: S3 secret key
            s3_bucket: S3 bucket name
            triton_url: Triton server URL
            triton_model_repo: Path to Triton model repository
            model_registry_url: Model registry service URL
            config_service_url: Config service URL
        """
        # S3 client for MinIO
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
        )
        self.s3_bucket = s3_bucket

        # Triton client
        self.triton_client = TritonManagementClient(triton_url)

        # Paths
        self.triton_model_repo = Path(triton_model_repo)

        # Service URLs
        self.model_registry_url = model_registry_url
        self.config_service_url = config_service_url

        # In-memory deployment tracking
        self._deployments: dict[str, DeploymentInfo] = {}

    async def get_model_info(self, model_id: str) -> Optional[dict]:
        """Fetch model info from registry."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.model_registry_url}/api/v1/models/{model_id}"
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Failed to fetch model info: {response.status_code}")
                    return None
        except Exception as e:
            print(f"Error fetching model info: {e}")
            return None

    async def update_model_status(self, model_id: str, status: str):
        """Update model status in registry."""
        try:
            async with httpx.AsyncClient() as client:
                await client.put(
                    f"{self.model_registry_url}/api/v1/models/{model_id}/status",
                    params={"status": status}
                )
        except Exception as e:
            print(f"Warning: Could not update model status: {e}")

    def download_from_s3(self, s3_path: str, local_path: Path) -> bool:
        """
        Download model from S3/MinIO.

        Args:
            s3_path: S3 path (s3://bucket/key or just key)
            local_path: Local file path

        Returns:
            True if successful
        """
        try:
            # Parse S3 path
            if s3_path.startswith("s3://"):
                parts = s3_path[5:].split("/", 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ""
            else:
                bucket = self.s3_bucket
                key = s3_path

            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            print(f"Downloading from S3: {bucket}/{key}")
            self.s3_client.download_file(bucket, key, str(local_path))
            print(f"✓ Downloaded to {local_path}")
            return True

        except ClientError as e:
            print(f"✗ S3 download failed: {e}")
            return False

    def copy_to_triton_repo(
        self,
        source_path: Path,
        triton_model_name: str,
        version: str = "1"
    ) -> bool:
        """
        Copy model to Triton model repository.

        Args:
            source_path: Source model file
            triton_model_name: Model name in Triton
            version: Model version

        Returns:
            True if successful
        """
        try:
            # Create model directory structure
            model_dir = self.triton_model_repo / triton_model_name / version
            model_dir.mkdir(parents=True, exist_ok=True)

            # Copy model file
            dest_path = model_dir / "model.pt"
            shutil.copy2(source_path, dest_path)
            print(f"✓ Copied model to {dest_path}")

            # Check if config.pbtxt exists, if not warn
            config_path = self.triton_model_repo / triton_model_name / "config.pbtxt"
            if not config_path.exists():
                print(f"⚠ Warning: config.pbtxt not found at {config_path}")
                print(f"  Model may not load correctly without config")

            return True

        except Exception as e:
            print(f"✗ Failed to copy model to Triton repo: {e}")
            return False

    async def deploy_model(
        self,
        model_id: str,
        triton_model_name: str,
        force: bool = False
    ) -> DeploymentInfo:
        """
        Deploy a model to Triton.

        Args:
            model_id: Model ID from registry
            triton_model_name: Name to use in Triton
            force: Force redeployment if already loaded

        Returns:
            DeploymentInfo with deployment status
        """
        # Create deployment record
        deployment_id = f"deploy_{uuid.uuid4().hex[:8]}"
        deployment = DeploymentInfo(
            deployment_id=deployment_id,
            model_id=model_id,
            triton_model_name=triton_model_name,
            status=DeploymentStatus.PENDING,
            started_at=datetime.utcnow(),
            metadata={}
        )
        self._deployments[deployment_id] = deployment

        try:
            # Step 1: Get model info from registry
            print(f"\n[Deployment {deployment_id}] Starting deployment...")
            model_info = await self.get_model_info(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found in registry")

            s3_path = model_info["s3_path"]
            print(f"Model S3 path: {s3_path}")

            # Step 2: Check if already loaded
            if not force and self.triton_client.is_model_ready(triton_model_name):
                print(f"⚠ Model {triton_model_name} already loaded (use force=true to redeploy)")
                deployment.status = DeploymentStatus.COMPLETED
                deployment.completed_at = datetime.utcnow()
                deployment.metadata["already_loaded"] = True
                return deployment

            # Step 3: Download from S3
            deployment.status = DeploymentStatus.DOWNLOADING
            await self.update_model_status(model_id, "deploying")

            temp_path = Path("/tmp") / f"{model_id}.pt"
            if not self.download_from_s3(s3_path, temp_path):
                raise Exception("Failed to download model from S3")

            # Step 4: Copy to Triton repository
            if not self.copy_to_triton_repo(temp_path, triton_model_name):
                raise Exception("Failed to copy model to Triton repository")

            # Clean up temp file
            temp_path.unlink(missing_ok=True)

            # Step 5: Load in Triton
            deployment.status = DeploymentStatus.LOADING
            if not self.triton_client.load_model(triton_model_name):
                raise Exception("Failed to load model in Triton")

            # Step 6: Run smoke test
            deployment.status = DeploymentStatus.TESTING
            smoke_test_passed = self.triton_client.smoke_test(triton_model_name)
            deployment.metadata["smoke_test_passed"] = smoke_test_passed

            if not smoke_test_passed:
                raise Exception("Smoke test failed")

            # Step 7: Mark as completed
            deployment.status = DeploymentStatus.COMPLETED
            deployment.completed_at = datetime.utcnow()
            await self.update_model_status(model_id, "deployed")

            print(f"✓ Deployment {deployment_id} completed successfully")
            return deployment

        except Exception as e:
            # Mark as failed
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            deployment.completed_at = datetime.utcnow()
            await self.update_model_status(model_id, "failed")

            print(f"✗ Deployment {deployment_id} failed: {e}")
            return deployment

    def unload_model(self, triton_model_name: str) -> bool:
        """
        Unload a model from Triton.

        Args:
            triton_model_name: Model name in Triton

        Returns:
            True if successful
        """
        return self.triton_client.unload_model(triton_model_name)

    def get_deployment(self, deployment_id: str) -> Optional[DeploymentInfo]:
        """Get deployment info by ID."""
        return self._deployments.get(deployment_id)

    def list_triton_models(self) -> list[dict]:
        """List all models currently in Triton."""
        return self.triton_client.list_models()
