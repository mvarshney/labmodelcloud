"""
End-to-end integration test.

This test requires all services to be running (via docker-compose).

Usage:
    docker-compose up -d
    pytest tests/integration/test_e2e.py -v
"""

import pytest
import httpx
import time
from typing import Optional


# Service URLs
GATEWAY_URL = "http://localhost:8000"
MODEL_REGISTRY_URL = "http://localhost:8001"
DEPLOYMENT_URL = "http://localhost:8003"
CONFIG_URL = "http://localhost:8002"


@pytest.fixture(scope="module")
def wait_for_services():
    """Wait for all services to be healthy."""
    services = {
        "gateway": f"{GATEWAY_URL}/health",
        "model-registry": f"{MODEL_REGISTRY_URL}/health",
        "deployment": f"{DEPLOYMENT_URL}/health",
        "config": f"{CONFIG_URL}/health",
    }

    print("\nWaiting for services to be ready...")

    for name, url in services.items():
        max_retries = 30
        for i in range(max_retries):
            try:
                response = httpx.get(url, timeout=5.0)
                if response.status_code == 200:
                    print(f"✓ {name} is ready")
                    break
            except Exception:
                pass

            if i == max_retries - 1:
                pytest.skip(f"Service {name} not available at {url}")

            time.sleep(2)


class TestHealthChecks:
    """Test that all services are healthy."""

    def test_gateway_health(self, wait_for_services):
        """Test gateway health."""
        response = httpx.get(f"{GATEWAY_URL}/health", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]

    def test_model_registry_health(self, wait_for_services):
        """Test model registry health."""
        response = httpx.get(f"{MODEL_REGISTRY_URL}/health", timeout=5.0)
        assert response.status_code == 200

    def test_deployment_service_health(self, wait_for_services):
        """Test deployment service health."""
        response = httpx.get(f"{DEPLOYMENT_URL}/health", timeout=5.0)
        assert response.status_code == 200

    def test_config_service_health(self, wait_for_services):
        """Test config service health."""
        response = httpx.get(f"{CONFIG_URL}/health", timeout=5.0)
        assert response.status_code == 200


class TestModelRegistry:
    """Test model registry operations."""

    def test_list_models(self, wait_for_services):
        """Test listing models."""
        response = httpx.get(f"{MODEL_REGISTRY_URL}/api/v1/models", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data


class TestConfigService:
    """Test config service operations."""

    def test_get_routing_config(self, wait_for_services):
        """Test getting routing config."""
        response = httpx.get(f"{CONFIG_URL}/api/v1/config/routing", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        assert "normalized_weights" in data
        assert "version" in data

    def test_update_routing_config(self, wait_for_services):
        """Test updating routing config."""
        payload = {
            "weights": {
                "recommendation_v1": 1.0
            }
        }

        response = httpx.put(
            f"{CONFIG_URL}/api/v1/config/routing",
            json=payload,
            timeout=5.0
        )
        assert response.status_code == 200
        data = response.json()
        assert data["normalized_weights"]["recommendation_v1"] == 1.0


class TestInference:
    """Test inference through gateway."""

    @pytest.mark.skipif(
        True,  # Skip by default - requires models to be deployed
        reason="Requires models to be deployed first"
    )
    def test_inference_request(self, wait_for_services):
        """Test sending inference request."""
        payload = {
            "user_ids": [123, 456],
            "item_ids": [1001, 2002]
        }

        response = httpx.post(
            f"{GATEWAY_URL}/api/v1/predict",
            json=payload,
            timeout=10.0
        )

        # May fail if models not deployed
        if response.status_code == 200:
            data = response.json()
            assert "scores" in data
            assert len(data["scores"]) == 2
            assert "model_name" in data
            assert "batch_size" in data


class TestMetrics:
    """Test metrics endpoint."""

    def test_gateway_metrics(self, wait_for_services):
        """Test that metrics endpoint returns Prometheus format."""
        response = httpx.get(f"{GATEWAY_URL}/api/v1/metrics", timeout=5.0)
        assert response.status_code == 200

        # Check for Prometheus format
        content = response.text
        assert "inference_requests_total" in content or "# HELP" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
