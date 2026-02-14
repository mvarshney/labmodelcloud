"""End-to-end tests that deploy the full Ray Serve app."""

import pytest
import ray
from ray import serve
import httpx
import time
from unittest.mock import patch
from pathlib import Path


@pytest.fixture(scope="module")
def serve_app(tmp_path_factory):
    """Deploy the full app with a temporary model store."""
    import json
    import numpy as np

    tmp_store = tmp_path_factory.mktemp("model_store")

    # Create test models
    manifest = {"models": []}
    for i in range(5):
        model_id = f"test-model-v{i}"
        model_dir = tmp_store / model_id
        model_dir.mkdir()
        rng = np.random.default_rng(seed=i)
        weights = rng.standard_normal((32, 32)).astype(np.float32)
        np.save(model_dir / "weights.npy", weights)
        manifest["models"].append({
            "model_id": model_id,
            "family": "test-model",
            "matrix_size": 32,
            "memory_mb": round(weights.nbytes / 1e6, 2),
        })
    with open(tmp_store / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # Patch config before importing app
    with (
        patch("src.config.MODEL_STORE_PATH", tmp_store),
        patch("src.config.MODEL_MANIFEST_PATH", tmp_store / "manifest.json"),
        patch("src.models.store.MODEL_STORE_PATH", tmp_store),
        patch("src.models.store.SIMULATED_DOWNLOAD_LATENCY", 0.01),
        patch("src.models.registry.MODEL_MANIFEST_PATH", tmp_store / "manifest.json"),
        patch("src.config.WORKER_NUM_CPUS", 0),
        patch("src.config.INGRESS_NUM_CPUS", 0),
    ):
        ray.init(ignore_reinit_error=True, num_cpus=4)

        from src.serving.worker import MultiplexedModelWorker
        from src.serving.ingress import Ingress

        worker = MultiplexedModelWorker.options(
            ray_actor_options={"num_cpus": 0},
            num_replicas=1,
            max_ongoing_requests=8,
        ).bind()
        ingress_app = Ingress.options(
            ray_actor_options={"num_cpus": 0},
        ).bind(worker)

        serve.run(ingress_app, name="test-app", route_prefix="/")
        time.sleep(2)  # Wait for deployment

        yield "http://127.0.0.1:8000"

        serve.shutdown()
        ray.shutdown()


@pytest.mark.skipif(
    not pytest.importorskip("ray", reason="Ray not installed"),
    reason="Ray not available",
)
class TestE2E:
    def test_health(self, serve_app):
        resp = httpx.get(f"{serve_app}/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_list_models(self, serve_app):
        resp = httpx.get(f"{serve_app}/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        assert len(models) == 5

    def test_predict(self, serve_app):
        resp = httpx.post(
            f"{serve_app}/v1/predict/test-model-v0",
            json={"input": [[1.0, 2.0, 3.0]]},
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "test-model-v0"
        assert len(data["predictions"]) == 1

    def test_predict_404(self, serve_app):
        resp = httpx.post(
            f"{serve_app}/v1/predict/nonexistent",
            json={"input": [[1.0]]},
        )
        assert resp.status_code == 404

    def test_model_info(self, serve_app):
        resp = httpx.get(f"{serve_app}/v1/models/test-model-v0")
        assert resp.status_code == 200
        assert resp.json()["model_id"] == "test-model-v0"

    def test_model_switching(self, serve_app):
        """Test that multiple models can be served, triggering eviction."""
        for i in range(5):
            resp = httpx.post(
                f"{serve_app}/v1/predict/test-model-v{i}",
                json={"input": [[float(i)] * 10]},
                timeout=30.0,
            )
            assert resp.status_code == 200
            assert resp.json()["model_id"] == f"test-model-v{i}"
