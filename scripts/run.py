#!/usr/bin/env python3
"""Start Ray and deploy the Serve application."""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Tell Ray Dashboard where Prometheus is running
os.environ.setdefault("RAY_PROMETHEUS_HOST", "http://localhost:9090")

import ray
from ray import serve


def main():
    # Initialize Ray (connects to existing cluster or starts a local one)
    ray.init(ignore_reinit_error=True)
    print(f"Ray dashboard: http://127.0.0.1:8265")

    # Import the app entry point (the bound ingress deployment)
    from src.app import ingress

    # Deploy
    serve.run(ingress, name="model-multiplexer", route_prefix="/")
    print("Serve app deployed at http://127.0.0.1:8000")
    print("  GET  /v1/models              — list all models")
    print("  GET  /v1/models/{model_id}   — model info")
    print("  POST /v1/predict/{model_id}  — run inference")
    print("  GET  /health                 — health check")
    print("\nPress Ctrl+C to shut down.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
