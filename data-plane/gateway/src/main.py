"""
Gateway Service - FastAPI application.

This is the entry point for the data plane gateway. It coordinates:
1. Triton client for inference
2. Weighted router for model selection
3. Config client for polling control plane
4. Prometheus metrics for observability
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
from contextlib import asynccontextmanager

from .api.routes import router, init_gateway
from .services.triton_client import TritonInferenceClient
from .services.router import WeightedRouter
from .services.config_client import ConfigClient
from .metrics.prometheus import MetricsCollector

# Configuration from environment
TRITON_URL = os.getenv("TRITON_URL", "triton:8001")  # gRPC port
CONFIG_SERVICE_URL = os.getenv("CONFIG_SERVICE_URL", "http://config-service:8002")
CONFIG_POLL_INTERVAL = int(os.getenv("CONFIG_POLL_INTERVAL", "30"))
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8000"))

# Global instances
triton_client: TritonInferenceClient
weighted_router: WeightedRouter
config_client: ConfigClient
metrics_collector: MetricsCollector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.

    Startup:
    - Initialize Triton client
    - Start config polling loop

    Shutdown:
    - Stop config polling
    """
    global triton_client, weighted_router, config_client, metrics_collector

    # Startup
    print("=" * 60)
    print("Starting Gateway Service")
    print("=" * 60)

    # Initialize Triton client
    print(f"\n[1/4] Initializing Triton client: {TRITON_URL}")
    triton_client = TritonInferenceClient(TRITON_URL)

    # Wait for Triton to be ready
    max_retries = 30
    for i in range(max_retries):
        if triton_client.is_server_ready():
            print("✓ Triton server is ready")
            break
        print(f"  Waiting for Triton... ({i+1}/{max_retries})")
        await asyncio.sleep(2)
    else:
        print("⚠ Warning: Triton server not ready after 60s")

    # Initialize router
    print("\n[2/4] Initializing weighted router")
    weighted_router = WeightedRouter()
    print("✓ Router initialized")

    # Initialize metrics
    print("\n[3/4] Initializing metrics collector")
    metrics_collector = MetricsCollector()
    print("✓ Metrics collector initialized")

    # Initialize config client with callback
    print(f"\n[4/4] Initializing config client: {CONFIG_SERVICE_URL}")
    print(f"  Poll interval: {CONFIG_POLL_INTERVAL}s")

    def on_config_update(weights: dict):
        """Callback when config changes."""
        weighted_router.update_weights(weights)
        metrics_collector.update_routing_weights(weights)

    config_client = ConfigClient(
        config_service_url=CONFIG_SERVICE_URL,
        poll_interval=CONFIG_POLL_INTERVAL,
        on_config_update=on_config_update
    )

    # Start config polling
    config_client.start()

    # Fetch initial config
    print("  Fetching initial config...")
    initial_config = await config_client.fetch_config()
    if initial_config:
        weights = initial_config.get("normalized_weights", {})
        if weights:
            weighted_router.update_weights(weights)
            metrics_collector.update_routing_weights(weights)
            print(f"✓ Initial config loaded: {weights}")
    else:
        print("⚠ Warning: Could not fetch initial config, using defaults")

    # Initialize gateway routes
    init_gateway(triton_client, weighted_router, metrics_collector)

    print("\n" + "=" * 60)
    print("✓ Gateway Service Started Successfully")
    print("=" * 60)
    print(f"Inference endpoint: http://0.0.0.0:{SERVICE_PORT}/api/v1/predict")
    print(f"Metrics endpoint: http://0.0.0.0:{SERVICE_PORT}/api/v1/metrics")
    print("=" * 60 + "\n")

    yield

    # Shutdown
    print("\nShutting down gateway service...")
    config_client.stop()
    print("✓ Gateway service stopped")


# Create FastAPI app
app = FastAPI(
    title="Inference Gateway",
    description="ML inference gateway with dynamic batching and traffic routing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["inference"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "gateway",
        "status": "healthy",
        "triton_url": TRITON_URL,
        "config_service_url": CONFIG_SERVICE_URL,
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    triton_ready = triton_client.is_server_ready() if triton_client else False
    routing_weights = weighted_router.get_weights() if weighted_router else {}

    return {
        "status": "healthy" if triton_ready else "degraded",
        "triton_ready": triton_ready,
        "triton_url": TRITON_URL,
        "routing_weights": routing_weights,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=False,  # Disable reload to avoid issues with lifespan
        log_level="info",
    )
