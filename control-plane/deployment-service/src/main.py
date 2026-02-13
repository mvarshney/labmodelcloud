"""Deployment Service - FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .api.routes import router, init_deployer
from .services.deployer import ModelDeployer

# Configuration from environment
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "models")
TRITON_URL = os.getenv("TRITON_URL", "triton:8000")
TRITON_MODEL_REPO = os.getenv("TRITON_MODEL_REPO", "/models")
MODEL_REGISTRY_URL = os.getenv("MODEL_REGISTRY_URL", "http://model-registry:8001")
CONFIG_SERVICE_URL = os.getenv("CONFIG_SERVICE_URL", "http://config-service:8002")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8003"))

# Create FastAPI app
app = FastAPI(
    title="Deployment Service",
    description="Model deployment orchestration service",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize deployer
deployer = ModelDeployer(
    s3_endpoint=S3_ENDPOINT,
    s3_access_key=S3_ACCESS_KEY,
    s3_secret_key=S3_SECRET_KEY,
    s3_bucket=S3_BUCKET,
    triton_url=TRITON_URL,
    triton_model_repo=TRITON_MODEL_REPO,
    model_registry_url=MODEL_REGISTRY_URL,
    config_service_url=CONFIG_SERVICE_URL,
)
init_deployer(deployer)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["deployment"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "deployment-service",
        "status": "healthy",
        "triton_url": TRITON_URL,
        "s3_endpoint": S3_ENDPOINT,
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    triton_live = deployer.triton_client.is_server_live()
    triton_ready = deployer.triton_client.is_server_ready()

    return {
        "status": "healthy" if triton_live else "degraded",
        "triton_live": triton_live,
        "triton_ready": triton_ready,
        "triton_url": TRITON_URL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=True,
        log_level="info",
    )
