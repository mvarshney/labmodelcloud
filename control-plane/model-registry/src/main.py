"""Model Registry Service - FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .api.routes import router, init_registry
from .services.registry import ModelRegistry

# Configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8001"))

# Create FastAPI app
app = FastAPI(
    title="Model Registry Service",
    description="ML Model Registry with MLflow integration",
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

# Initialize registry
registry = ModelRegistry(mlflow_tracking_uri=MLFLOW_TRACKING_URI)
init_registry(registry)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["models"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "model-registry",
        "status": "healthy",
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "models_count": len(registry.list_models()),
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
