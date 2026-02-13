"""Config Service - FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .api.routes import router, init_config_service
from .services.config import ConfigService

# Configuration from environment
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8002"))

# Create FastAPI app
app = FastAPI(
    title="Config Service",
    description="Configuration management service for routing and deployment settings",
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

# Initialize config service
config_service = ConfigService()
init_config_service(config_service)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["config"])


@app.get("/")
async def root():
    """Health check endpoint."""
    _, _, version = config_service.get_routing_config()
    return {
        "service": "config-service",
        "status": "healthy",
        "config_version": version,
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    config, normalized, version = config_service.get_routing_config()
    return {
        "status": "healthy",
        "config_version": version,
        "models": list(config.weights.keys()),
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
