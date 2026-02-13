"""API routes for Config Service."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from ..models.schemas import RoutingConfig, RoutingConfigResponse
from ..services.config import ConfigService

router = APIRouter()

# Global config service instance
_config_service: Optional[ConfigService] = None


def get_config_service() -> ConfigService:
    """Dependency to get config service instance."""
    if _config_service is None:
        raise RuntimeError("Config service not initialized")
    return _config_service


def init_config_service(service: ConfigService):
    """Initialize the global config service instance."""
    global _config_service
    _config_service = service


@router.get("/config/routing", response_model=RoutingConfigResponse)
async def get_routing_config(
    service: ConfigService = Depends(get_config_service),
) -> RoutingConfigResponse:
    """
    Get current routing configuration.

    This endpoint is polled by the gateway service to get traffic routing weights.
    """
    try:
        config, normalized, version = service.get_routing_config()
        return RoutingConfigResponse(
            config=config,
            normalized_weights=normalized,
            version=version
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get routing config: {str(e)}"
        )


@router.put("/config/routing", response_model=RoutingConfigResponse)
async def update_routing_config(
    config: RoutingConfig,
    service: ConfigService = Depends(get_config_service),
) -> RoutingConfigResponse:
    """
    Update routing configuration.

    This endpoint allows updating traffic routing weights dynamically.
    The gateway will pick up changes on its next polling cycle (default: 30s).

    Example use cases:
    - Gradual rollout: {"recommendation_v1": 0.9, "recommendation_v2": 0.1}
    - A/B testing: {"recommendation_v1": 0.5, "recommendation_v2": 0.5}
    - Full rollout: {"recommendation_v2": 1.0}
    """
    try:
        config, normalized, version = service.update_routing_config(config)
        return RoutingConfigResponse(
            config=config,
            normalized_weights=normalized,
            version=version
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update routing config: {str(e)}"
        )
