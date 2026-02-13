"""Configuration management service."""

from typing import Dict
import threading

from ..models.schemas import RoutingConfig


class ConfigService:
    """
    Configuration management service.

    Stores and serves routing configuration for the data plane.
    In production, this would be backed by a database (Redis, etcd, etc.).
    """

    def __init__(self):
        """Initialize the config service."""
        # In-memory storage with thread safety
        self._lock = threading.Lock()
        self._routing_config: RoutingConfig = RoutingConfig(
            weights={"recommendation_v1": 1.0}  # Default config
        )
        self._config_version = 1

    def get_routing_config(self) -> tuple[RoutingConfig, Dict[str, float], int]:
        """
        Get current routing configuration.

        Returns:
            Tuple of (config, normalized_weights, version)
        """
        with self._lock:
            normalized = self._normalize_weights(self._routing_config.weights)
            return self._routing_config, normalized, self._config_version

    def update_routing_config(self, config: RoutingConfig) -> tuple[RoutingConfig, Dict[str, float], int]:
        """
        Update routing configuration atomically.

        Args:
            config: New routing configuration

        Returns:
            Tuple of (config, normalized_weights, version)
        """
        with self._lock:
            self._routing_config = config
            self._config_version += 1
            normalized = self._normalize_weights(config.weights)

            print(f"✓ Routing config updated to version {self._config_version}")
            print(f"  Weights: {config.weights}")
            print(f"  Normalized: {normalized}")

            return config, normalized, self._config_version

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.

        Args:
            weights: Raw weights

        Returns:
            Normalized weights (sum to 1.0)
        """
        total = sum(weights.values())
        if total == 0:
            # Equal distribution if all weights are 0
            n = len(weights)
            return {model: 1.0 / n for model in weights}

        return {model: weight / total for model, weight in weights.items()}
