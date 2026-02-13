"""
Traffic routing service with weighted random selection.

This implements a simple but effective routing strategy for A/B testing
and gradual rollouts.
"""

import random
import threading
from typing import Dict, Optional


class WeightedRouter:
    """
    Weighted random router for model selection.

    Routes traffic to different models based on configured weights.
    Thread-safe for concurrent requests.
    """

    def __init__(self):
        """Initialize router with default config."""
        self._lock = threading.Lock()
        self._weights: Dict[str, float] = {"recommendation_v1": 1.0}
        self._normalized_weights: Dict[str, float] = {"recommendation_v1": 1.0}

    def update_weights(self, weights: Dict[str, float]):
        """
        Update routing weights atomically.

        Args:
            weights: New weights mapping (will be normalized)
        """
        with self._lock:
            self._weights = weights.copy()
            self._normalized_weights = self._normalize_weights(weights)

            print(f"✓ Router weights updated: {self._normalized_weights}")

    def select_model(self) -> Optional[str]:
        """
        Select a model based on weighted random selection.

        This is the core routing algorithm!

        Algorithm:
        1. Generate random number in [0, 1)
        2. Iterate through models, accumulating weights
        3. Return first model where cumulative weight > random number

        Example with weights {v1: 0.7, v2: 0.3}:
        - Random = 0.2 → cumulative = 0.7 (v1) → return v1
        - Random = 0.8 → cumulative = 0.7 (v1) → cumulative = 1.0 (v2) → return v2

        This gives v1 70% of traffic, v2 30% of traffic.

        Returns:
            Selected model name, or None if no models configured
        """
        with self._lock:
            if not self._normalized_weights:
                return None

            # Generate random number [0, 1)
            rand = random.random()

            # Weighted selection
            cumulative = 0.0
            for model_name, weight in self._normalized_weights.items():
                cumulative += weight
                if rand < cumulative:
                    return model_name

            # Fallback (should not reach here with normalized weights)
            return list(self._normalized_weights.keys())[0]

    def get_weights(self) -> Dict[str, float]:
        """Get current normalized weights."""
        with self._lock:
            return self._normalized_weights.copy()

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.

        Args:
            weights: Raw weights

        Returns:
            Normalized weights
        """
        total = sum(weights.values())
        if total == 0:
            n = len(weights)
            return {model: 1.0 / n for model in weights}

        return {model: weight / total for model, weight in weights.items()}
