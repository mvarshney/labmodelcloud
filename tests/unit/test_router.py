"""Unit tests for WeightedRouter."""

import pytest
from collections import Counter


# Mock the router module
class WeightedRouter:
    """Weighted random router (copy for testing)."""

    def __init__(self):
        self._weights = {}
        self._normalized_weights = {}

    def update_weights(self, weights):
        self._weights = weights.copy()
        self._normalized_weights = self._normalize_weights(weights)

    def select_model(self):
        if not self._normalized_weights:
            return None

        import random
        rand = random.random()
        cumulative = 0.0

        for model_name, weight in self._normalized_weights.items():
            cumulative += weight
            if rand < cumulative:
                return model_name

        return list(self._normalized_weights.keys())[0]

    def get_weights(self):
        return self._normalized_weights.copy()

    @staticmethod
    def _normalize_weights(weights):
        total = sum(weights.values())
        if total == 0:
            n = len(weights)
            return {model: 1.0 / n for model in weights}
        return {model: weight / total for model, weight in weights.items()}


class TestWeightedRouter:
    """Test cases for WeightedRouter."""

    def test_single_model(self):
        """Test routing with single model."""
        router = WeightedRouter()
        router.update_weights({"model_v1": 1.0})

        # All requests should go to model_v1
        for _ in range(100):
            assert router.select_model() == "model_v1"

    def test_equal_weights(self):
        """Test routing with equal weights."""
        router = WeightedRouter()
        router.update_weights({
            "model_v1": 0.5,
            "model_v2": 0.5
        })

        # Over many requests, distribution should be ~50/50
        results = [router.select_model() for _ in range(1000)]
        counts = Counter(results)

        # Allow 10% tolerance
        assert 450 <= counts["model_v1"] <= 550
        assert 450 <= counts["model_v2"] <= 550

    def test_weighted_distribution(self):
        """Test routing with 70/30 split."""
        router = WeightedRouter()
        router.update_weights({
            "model_v1": 0.7,
            "model_v2": 0.3
        })

        # Over many requests, distribution should be ~70/30
        results = [router.select_model() for _ in range(1000)]
        counts = Counter(results)

        # Allow 10% tolerance
        assert 650 <= counts["model_v1"] <= 750
        assert 250 <= counts["model_v2"] <= 350

    def test_weight_normalization(self):
        """Test that weights are normalized correctly."""
        router = WeightedRouter()

        # Weights that don't sum to 1.0
        router.update_weights({
            "model_v1": 70,
            "model_v2": 30
        })

        weights = router.get_weights()
        assert abs(weights["model_v1"] - 0.7) < 0.001
        assert abs(weights["model_v2"] - 0.3) < 0.001

    def test_zero_weights(self):
        """Test handling of zero weights."""
        router = WeightedRouter()
        router.update_weights({
            "model_v1": 0,
            "model_v2": 0
        })

        weights = router.get_weights()
        # Should distribute equally
        assert abs(weights["model_v1"] - 0.5) < 0.001
        assert abs(weights["model_v2"] - 0.5) < 0.001

    def test_empty_weights(self):
        """Test with no models."""
        router = WeightedRouter()
        assert router.select_model() is None

    def test_weight_update(self):
        """Test updating weights dynamically."""
        router = WeightedRouter()

        # Start with v1 only
        router.update_weights({"model_v1": 1.0})
        assert router.select_model() == "model_v1"

        # Switch to v2 only
        router.update_weights({"model_v2": 1.0})
        assert router.select_model() == "model_v2"

        # Switch to 50/50
        router.update_weights({
            "model_v1": 0.5,
            "model_v2": 0.5
        })

        results = [router.select_model() for _ in range(100)]
        assert "model_v1" in results
        assert "model_v2" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
