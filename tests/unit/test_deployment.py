"""Unit tests for deployment service components."""

import pytest
from pathlib import Path


class TestConfigNormalization:
    """Test config weight normalization."""

    @staticmethod
    def normalize_weights(weights):
        """Helper function to normalize weights."""
        total = sum(weights.values())
        if total == 0:
            n = len(weights)
            return {model: 1.0 / n for model in weights}
        return {model: weight / total for model, weight in weights.items()}

    def test_already_normalized(self):
        """Test weights that are already normalized."""
        weights = {"v1": 0.7, "v2": 0.3}
        result = self.normalize_weights(weights)

        assert abs(result["v1"] - 0.7) < 0.001
        assert abs(result["v2"] - 0.3) < 0.001

    def test_unnormalized_weights(self):
        """Test weights that need normalization."""
        weights = {"v1": 70, "v2": 30}
        result = self.normalize_weights(weights)

        assert abs(result["v1"] - 0.7) < 0.001
        assert abs(result["v2"] - 0.3) < 0.001

    def test_zero_weights(self):
        """Test zero weights."""
        weights = {"v1": 0, "v2": 0}
        result = self.normalize_weights(weights)

        assert abs(result["v1"] - 0.5) < 0.001
        assert abs(result["v2"] - 0.5) < 0.001

    def test_single_model(self):
        """Test single model."""
        weights = {"v1": 100}
        result = self.normalize_weights(weights)

        assert abs(result["v1"] - 1.0) < 0.001


class TestModelPathParsing:
    """Test S3 path parsing."""

    @staticmethod
    def parse_s3_path(s3_path, default_bucket="models"):
        """Parse S3 path into bucket and key."""
        if s3_path.startswith("s3://"):
            parts = s3_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            bucket = default_bucket
            key = s3_path

        return bucket, key

    def test_full_s3_path(self):
        """Test full S3 path."""
        bucket, key = self.parse_s3_path("s3://mybucket/path/to/model.pt")
        assert bucket == "mybucket"
        assert key == "path/to/model.pt"

    def test_key_only(self):
        """Test key without bucket."""
        bucket, key = self.parse_s3_path("path/to/model.pt", "default-bucket")
        assert bucket == "default-bucket"
        assert key == "path/to/model.pt"

    def test_bucket_no_key(self):
        """Test bucket without key."""
        bucket, key = self.parse_s3_path("s3://mybucket")
        assert bucket == "mybucket"
        assert key == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
