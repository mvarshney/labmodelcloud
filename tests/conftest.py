import json
import pytest
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def tmp_model_store(tmp_path):
    """Create a temporary model store with a few test models."""
    families = {
        "test-model": (64, 3),  # Small models for fast tests
    }
    manifest = {"models": []}

    for family_name, (matrix_size, count) in families.items():
        for i in range(count):
            model_id = f"{family_name}-v{i}"
            model_dir = tmp_path / model_id
            model_dir.mkdir()

            rng = np.random.default_rng(seed=42 + i)
            weights = rng.standard_normal((matrix_size, matrix_size)).astype(np.float32)
            np.save(model_dir / "weights.npy", weights)

            manifest["models"].append({
                "model_id": model_id,
                "family": family_name,
                "matrix_size": matrix_size,
                "memory_mb": round(weights.nbytes / 1e6, 2),
            })

    with open(tmp_path / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return tmp_path
