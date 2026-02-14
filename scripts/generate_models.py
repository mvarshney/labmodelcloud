#!/usr/bin/env python3
"""Generate 35 simulated model variants and a manifest.json."""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_FAMILIES, MODEL_STORE_PATH


def generate_models():
    MODEL_STORE_PATH.mkdir(parents=True, exist_ok=True)

    manifest = {"models": []}

    for family_name, (matrix_size, count) in MODEL_FAMILIES.items():
        for i in range(count):
            model_id = f"{family_name}-v{i}"
            model_dir = MODEL_STORE_PATH / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # Generate random weights with a fixed seed for reproducibility
            rng = np.random.default_rng(seed=hash(model_id) % (2**31))
            weights = rng.standard_normal((matrix_size, matrix_size)).astype(np.float32)

            weights_path = model_dir / "weights.npy"
            np.save(weights_path, weights)

            mem_bytes = weights.nbytes
            manifest["models"].append({
                "model_id": model_id,
                "family": family_name,
                "matrix_size": matrix_size,
                "memory_mb": round(mem_bytes / 1e6, 1),
            })

            print(f"  Generated {model_id}: {matrix_size}x{matrix_size} ({mem_bytes / 1e6:.1f} MB)")

    manifest_path = MODEL_STORE_PATH / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {len(manifest['models'])} models in {MODEL_STORE_PATH}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    generate_models()
