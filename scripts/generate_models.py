"""
Generate dummy PyTorch recommendation models for Triton deployment.

This script creates two versions of a simple recommendation model:
- Model V1: 3-layer MLP with user/item embeddings
- Model V2: Same architecture, different random weights (to simulate A/B testing)

Both models are exported as TorchScript (.pt) files compatible with Triton.
"""

import torch
import torch.nn as nn
import os
from pathlib import Path


class RecommendationModel(nn.Module):
    """
    Simple recommendation model using embeddings and MLP.

    Architecture:
    1. User embedding (vocab_size=10000, dim=32)
    2. Item embedding (vocab_size=50000, dim=32)
    3. Concatenate embeddings → [batch, 64]
    4. 3-layer MLP: 64 → 128 → 64 → 1
    5. Sigmoid activation → recommendation score [0, 1]
    """

    def __init__(self, user_vocab_size=10000, item_vocab_size=50000, embedding_dim=32):
        super().__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim)
        self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim)

        # MLP layers
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            user_ids: [batch_size] - User IDs
            item_ids: [batch_size] - Item IDs

        Returns:
            scores: [batch_size, 1] - Recommendation scores between 0 and 1
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # [batch, 32]
        item_emb = self.item_embedding(item_ids)  # [batch, 32]

        # Concatenate
        x = torch.cat([user_emb, item_emb], dim=1)  # [batch, 64]

        # MLP
        x = self.relu(self.fc1(x))  # [batch, 128]
        x = self.relu(self.fc2(x))  # [batch, 64]
        x = self.fc3(x)              # [batch, 1]
        x = self.sigmoid(x)          # [batch, 1]

        return x


def export_model(model: nn.Module, output_path: str, model_name: str):
    """
    Export PyTorch model as TorchScript for Triton.

    Args:
        model: PyTorch model to export
        output_path: Directory to save the model
        model_name: Name of the model (e.g., 'recommendation_v1')
    """
    # Create output directory structure for Triton
    # Format: models/{model_name}/{version}/model.pt
    model_dir = Path(output_path) / model_name / "1"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Create example inputs for tracing
    batch_size = 8
    example_user_ids = torch.randint(0, 10000, (batch_size,))
    example_item_ids = torch.randint(0, 50000, (batch_size,))

    # Trace the model
    traced_model = torch.jit.trace(model, (example_user_ids, example_item_ids))

    # Save as TorchScript
    model_path = model_dir / "model.pt"
    traced_model.save(str(model_path))

    print(f"✓ Exported {model_name} to {model_path}")

    # Test the exported model
    loaded_model = torch.jit.load(str(model_path))
    with torch.no_grad():
        output = loaded_model(example_user_ids, example_item_ids)
    print(f"  Model output shape: {output.shape}, sample values: {output[:3].squeeze().tolist()}")


def main():
    """Generate and export two model versions."""

    output_path = Path(__file__).parent.parent / "triton" / "models"
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Dummy Recommendation Models")
    print("=" * 60)

    # Model V1 - with seed for reproducibility
    print("\n[1/2] Creating Model V1...")
    torch.manual_seed(42)
    model_v1 = RecommendationModel()
    export_model(model_v1, str(output_path), "recommendation_v1")

    # Model V2 - different seed for different weights
    print("\n[2/2] Creating Model V2...")
    torch.manual_seed(123)
    model_v2 = RecommendationModel()
    export_model(model_v2, str(output_path), "recommendation_v2")

    print("\n" + "=" * 60)
    print("✓ Model generation complete!")
    print("=" * 60)
    print(f"\nModels saved to: {output_path}")
    print("\nNext steps:")
    print("1. Create Triton config.pbtxt files for each model")
    print("2. Upload models to MinIO (use upload_to_minio.py)")
    print("3. Deploy models via deployment service")


if __name__ == "__main__":
    main()
