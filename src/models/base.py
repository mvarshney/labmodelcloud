from __future__ import annotations

import numpy as np


class SimulatedModel:
    """A model that holds a numpy weight matrix and does real matrix multiplication.

    Simulates realistic CPU load and memory footprint without requiring a GPU.
    """

    def __init__(self, model_id: str, weights: np.ndarray):
        self.model_id = model_id
        self.weights = weights
        self.shape = weights.shape
        self._memory_bytes = weights.nbytes

    def predict(self, input_data: list[list[float]]) -> list[list[float]]:
        """Run inference: matrix multiply input against weights.

        Args:
            input_data: 2D list where each row is a feature vector.
                        Columns must match weights.shape[0].

        Returns:
            2D list of prediction results.
        """
        x = np.array(input_data, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Pad or truncate input columns to match weight rows
        if x.shape[1] < self.weights.shape[0]:
            x = np.pad(x, ((0, 0), (0, self.weights.shape[0] - x.shape[1])))
        elif x.shape[1] > self.weights.shape[0]:
            x = x[:, : self.weights.shape[0]]
        result = x @ self.weights
        return result.tolist()

    @property
    def memory_bytes(self) -> int:
        return self._memory_bytes

    def __repr__(self) -> str:
        return f"SimulatedModel(id={self.model_id!r}, shape={self.shape}, mem={self._memory_bytes / 1e6:.1f}MB)"

    def __del__(self):
        # Explicitly free the weight array to release memory promptly
        self.weights = None
