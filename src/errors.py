class ModelNotFoundError(Exception):
    """Raised when a requested model_id is not in the registry."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' not found in registry")


class CudaOutOfMemoryError(Exception):
    """Raised when model loading fails due to memory pressure."""

    def __init__(self, model_id: str, detail: str = ""):
        self.model_id = model_id
        super().__init__(
            f"Out of memory loading model '{model_id}'"
            + (f": {detail}" if detail else "")
        )
