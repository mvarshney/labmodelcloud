import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_STORE_PATH = Path(os.getenv("MODEL_STORE_PATH", PROJECT_ROOT / "model_store"))
MODEL_MANIFEST_PATH = MODEL_STORE_PATH / "manifest.json"

# Model families: name → (weight_matrix_size, count)
MODEL_FAMILIES = {
    "text-classifier": (1024, 10),
    "embedding-model": (2048, 10),
    "summarizer": (1536, 10),
    "sentiment-analyzer": (512, 5),
}

# Serving
MAX_MODELS_PER_REPLICA = int(os.getenv("MAX_MODELS_PER_REPLICA", "3"))
MAX_ONGOING_REQUESTS = int(os.getenv("MAX_ONGOING_REQUESTS", "8"))
TARGET_ONGOING_REQUESTS = int(os.getenv("TARGET_ONGOING_REQUESTS", "3"))
MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "10"))
UPSCALE_DELAY_S = int(os.getenv("UPSCALE_DELAY_S", "10"))
DOWNSCALE_DELAY_S = int(os.getenv("DOWNSCALE_DELAY_S", "60"))

# Simulated latency for weight loading (seconds)
SIMULATED_DOWNLOAD_LATENCY = float(os.getenv("SIMULATED_DOWNLOAD_LATENCY", "0.5"))

# Worker resources
WORKER_NUM_CPUS = float(os.getenv("WORKER_NUM_CPUS", "1"))
WORKER_MEMORY_BYTES = int(os.getenv("WORKER_MEMORY_BYTES", str(200 * 1024 * 1024)))
INGRESS_NUM_CPUS = float(os.getenv("INGRESS_NUM_CPUS", "0.5"))
