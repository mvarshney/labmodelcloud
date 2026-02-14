from ray.serve.metrics import Counter, Histogram

# Cache misses — incremented when a model must be loaded from store
model_cache_miss_total = Counter(
    "model_cache_miss_total",
    description="Total number of model cache misses requiring weight loading.",
    tag_keys=("model_id",),
)

# Inference request latency (end-to-end from ingress perspective)
inference_request_latency_seconds = Histogram(
    "inference_request_latency_seconds",
    description="End-to-end inference request latency in seconds.",
    boundaries=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    tag_keys=("model_id",),
)

# Model loading duration (weight download + deserialization)
model_load_duration_seconds = Histogram(
    "model_load_duration_seconds",
    description="Time to load model weights from store in seconds.",
    boundaries=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    tag_keys=("model_id",),
)
