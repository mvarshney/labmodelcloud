"""
Prometheus metrics for monitoring inference performance.

Key metrics for ML serving:
1. Request rate (QPS) - How many requests per second
2. Latency (P50, P95, P99) - Response time distribution
3. Model selection - Traffic distribution across models
4. Batch size - To observe dynamic batching efficiency
5. Error rate - Failed inference requests
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time


# Request counter by model
inference_requests_total = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["model_name", "status"]  # Labels: model_name, status (success/error)
)

# Latency histogram
inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Inference request latency in seconds",
    ["model_name"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

# Batch size histogram (KEY metric for observing dynamic batching!)
inference_batch_size = Histogram(
    "inference_batch_size",
    "Batch size used in inference requests",
    ["model_name"],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)

# Current routing weights
routing_weights = Gauge(
    "routing_weights",
    "Current routing weights for models",
    ["model_name"]
)

# Config version
config_version = Gauge(
    "config_version",
    "Current config version number"
)


class MetricsCollector:
    """
    Metrics collector for inference gateway.

    Usage:
        collector = MetricsCollector()

        # Record successful request
        with collector.measure_latency("recommendation_v1"):
            result = do_inference()
        collector.record_success("recommendation_v1", batch_size=8)
    """

    def __init__(self):
        """Initialize metrics collector."""
        pass

    def measure_latency(self, model_name: str):
        """
        Context manager to measure latency.

        Usage:
            with collector.measure_latency("model_v1"):
                do_inference()
        """
        return inference_latency_seconds.labels(model_name=model_name).time()

    def record_success(self, model_name: str, batch_size: int):
        """
        Record a successful inference request.

        Args:
            model_name: Model that served the request
            batch_size: Batch size used
        """
        inference_requests_total.labels(
            model_name=model_name,
            status="success"
        ).inc()

        inference_batch_size.labels(
            model_name=model_name
        ).observe(batch_size)

    def record_error(self, model_name: str):
        """
        Record a failed inference request.

        Args:
            model_name: Model that failed
        """
        inference_requests_total.labels(
            model_name=model_name,
            status="error"
        ).inc()

    def update_routing_weights(self, weights: dict):
        """
        Update routing weight metrics.

        Args:
            weights: Model name to weight mapping
        """
        for model_name, weight in weights.items():
            routing_weights.labels(model_name=model_name).set(weight)

    def update_config_version(self, version: int):
        """
        Update config version metric.

        Args:
            version: Config version number
        """
        config_version.set(version)


def get_metrics() -> bytes:
    """
    Get Prometheus metrics in text format.

    Returns:
        Metrics in Prometheus text format
    """
    return generate_latest()


def get_content_type() -> str:
    """
    Get content type for metrics endpoint.

    Returns:
        Content type string
    """
    return CONTENT_TYPE_LATEST
