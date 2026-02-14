"""Deployment graph: binds ingress → worker with autoscaling config."""

from ray import serve

from src.config import (
    DOWNSCALE_DELAY_S,
    INGRESS_NUM_CPUS,
    MAX_ONGOING_REQUESTS,
    MAX_REPLICAS,
    MIN_REPLICAS,
    TARGET_ONGOING_REQUESTS,
    UPSCALE_DELAY_S,
    WORKER_NUM_CPUS,
)
from src.serving.ingress import Ingress
from src.serving.worker import MultiplexedModelWorker

worker = MultiplexedModelWorker.options(
    ray_actor_options={"num_cpus": WORKER_NUM_CPUS},
    autoscaling_config={
        "target_ongoing_requests": TARGET_ONGOING_REQUESTS,
        "min_replicas": MIN_REPLICAS,
        "max_replicas": MAX_REPLICAS,
        "upscale_delay_s": UPSCALE_DELAY_S,
        "downscale_delay_s": DOWNSCALE_DELAY_S,
    },
    max_ongoing_requests=MAX_ONGOING_REQUESTS,
    health_check_period_s=10,
    health_check_timeout_s=30,
).bind()

ingress = Ingress.options(
    ray_actor_options={"num_cpus": INGRESS_NUM_CPUS},
).bind(worker)
