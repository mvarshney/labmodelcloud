from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

class PrometheusMetrics:
    def __init__(self, gateway_url: str):
        self.registry = CollectorRegistry()
        self.latency_gauge = Gauge('model_inference_latency', 'Latency of model inference in seconds', ['model_name'], registry=self.registry)
        self.gateway_url = gateway_url

    def record_latency(self, model_name: str, latency: float):
        self.latency_gauge.labels(model_name=model_name).set(latency)
        push_to_gateway(self.gateway_url, job='model_inference', registry=self.registry)