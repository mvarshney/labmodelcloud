# ML Model Serving Platform

A production-grade ML model serving platform built as a learning exercise to understand modern AI infrastructure patterns. This platform demonstrates key concepts in ML serving including **Triton Inference Server**, **dynamic batching**, distributed systems patterns, and service orchestration.

## 🎯 Learning Objectives

This project is designed to teach:

1. **[P0] Triton Integration & Dynamic Batching** - How to serve PyTorch models with automatic batching for GPU efficiency
2. **Config Sync & Service Discovery** - Distributed systems patterns for configuration management
3. **Traffic Routing** - Weighted routing for A/B testing and gradual rollouts
4. **Observability** - Prometheus metrics and Grafana dashboards

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Control Plane                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Model      │  │  Deployment  │  │    Config    │  │
│  │  Registry    │  │   Service    │  │   Service    │  │
│  │  (MLflow)    │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          │ REST API
                          │
┌─────────────────────────────────────────────────────────┐
│                      Data Plane                          │
│  ┌────────────────────────────────────────────────┐     │
│  │           Python Gateway (FastAPI)             │     │
│  │  • Weighted routing                            │     │
│  │  • Config polling (30s)                        │     │
│  │  • Prometheus metrics                          │     │
│  └─────────────────────┬──────────────────────────┘     │
│                        │ gRPC                            │
│  ┌─────────────────────▼──────────────────────────┐     │
│  │         Triton Inference Server                │     │
│  │  • PyTorch backend                             │     │
│  │  • Dynamic batching (max_batch_size=32)        │     │
│  │  • CPU execution                               │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 📦 Components

### Control Plane
- **Model Registry**: Manages model metadata with MLflow integration
- **Deployment Service**: Orchestrates model deployment to Triton
- **Config Service**: Stores routing configuration with atomic updates

### Data Plane
- **Gateway**: HTTP API for inference with routing and metrics
- **Triton Server**: NVIDIA Triton for high-performance model serving

### Infrastructure
- **MinIO**: S3-compatible storage for model artifacts
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- 8GB RAM (for running all services)

### 1. Generate Models

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Generate dummy PyTorch models
python scripts/generate_models.py

# Create Triton config files
python scripts/create_triton_configs.py
```

This creates:
- `triton/models/recommendation_v1/1/model.pt`
- `triton/models/recommendation_v2/1/model.pt`
- Config files with dynamic batching enabled

### 2. Start Services

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f gateway
```

Services will be available at:
- Gateway: http://localhost:8000
- Model Registry: http://localhost:8001
- Config Service: http://localhost:8002
- Deployment Service: http://localhost:8003
- MLflow: http://localhost:5000
- MinIO Console: http://localhost:9001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### 3. Deploy Models

```bash
# Upload models to MinIO
python scripts/upload_to_minio.py

# Deploy recommendation_v1
python scripts/deploy_model.py --model-name recommendation_v1 --version 1

# Deploy recommendation_v2 (for A/B testing)
python scripts/deploy_model.py --model-name recommendation_v2 --version 1
```

### 4. Send Inference Requests

```bash
# Single request
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_ids": [123, 456, 789],
    "item_ids": [1001, 2002, 3003]
  }'

# Response
{
  "scores": [0.87, 0.54, 0.92],
  "model_name": "recommendation_v1",
  "batch_size": 3
}
```

### 5. Update Traffic Routing

```bash
# 70% to v1, 30% to v2 (A/B test)
curl -X PUT http://localhost:8002/api/v1/config/routing \
  -H "Content-Type: application/json" \
  -d '{
    "weights": {
      "recommendation_v1": 0.7,
      "recommendation_v2": 0.3
    }
  }'

# Gateway will pick up changes in ~30s
```

## 🔬 Understanding Dynamic Batching

**Dynamic batching** is Triton's key feature for improving throughput. Here's how it works:

### Without Batching
```
Request 1 arrives → Process immediately → Return result (10ms)
Request 2 arrives → Process immediately → Return result (10ms)
Request 3 arrives → Process immediately → Return result (10ms)

Total: 3 requests in 30ms = 100 QPS
```

### With Dynamic Batching
```
Request 1 arrives at t=0ms
Request 2 arrives at t=1ms  } Wait up to max_queue_delay (5ms)
Request 3 arrives at t=3ms

At t=5ms: Process batch of 3 → Return all results (12ms total)

Total: 3 requests in 17ms = 176 QPS (76% improvement!)
```

### Configuration

In `triton/config.pbtxt`:

```protobuf
dynamic_batching {
  max_queue_delay_microseconds: 5000  # Wait up to 5ms
  preferred_batch_size: [4, 8, 16, 32]
  max_batch_size: 32
}
```

**Trade-offs:**
- ↑ `max_queue_delay` → ↑ throughput, ↑ latency
- ↓ `max_queue_delay` → ↓ latency, ↓ throughput

### Observing Batching

Run the load test to see batching in action:

```bash
python scripts/load_test.py --mode progressive
```

This will:
1. Start at 10 QPS (batch size ~1-2)
2. Increase to 50 QPS (batch size ~4-8)
3. Increase to 100 QPS (batch size ~8-16)
4. Increase to 200 QPS (batch size ~16-32)

Watch the `Batch Size Distribution` panel in Grafana to see batching efficiency!

## 📊 Monitoring

### View Metrics

**Prometheus:**
```bash
open http://localhost:9090

# Query examples
rate(inference_requests_total[1m])
histogram_quantile(0.95, rate(inference_latency_seconds_bucket[1m]))
histogram_quantile(0.50, rate(inference_batch_size_bucket[1m]))
```

**Grafana:**
```bash
open http://localhost:3000
# Login: admin / admin

# Import dashboard from monitoring/grafana/dashboards/inference-metrics.json
```

### Key Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| `inference_requests_total` | Total requests | Track QPS |
| `inference_latency_seconds` | Request latency | P50/P95/P99 latency |
| `inference_batch_size` | Batch sizes used | **KEY: Shows batching efficiency** |
| `routing_weights` | Traffic distribution | Verify A/B test splits |

## 🧪 Testing

### Unit Tests

```bash
pytest tests/unit/ -v
```

### Integration Tests

```bash
# Start services first
docker-compose up -d

# Run tests
pytest tests/integration/ -v
```

## 🎓 Learning Exercises

### Exercise 1: Tune Batching Parameters

Try different `max_queue_delay` values in `triton/config.pbtxt`:

```protobuf
# Low latency (sacrifice throughput)
max_queue_delay_microseconds: 1000  # 1ms

# High throughput (accept higher latency)
max_queue_delay_microseconds: 20000  # 20ms
```

Restart Triton and run load tests to observe the trade-off!

### Exercise 2: A/B Testing

Deploy two model versions and gradually shift traffic:

```bash
# Start: 100% v1
curl -X PUT http://localhost:8002/api/v1/config/routing \
  -d '{"weights": {"recommendation_v1": 1.0}}'

# Gradual rollout: 90% v1, 10% v2
curl -X PUT http://localhost:8002/api/v1/config/routing \
  -d '{"weights": {"recommendation_v1": 0.9, "recommendation_v2": 0.1}}'

# Full rollout: 100% v2
curl -X PUT http://localhost:8002/api/v1/config/routing \
  -d '{"weights": {"recommendation_v2": 1.0}}'
```

Watch traffic distribution in Grafana!

### Exercise 3: Stress Testing

Find the maximum QPS your system can handle:

```bash
# Test with increasing load
python scripts/load_test.py --qps 100 --duration 60
python scripts/load_test.py --qps 200 --duration 60
python scripts/load_test.py --qps 500 --duration 60

# Monitor:
# - P99 latency staying under threshold
# - No error rate increase
# - Batch sizes approaching max_batch_size
```

## 🛠️ Development

### Project Structure

```
.
├── control-plane/              # Control plane services
│   ├── model-registry/         # Model metadata + MLflow
│   ├── deployment-service/     # Model deployment orchestration
│   └── config-service/         # Routing config management
├── data-plane/
│   └── gateway/                # Inference gateway
├── triton/
│   └── models/                 # Triton model repository
├── k8s/                        # Kubernetes manifests
├── scripts/                    # Utility scripts
├── tests/                      # Unit and integration tests
└── monitoring/                 # Prometheus & Grafana config
```

### Adding a New Model

1. Train and export as TorchScript (.pt)
2. Create `config.pbtxt` with batching config
3. Upload to MinIO
4. Register via Model Registry API
5. Deploy via Deployment Service API
6. Update routing weights

## 🚢 Kubernetes Deployment

For k3s single-node deployment:

```bash
# Build Docker images
docker-compose build

# Deploy to k3s
cd k8s
./deploy-all.sh

# Port forward services
kubectl port-forward svc/gateway 8000:8000
kubectl port-forward svc/grafana 3000:3000
```

See [k8s/README.md](k8s/README.md) for details.

## 📚 Key Concepts Learned

### 1. Triton Inference Server
- **Model backends**: PyTorch, TensorFlow, ONNX, TensorRT
- **Dynamic batching**: Automatic request batching for GPU efficiency
- **gRPC protocol**: Binary protocol for efficient tensor communication
- **Model versioning**: Serve multiple versions simultaneously

### 2. Distributed Systems Patterns
- **Service discovery**: DNS-based discovery in Kubernetes
- **Config sync**: Polling-based config updates (push-based with etcd/Consul in production)
- **Health checks**: Liveness and readiness probes
- **Graceful degradation**: Continue serving with stale config if updates fail

### 3. ML Infrastructure
- **Model registry**: Centralized metadata + versioning (MLflow)
- **Model storage**: Artifact storage (S3/MinIO)
- **Traffic routing**: Weighted random for A/B testing
- **Observability**: Prometheus + Grafana for metrics

### 4. Production Patterns
- **Zero-downtime deployment**: Deploy new versions without stopping traffic
- **Atomic config updates**: Thread-safe configuration changes
- **Metrics-driven decisions**: Use P95/P99 latency for SLAs
- **Load testing**: Progressive load testing to find limits

## 🔮 Future Enhancements (Phase 2+)

- [ ] Feature store integration (Feast)
- [ ] A/B experiment framework with user assignment
- [ ] Model caching with Redis
- [ ] TensorRT optimization for GPU
- [ ] Multi-model ensemble serving
- [ ] Request batching in gateway
- [ ] gRPC gateway (in addition to HTTP)
- [ ] Canary deployments with automatic rollback
- [ ] Model performance monitoring (drift detection)

## 🐛 Troubleshooting

### Models not loading in Triton

```bash
# Check Triton logs
docker-compose logs triton

# Common issues:
# 1. config.pbtxt missing or malformed
# 2. Model file not at correct path: /models/{name}/1/model.pt
# 3. Input/output names mismatch
```

### Gateway can't connect to Triton

```bash
# Check Triton health
curl http://localhost:8000/v2/health/ready

# Check network
docker-compose exec gateway ping triton
```

### Config updates not taking effect

```bash
# Config polling is every 30s by default
# Check gateway logs for config sync:
docker-compose logs gateway | grep "Config updated"

# Force restart gateway to pick up changes immediately
docker-compose restart gateway
```

## 📖 Additional Resources

- [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [PyTorch Model Serving](https://pytorch.org/serve/)

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

This is a learning project. Feel free to fork and extend with your own experiments!

---

**Happy Learning! 🚀**

For questions or feedback, check the project's CLAUDE.md for implementation details.
