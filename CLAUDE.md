# High-Performance Model Multiplexing on Ray Serve

## 1. Project Purpose

A hands-on learning lab for understanding production model serving infrastructure. The system uses Ray Serve's `@serve.multiplexed` to serve 35 model variants through a single deployment pool with LRU eviction, autoscaling, and head/worker separation.

**Scope:** Model Serving Infra as used by big tech companies. Ray is the platform, but learning Ray itself is not the goal — understanding the serving patterns (multiplexing, eviction, cold-start, autoscaling) is.

**Core constraint:** Memory efficiency — we cannot load all 35 models simultaneously. Dynamic loading with LRU eviction maximizes utilization while minimizing cold-start latency.

## 2. Architecture

```
curl → Ingress (FastAPI, 0.5 CPU) → MultiplexedModelWorker (1 CPU, autoscale 1→10)
                                          ↓
                                    LRU Cache (3 models/replica)
                                          ↓
                                    ModelStore (async I/O) → model_store/*.npy
```

- **Ingress** (`src/serving/ingress.py`): Validates model_id against registry, routes via `DeploymentHandle.options(multiplexed_model_id=...)`, tracks latency metrics
- **Worker** (`src/serving/worker.py`): `@serve.multiplexed(max_num_models_per_replica=3)`, loads/evicts models, reports health
- **ModelStore** (`src/models/store.py`): Async weight fetcher with simulated S3 latency (500ms) and threaded numpy deserialization
- **ModelRegistry** (`src/models/registry.py`): Validates model IDs against `manifest.json`
- **SimulatedModel** (`src/models/base.py`): Numpy weight matrix with real matrix-multiply `predict()`

## 3. File Structure

```
src/
├── config.py                  # Central config (all env-var driven)
├── errors.py                  # ModelNotFoundError, CudaOutOfMemoryError
├── app.py                     # Deployment graph: ingress → worker with autoscaling
├── models/
│   ├── base.py                # SimulatedModel (numpy weights + predict)
│   ├── registry.py            # ModelRegistry (manifest-based validation)
│   └── store.py               # ModelStore (async weight loading)
├── serving/
│   ├── worker.py              # MultiplexedModelWorker (@serve.multiplexed)
│   └── ingress.py             # FastAPI ingress (@serve.ingress)
└── metrics/
    └── collectors.py          # Prometheus counters/histograms via ray.serve.metrics

scripts/
├── generate_models.py         # Generate 35 model weight files + manifest.json
├── run.py                     # Start Ray + deploy Serve app
└── load_test.py               # Concurrent request load test (100 reqs, 10 concurrency)

tests/
├── conftest.py                # Shared fixtures (tmp_model_store)
├── test_registry.py           # Registry unit tests
├── test_store.py              # Store async unit tests
└── test_e2e.py                # Full integration tests (deploys Ray Serve)

model_store/                   # Generated artifacts (gitignored)
monitoring/prometheus.yml      # Prometheus scrape config
```

## 4. Model Families

| Family | Variants | Weight Matrix | Memory Each |
|---|---|---|---|
| `text-classifier-v{0..9}` | 10 | 1024x1024 | ~4 MB |
| `embedding-model-v{0..9}` | 10 | 2048x2048 | ~16 MB |
| `summarizer-v{0..9}` | 10 | 1536x1536 | ~9 MB |
| `sentiment-analyzer-v{0..4}` | 5 | 512x512 | ~1 MB |

Defined in `src/config.py:MODEL_FAMILIES`. To add/remove families, edit this dict and re-run `scripts/generate_models.py`.

## 5. Key Design Patterns

### Multiplexed Decorator
```python
@serve.multiplexed(max_num_models_per_replica=3)
async def get_model(self, model_id: str) -> SimulatedModel:
    # Called on cache miss — loads from store, Ray handles LRU eviction
```

### Async Weight Loading
- `asyncio.sleep(0.5)` simulates S3 download latency
- `loop.run_in_executor()` for CPU-bound numpy deserialization
- Event loop never blocked during I/O

### Error Handling
- `ModelNotFoundError` → HTTP 404 at ingress
- `CudaOutOfMemoryError` → worker sets `self._healthy = False` → Ray health check detects → replica restarted
- Corrupt weight files → caught, worker goes unhealthy, Ray auto-recovers

### Metrics (via `ray.serve.metrics`)
- `model_cache_miss_total` (Counter, per model_id)
- `inference_request_latency_seconds` (Histogram, per model_id)
- `model_load_duration_seconds` (Histogram, per model_id)
- Exposed at `:8080/metrics` in Prometheus format

### Autoscaling
```python
autoscaling_config={
    "target_ongoing_requests": 3,    # scale up threshold
    "min_replicas": 1,
    "max_replicas": 10,
    "upscale_delay_s": 10,
    "downscale_delay_s": 60,
}
max_ongoing_requests=8               # backpressure limit per replica
```

## 6. Instructions for the LLM

When modifying or extending this project:

- **CPU only.** The environment does not have GPUs. All code must run on CPU. Models use numpy matrices, not PyTorch/TensorFlow.
- **Async/await** for all I/O-bound operations (weight loading, network calls). Use `run_in_executor` for CPU-bound work to avoid blocking the event loop.
- **Error handling:** If a model fails to load, the worker must mark itself unhealthy (`self._healthy = False`) so Ray's health check restarts it. Never silently swallow load errors.
- **Configuration via env vars.** All tunable parameters live in `src/config.py` and are read from environment variables with sensible defaults. Don't hardcode values elsewhere.
- **Python 3.9 compatibility.** Use `from __future__ import annotations` in any file that uses `X | Y` union types or `list[X]` in function signatures.
- **Manifest-driven registry.** The source of truth for available models is `model_store/manifest.json`. Adding a model means creating its weight file AND adding it to the manifest. The registry loads the manifest at startup.
- **Keep models simulated.** The point of this lab is the serving infrastructure, not the models themselves. Models are numpy weight matrices with `predict()` doing matrix multiplication. Don't introduce real ML frameworks unless explicitly asked.

## 7. Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_models.py    # 35 models, ~300 MB
python scripts/run.py                # Ray dashboard :8265, API :8000
curl -X POST localhost:8000/v1/predict/text-classifier-v0 \
  -H "Content-Type: application/json" -d '{"input":[[1,2,3]]}'
```

See `README.md` for detailed usage, experiments, and metrics guide.
