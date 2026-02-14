# High-Performance Model Multiplexing on Ray Serve

A hands-on learning lab for understanding how production model serving platforms work. This system uses Ray Serve's `@serve.multiplexed` to serve **35 model variants** through a single deployment pool with LRU eviction, autoscaling, and head/worker separation — all running on CPU.

```
35 models  →  3 fit in cache per replica  →  constant eviction  →  observable cold/warm latency
```

## Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │               Ray Cluster                   │
                         │                                              │
  curl/load_test ──────▶ │  ┌──────────┐     ┌─────────────────────┐   │
                         │  │ Ingress  │────▶│  MultiplexedWorker  │   │
                         │  │ (FastAPI)│     │  max_models=3/replica│   │
                         │  │ 0.5 CPU  │     │  autoscale 1→10     │   │
                         │  └──────────┘     │                     │   │
                         │       │           │  ┌───────────────┐  │   │
                         │   validates       │  │  LRU Cache    │  │   │
                         │   model_id        │  │  [model_A]    │  │   │
                         │   via registry    │  │  [model_B]    │  │   │
                         │                   │  │  [model_C]    │  │   │
                         │                   │  └───────────────┘  │   │
                         │                   └─────────────────────┘   │
                         │                            │                │
                         │                     ┌──────┴──────┐         │
                         │                     │ ModelStore  │         │
                         │                     │ (async I/O) │         │
                         │                     └──────┬──────┘         │
                         └────────────────────────────┼────────────────┘
                                                      │
                                               model_store/
                                            (35 .npy weight files)
```

**Key design decisions:**
- **Head/Worker separation** — Ingress (validation + routing) and Workers (inference) scale independently
- **LRU eviction** — Each worker replica caches only 3 models; the 4th request evicts the least-recently-used one
- **Simulated weight loading** — 500ms `asyncio.sleep` mimics S3 download, so cold-starts are visible in latency
- **Real computation** — Models hold numpy weight matrices (1–16 MB) and `predict()` does actual matrix multiplication

### Model Families

| Family | Variants | Weight Matrix | Memory Each |
|---|---|---|---|
| `text-classifier-v{0..9}` | 10 | 1024 x 1024 | ~4 MB |
| `embedding-model-v{0..9}` | 10 | 2048 x 2048 | ~16 MB |
| `summarizer-v{0..9}` | 10 | 1536 x 1536 | ~9 MB |
| `sentiment-analyzer-v{0..4}` | 5 | 512 x 512 | ~1 MB |

---

## Installation

```bash
# Clone and enter the project
cd labmodelcloud

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Model Artifacts

This creates 35 model weight files (~300 MB total) and a `manifest.json`:

```bash
python scripts/generate_models.py
```

```
  Generated text-classifier-v0: 1024x1024 (4.2 MB)
  Generated text-classifier-v1: 1024x1024 (4.2 MB)
  ...
  Generated sentiment-analyzer-v4: 512x512 (1.0 MB)

Generated 35 models in model_store
Manifest written to model_store/manifest.json
```

You can inspect the manifest:

```bash
cat model_store/manifest.json | python -m json.tool | head -20
```

---

## Running the Server

```bash
python scripts/run.py
```

You'll see:

```
Ray dashboard: http://127.0.0.1:8265
Serve app deployed at http://127.0.0.1:8000
  GET  /v1/models              — list all models
  GET  /v1/models/{model_id}   — model info
  POST /v1/predict/{model_id}  — run inference
  GET  /health                 — health check
```

Open the **Ray Dashboard** at [http://127.0.0.1:8265](http://127.0.0.1:8265) to see the cluster, deployments, and logs in real-time.

---

## Sending Traffic

### Health check

```bash
curl http://localhost:8000/health
```
```json
{"status": "healthy"}
```

### List all models

```bash
curl http://localhost:8000/v1/models | python -m json.tool
```
```json
[
  {"model_id": "text-classifier-v0", "family": "text-classifier", "matrix_size": 1024, "memory_mb": 4.2},
  {"model_id": "text-classifier-v1", "family": "text-classifier", "matrix_size": 1024, "memory_mb": 4.2},
  ...
]
```

### Get info for a specific model

```bash
curl http://localhost:8000/v1/models/embedding-model-v3
```
```json
{"model_id": "embedding-model-v3", "family": "embedding-model", "matrix_size": 2048, "memory_mb": 16.8}
```

### Run inference

```bash
curl -X POST http://localhost:8000/v1/predict/text-classifier-v0 \
  -H "Content-Type: application/json" \
  -d '{"input": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'
```
```json
{
  "model_id": "text-classifier-v0",
  "shape": [1, 1024],
  "predictions": [[0.234, -1.567, ...]]
}
```

The input is automatically padded/truncated to match the model's weight matrix dimensions. A `text-classifier` has a 1024x1024 matrix, so a 5-element input gets zero-padded to 1024 and the output has 1024 values.

### Request a model that doesn't exist

```bash
curl -X POST http://localhost:8000/v1/predict/nonexistent-model \
  -H "Content-Type: application/json" \
  -d '{"input": [[1.0]]}'
```
```json
{"detail": "Model 'nonexistent-model' not found in registry"}
```

Returns HTTP 404.

---

## Observing Cold-Start vs Cached Latency

This is the core thing to learn from this lab. Each worker replica can only hold 3 models in memory.

### Manual experiment: force an eviction

```bash
# Request 3 different models — they all get loaded (cold start, ~500ms each)
curl -s -o /dev/null -w "model-v0: %{time_total}s\n" \
  -X POST http://localhost:8000/v1/predict/text-classifier-v0 \
  -H "Content-Type: application/json" -d '{"input":[[1]]}'

curl -s -o /dev/null -w "model-v1: %{time_total}s\n" \
  -X POST http://localhost:8000/v1/predict/text-classifier-v1 \
  -H "Content-Type: application/json" -d '{"input":[[1]]}'

curl -s -o /dev/null -w "model-v2: %{time_total}s\n" \
  -X POST http://localhost:8000/v1/predict/text-classifier-v2 \
  -H "Content-Type: application/json" -d '{"input":[[1]]}'

# Now request a 4th model — this evicts the LRU (v0) and loads v3
curl -s -o /dev/null -w "model-v3 (evicts v0): %{time_total}s\n" \
  -X POST http://localhost:8000/v1/predict/text-classifier-v3 \
  -H "Content-Type: application/json" -d '{"input":[[1]]}'

# Request v1 again — it's still cached, so this should be fast
curl -s -o /dev/null -w "model-v1 (cached): %{time_total}s\n" \
  -X POST http://localhost:8000/v1/predict/text-classifier-v1 \
  -H "Content-Type: application/json" -d '{"input":[[1]]}'

# Request v0 again — it was evicted, so this is a cold start
curl -s -o /dev/null -w "model-v0 (evicted, reload): %{time_total}s\n" \
  -X POST http://localhost:8000/v1/predict/text-classifier-v0 \
  -H "Content-Type: application/json" -d '{"input":[[1]]}'
```

**Expected output:**

```
model-v0: 0.550s          ← cold start
model-v1: 0.540s          ← cold start
model-v2: 0.530s          ← cold start
model-v3 (evicts v0): 0.545s  ← cold start (v0 evicted)
model-v1 (cached): 0.015s     ← cache hit!
model-v0 (evicted, reload): 0.535s  ← cold start (had to reload)
```

### Automated load test

The load test sends 100 random requests across all 35 models with concurrency of 10:

```bash
python scripts/load_test.py
```

```
Load test: 100 requests, concurrency=10
Targeting 35 models

Results (100 requests):
  Success: 100/100
  Errors:  0

Latency (seconds):
  Min:    0.012
  Max:    2.341
  Mean:   0.587
  Median: 0.542
  P95:    1.203
  P99:    2.105

Cache analysis:
  Fast (<100ms, likely cached): 14
  Slow (>=100ms, likely cold):  86
```

The cache hit ratio is intentionally low (~14%) because we have 35 models but only 3 slots per replica. This is by design — in production, you'd tune `MAX_MODELS_PER_REPLICA` based on your memory budget.

---

## Viewing Metrics

Ray Serve exposes Prometheus metrics on port 8080 by default.

### Raw metrics

```bash
curl -s http://localhost:8080/metrics | grep -E "model_cache_miss|inference_request_latency|model_load_duration"
```

You'll see counters and histograms like:

```
# HELP model_cache_miss_total Total number of model cache misses requiring weight loading.
model_cache_miss_total{model_id="text-classifier-v0"} 2.0
model_cache_miss_total{model_id="text-classifier-v1"} 1.0
model_cache_miss_total{model_id="embedding-model-v3"} 1.0

# HELP model_load_duration_seconds Time to load model weights from store in seconds.
model_load_duration_seconds_bucket{model_id="text-classifier-v0",le="0.5"} 0.0
model_load_duration_seconds_bucket{model_id="text-classifier-v0",le="1.0"} 2.0
```

### Our custom metrics

| Metric | Type | What it tells you |
|---|---|---|
| `model_cache_miss_total` | Counter | How many times each model had to be loaded from disk (per model_id) |
| `inference_request_latency_seconds` | Histogram | End-to-end request time including routing + inference (per model_id) |
| `model_load_duration_seconds` | Histogram | Weight loading time only — the "cold start cost" (per model_id) |

### Ray Serve built-in metrics

Ray Serve also exposes its own metrics automatically:

```bash
# Number of requests in progress per deployment
curl -s http://localhost:8080/metrics | grep "ray_serve_num_ongoing_requests"

# Request latency by deployment
curl -s http://localhost:8080/metrics | grep "ray_serve_request_latency"

# Number of replicas per deployment
curl -s http://localhost:8080/metrics | grep "ray_serve_deployment_replica"
```

### Prometheus + Grafana (optional)

If you have Docker, you can run Prometheus to scrape and graph these metrics:

```bash
docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

Then open [http://localhost:9090](http://localhost:9090) and query:

- `rate(model_cache_miss_total[1m])` — cache miss rate per second
- `histogram_quantile(0.95, rate(inference_request_latency_seconds_bucket[5m]))` — p95 latency
- `histogram_quantile(0.5, rate(model_load_duration_seconds_bucket[5m]))` — median cold-start time

---

## Experiments

### Experiment 1: Kill a Worker and Watch Recovery

Ray Serve automatically restarts crashed replicas. Let's see it in action.

**Step 1:** In a separate terminal, start tailing the Ray Serve logs:

```bash
# Ray logs are in /tmp/ray/session_latest/logs/
# Watch for worker lifecycle events:
tail -f /tmp/ray/session_latest/logs/serve/replica_*MultiplexedModelWorker*.log
```

**Step 2:** Find the worker actor's PID:

```bash
# List Ray actors
python -c "
import ray
ray.init(address='auto')
for actor in ray.util.list_named_actors(all_namespaces=True):
    print(actor)
"
```

Or find the PID from the Ray Dashboard at [http://127.0.0.1:8265/#/actors](http://127.0.0.1:8265/#/actors) — look for the `MultiplexedModelWorker` actor and note its PID.

**Step 3:** Kill the worker process:

```bash
# Replace <PID> with the actual PID from the dashboard
kill -9 <PID>
```

**Step 4:** Observe recovery. In the Ray Dashboard you'll see:

1. The replica briefly shows as **UNHEALTHY** or disappears
2. Ray Serve detects the failure via its health check (runs every 10s)
3. A **new replica** is automatically spawned
4. The new replica starts accepting requests (models load on demand again — cold cache)

**Step 5:** Send a request to confirm recovery:

```bash
# This may take a moment while the new replica initializes
curl -w "\nTotal time: %{time_total}s\n" \
  -X POST http://localhost:8000/v1/predict/text-classifier-v0 \
  -H "Content-Type: application/json" -d '{"input":[[1,2,3]]}'
```

The first request after recovery will be slow (cold start). Subsequent requests to cached models will be fast again.

### Experiment 2: Trigger Autoscaling Under Load

The worker is configured to scale up when ongoing requests exceed 3 per replica.

**Step 1:** Watch replica count in the Ray Dashboard at [http://127.0.0.1:8265/#/serve](http://127.0.0.1:8265/#/serve)

**Step 2:** Run the load test with high concurrency:

```bash
python scripts/load_test.py
```

**Step 3:** While the load test runs, refresh the Ray Dashboard. You should see:

- `MultiplexedModelWorker` replicas increasing from 1 toward 2-3+
- Each new replica starts with an empty cache
- After load subsides, replicas scale back down after 60s (`DOWNSCALE_DELAY_S`)

You can also check replica count via the API:

```bash
python -c "
import ray
from ray import serve
ray.init(address='auto')
for app in serve.status().applications.values():
    for name, depl in app.deployments.items():
        print(f'{name}: {depl.status}, replicas target={depl.replica_states}')
"
```

### Experiment 3: Change the Cache Size and Compare

See what happens when you allow more (or fewer) models per replica.

```bash
# Default: 3 models per replica (high eviction rate)
python scripts/run.py
python scripts/load_test.py
# Note the cache hit ratio

# Stop the server (Ctrl+C), then restart with a larger cache:
MAX_MODELS_PER_REPLICA=10 python scripts/run.py
python scripts/load_test.py
# Cache hit ratio should be much higher, cold starts fewer

# Try with cache size of 1 — maximum eviction pressure:
MAX_MODELS_PER_REPLICA=1 python scripts/run.py
python scripts/load_test.py
# Almost every request is a cold start
```

### Experiment 4: Observe Health Checks After a Bad Model

If a model fails to load, the worker marks itself unhealthy and Ray restarts it.

```bash
# Create a corrupted model file
mkdir -p model_store/broken-model-v0
echo "not a valid numpy file" > model_store/broken-model-v0/weights.npy
```

Add it to the manifest so the registry accepts it:

```bash
python -c "
import json
with open('model_store/manifest.json') as f:
    m = json.load(f)
m['models'].append({'model_id': 'broken-model-v0', 'family': 'broken', 'matrix_size': 0, 'memory_mb': 0})
with open('model_store/manifest.json', 'w') as f:
    json.dump(m, f, indent=2)
"
```

Restart the server and request the broken model:

```bash
python scripts/run.py  # in one terminal

curl -X POST http://localhost:8000/v1/predict/broken-model-v0 \
  -H "Content-Type: application/json" -d '{"input":[[1]]}'
```

Watch the logs — the worker catches the exception, sets `self._healthy = False`, and Ray's health check will detect this and restart the replica. In the Ray Dashboard, you'll see the replica briefly go unhealthy then get replaced.

Clean up afterward:

```bash
rm -rf model_store/broken-model-v0
# Re-run generate_models.py to restore the manifest
python scripts/generate_models.py
```

---

## Adding a New Model to the System

Adding a model is a two-step process: create the weight artifact, then update the manifest.

### Option A: Add manually

```bash
# 1. Create the weight file
python -c "
import numpy as np
from pathlib import Path
model_id = 'my-custom-model-v0'
size = 768  # weight matrix dimensions
model_dir = Path('model_store') / model_id
model_dir.mkdir(parents=True, exist_ok=True)
weights = np.random.default_rng(0).standard_normal((size, size)).astype(np.float32)
np.save(model_dir / 'weights.npy', weights)
print(f'Created {model_id}: {size}x{size} ({weights.nbytes/1e6:.1f} MB)')
"

# 2. Add it to the manifest
python -c "
import json
with open('model_store/manifest.json') as f:
    manifest = json.load(f)
manifest['models'].append({
    'model_id': 'my-custom-model-v0',
    'family': 'my-custom-model',
    'matrix_size': 768,
    'memory_mb': 2.4
})
with open('model_store/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'Manifest now has {len(manifest[\"models\"])} models')
"

# 3. Restart the server to pick up the new manifest
#    (The registry loads the manifest at startup)
python scripts/run.py

# 4. Verify
curl http://localhost:8000/v1/models/my-custom-model-v0
curl -X POST http://localhost:8000/v1/predict/my-custom-model-v0 \
  -H "Content-Type: application/json" -d '{"input":[[1,2,3]]}'
```

### Option B: Add a new model family in config, then regenerate

Edit `src/config.py` and add a new entry to `MODEL_FAMILIES`:

```python
MODEL_FAMILIES = {
    "text-classifier": (1024, 10),
    "embedding-model": (2048, 10),
    "summarizer": (1536, 10),
    "sentiment-analyzer": (512, 5),
    "my-new-family": (768, 5),       # ← add this line
}
```

Then regenerate all models:

```bash
python scripts/generate_models.py  # now generates 40 models
python scripts/run.py              # restart to pick up new manifest
```

---

## Removing a Model from the System

### Remove a single model

```bash
# 1. Remove from manifest
python -c "
import json
with open('model_store/manifest.json') as f:
    manifest = json.load(f)
manifest['models'] = [m for m in manifest['models'] if m['model_id'] != 'sentiment-analyzer-v4']
with open('model_store/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'Manifest now has {len(manifest[\"models\"])} models')
"

# 2. Optionally delete the weight file to free disk space
rm -rf model_store/sentiment-analyzer-v4

# 3. Restart the server
python scripts/run.py

# 4. Verify it's gone
curl http://localhost:8000/v1/models/sentiment-analyzer-v4
# → 404: "Model 'sentiment-analyzer-v4' not found in registry"
```

### Remove an entire model family

Edit `src/config.py` to remove the family from `MODEL_FAMILIES`, then regenerate:

```bash
python scripts/generate_models.py
python scripts/run.py
```

Note: `generate_models.py` creates only the families listed in config. Old model directories on disk won't be deleted automatically — remove them manually if you want to reclaim space:

```bash
rm -rf model_store/sentiment-analyzer-v*
```

---

## Configuration Reference

All settings are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_STORE_PATH` | `./model_store` | Path to model weight artifacts |
| `MAX_MODELS_PER_REPLICA` | `3` | LRU cache size per worker replica |
| `MAX_ONGOING_REQUESTS` | `8` | Max concurrent requests per replica before backpressure |
| `TARGET_ONGOING_REQUESTS` | `3` | Autoscaler target — scales up when exceeded |
| `MIN_REPLICAS` | `1` | Minimum worker replicas |
| `MAX_REPLICAS` | `10` | Maximum worker replicas |
| `UPSCALE_DELAY_S` | `10` | Seconds to wait before adding a replica |
| `DOWNSCALE_DELAY_S` | `60` | Seconds to wait before removing a replica |
| `SIMULATED_DOWNLOAD_LATENCY` | `0.5` | Simulated S3 download time in seconds |
| `WORKER_NUM_CPUS` | `1` | CPUs allocated per worker replica |
| `INGRESS_NUM_CPUS` | `0.5` | CPUs allocated for the ingress |

Example: run with fast loading and larger cache:

```bash
SIMULATED_DOWNLOAD_LATENCY=0.05 MAX_MODELS_PER_REPLICA=10 python scripts/run.py
```

---

## Running Tests

```bash
# Unit tests (no Ray cluster needed)
pytest tests/test_registry.py tests/test_store.py -v

# End-to-end tests (starts a local Ray cluster automatically)
pytest tests/test_e2e.py -v

# All tests
pytest tests/ -v
```

---

## Project Structure

```
labmodelcloud/
├── src/
│   ├── config.py               # Central config (env vars, model families)
│   ├── errors.py               # ModelNotFoundError, CudaOutOfMemoryError
│   ├── app.py                  # Deployment graph: ingress → worker
│   ├── models/
│   │   ├── base.py             # SimulatedModel (numpy weights + predict)
│   │   ├── registry.py         # ModelRegistry (validates model IDs)
│   │   └── store.py            # ModelStore (async weight loading)
│   ├── serving/
│   │   ├── worker.py           # @serve.multiplexed worker with LRU
│   │   └── ingress.py          # FastAPI ingress (routing + validation)
│   └── metrics/
│       └── collectors.py       # Prometheus counters and histograms
├── scripts/
│   ├── generate_models.py      # Generate 35 model weight files
│   ├── run.py                  # Start Ray Serve
│   └── load_test.py            # Concurrent load test
├── tests/
│   ├── conftest.py             # Shared test fixtures
│   ├── test_registry.py        # Registry unit tests
│   ├── test_store.py           # Store unit tests
│   └── test_e2e.py             # Full integration tests
├── model_store/                # Generated artifacts (gitignored)
├── monitoring/
│   └── prometheus.yml          # Prometheus scrape config
├── requirements.txt
└── CLAUDE.md                   # Architecture decisions and LLM instructions
```
