ML Model Serving Platform
======================================
Project Overview
-----------------------------
Goals
~~~~~~~~
Build a production-grade ML model serving platform for recommendation models with the following capabilities:

* Serve PyTorch models on GPUs with high throughput and low latency
* Support multiple model variants with configurable traffic routing
* Provide control plane for model lifecycle management
* Enable dynamic batching for GPU efficiency
* Learn modern ML infrastructure patterns and distributed systems concepts

Non-Goals (What We Won't Build in Phase 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
❌ Feature fetching and caching (deferred to Phase 2)
❌ A/B experiment framework with user assignment (deferred to Phase 2)
❌ GPU placement optimization (deferred to Phase 3)
❌ TensorRT optimization (deferred to Phase 3)
❌ Multi-datacenter orchestration
❌ Advanced deployment strategies (blue-green, canary)
❌ Model training pipelines
❌ Online learning / model updates from live traffic

Success Criteria
~~~~~~~~~~~~~~~~~
Functional:

Register and deploy 2 models via control plane API
Route traffic between models based on configurable weights
Update routing weights dynamically without restarts
Achieve 1000+ QPS with acceptable latency

Performance:

P99 latency < 50ms for inference
Demonstrate batching efficiency (1x → 10x+ throughput improvement)
GPU utilization > 70%

Operational:

Zero-downtime model deployment
Observable via metrics (Prometheus)
Deployable via docker-compose (dev) and K8s (prod)


System Architecture
---------------------------------
High-Level Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~
┌─────────────────────────────────────────────────────────┐
│                   Control Plane                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Model      │  │  Deployment  │  │    Config    │  │
│  │  Registry    │  │   Service    │  │   Service    │  │
│  │  (MLflow)    │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          │ gRPC / REST
                          │
┌─────────────────────────────────────────────────────────┐
│                      Data Plane                          │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │           Python Gateway (FastAPI)             │     │
│  │  ┌──────────────┐  ┌──────────────┐           │     │
│  │  │   Router     │  │ Config Client│           │     │
│  │  │ (Weighted    │  │ (polls CP)   │           │     │
│  │  │  Random)     │  │              │           │     │
│  │  └──────────────┘  └──────────────┘           │     │
│  └────────────────────────────────────────────────┘     │
│                          │                               │
│                          │ gRPC                          │
│                          ▼                               │
│  ┌────────────────────────────────────────────────┐     │
│  │         Triton Inference Server                │     │
│  │  ┌──────────────┐  ┌──────────────┐           │     │
│  │  │  Model V1    │  │  Model V2    │           │     │
│  │  │  (GPU 0)     │  │  (GPU 0)     │           │     │
│  │  │              │  │              │           │     │
│  │  │ Dynamic      │  │ Dynamic      │           │     │
│  │  │ Batching     │  │ Batching     │           │     │
│  │  └──────────────┘  └──────────────┘           │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
Component Responsibilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Control Plane:

Model Registry Service: Manages model metadata, integrates with MLflow
Deployment Service: Orchestrates model deployment to Triton instances
Config Service: Stores and serves routing configuration

Data Plane:

Python Gateway: HTTP/gRPC API, routing, config polling
Triton Inference Server: GPU model serving with dynamic batching


Technology Choices
~~~~~~~~~~~~~~~~~~~~~~~~~
Core Stack

Language (Gateway): Python 3.11+ with asyncio
Language (Control Plane): Python 3.11+ (FastAPI services)
Model Serving: NVIDIA Triton Inference Server 2.40+
Model Format: PyTorch JIT (.pt files)
Model Registry: MLflow 2.10+
API Framework: FastAPI 0.109+
RPC Protocol: gRPC (Triton communication), REST (control plane APIs)
Container Orchestration: Kubernetes 1.28+ (prod), docker-compose (dev)
Metrics: Prometheus + Grafana
Storage: S3/MinIO for model artifacts

Why These Choices?
~~~~~~~~~~~~~~~~~~~~~~~~~~
# Python for Gateway:

Natural integration with ML ecosystem
Async/await for I/O-bound operations
Interview relevance for AI infra roles
Fast iteration for learning

# Triton for Inference:

Industry standard (NVIDIA official)
Dynamic batching out-of-the-box
Multi-backend support (PyTorch, ONNX, TensorRT)
Production-proven at scale

# gRPC for Triton Communication:

Binary protocol (efficient for tensors)
HTTP/2 multiplexing
Strong typing via protobuf

# K8s Service Discovery:

Learn real distributed systems patterns
DNS-based discovery (simple, robust)
Foundation for multi-datacenter later


# High-level gateway structure
class InferenceGateway:
    def __init__(self):
        self.triton_client = TritonGRPCClient()
        self.router = WeightedRouter()
        self.config_client = ControlPlaneClient()
        self.metrics = PrometheusMetrics()
    
    async def predict(self, request: PredictRequest) -> PredictResponse:
        # 1. Select model based on routing config
        model_name = await self.router.select_model()
        
        # 2. Send to Triton (batching handled by Triton)
        result = await self.triton_client.infer(model_name, request.tensor)
        
        # 3. Record metrics
        self.metrics.record_latency(model_name, latency)
        
        return PredictResponse(score=result)
    
    async def sync_config(self):
        # Poll control plane every 30s for routing config
        while True:
            config = await self.config_client.get_routing_config()
            self.router.update_weights(config)
            await asyncio.sleep(30)


### Request Flow
1. Client → POST /predict
   Body: {"tensor": [[0.1, 0.2, ..., 0.5]]}  # Shape: [batch_size, features]

2. Gateway:
   - Validates input schema
   - Selects model: weighted_random(models, weights)
   - Forwards to Triton via gRPC

3. Triton:
   - Adds request to dynamic batch queue
   - Waits up to max_batch_delay (e.g., 10ms)
   - Accumulates up to max_batch_size (e.g., 32)
   - Executes batch inference on GPU
   - Returns individual results

4. Gateway:
   - Extracts result for this request
   - Records metrics (latency, model_name)
   - Returns response

5. Client ← Response
   Body: {"score": 0.87}


### Triton Integration

**Model Repository Structure:**
```
/models/
├── recommendation_v1/
│   ├── 1/                    # Version 1
│   │   └── model.pt          # PyTorch JIT model
│   └── config.pbtxt          # Triton config
└── recommendation_v2/
    ├── 1/
    │   └── model.pt
    └── config.pbtxt



Model Registry Service
============================
Responsibilities:

Register new models with metadata
Store model information in MLflow
Track model versions and lineage
Provide model discovery API


Deployment Service
==========================
Responsibilities:

Download models from S3 to Triton model repository
Load models into Triton via Management API
Run smoke tests to validate deployment
Update routing configuration after successful deployment
Unload models from Triton