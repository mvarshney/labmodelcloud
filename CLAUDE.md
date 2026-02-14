High-Performance Model Multiplexing on Ray Serve
1. Project Purpose
The goal is to build a professional-grade model serving infrastructure using Ray Serve. This system must support a "Model Family" architecture where 30+ large model variants (e.g., could be LoRA adapters or entirely different model architectures and/or weights) are served efficiently across a distributed cluster.

We are building this as a hands on learning lab to deeply undertand the ML Serving platform and the nuances that only show up when one actually runs such system. Our scope is Model Serving Infra as used by big tech companies. 

We are using Ray as a free and easy to use platform for building Model Serving platform. Learning Ray or Ray Serve is not our goal.

The core constraint is VRAM Efficiency: we cannot load all 30 models simultaneously. We must use dynamic loading and intelligent eviction to maximize GPU utilization while minimizing cold-start latency.

2. Technical Goals & Requirements
Model Multiplexing: Use a single deployment pool to serve 30+ distinct model IDs.

Fractional GPU Allocation: Support bin-packing multiple replicas or adapters on a single GPU (e.g., num_gpus: 0.5).

LRU Weight Caching: Implement a custom logic to swap model weights in VRAM, evicting the Least Recently Used (LRU) variant when memory is full.

Zero-Downtime Scaling: Use Ray Serve’s autoscaling to handle bursty traffic without dropping requests.

Head/Worker Architecture: Separate the Ingress (FastAPI) from the Inference Workers for better lifecycle management.

3. Implementation Design
A. The "Multiplexed" Inference Logic
We will use @serve.multiplexed to allow a single replica to represent many model IDs.

The Cache: A dictionary-based cache within the actor that stores loaded model objects.

The Fetcher: An asynchronous method to pull weights from remote storage (S3/GCS) or a shared local NVMe mount.

B. Resource Constraints
To simulate professional infrastructure, we will enforce:

max_ongoing_requests: To prevent head-of-line blocking during weight swaps.

target_ongoing_requests: For autoscaling triggers.

ray_actor_options: Strict memory limits to trigger Ray’s object spilling.

C. The Ingress Layer
A FastAPI app that acts as the "Traffic Controller." It will:

Validate the model_id against a registry.

Route the request to the MultiplexedWorker using a DeploymentHandle.

Track metrics (request latency, cache hit/miss ratio).

4. Proposed File Structure
<To be decided later>

5. Instructions for the LLM (Claude/GPT)
When generating the code for this project, please adhere to these design patterns:

Ensure the code can run on CPU. The testing environment is Ubuntu VM where we can deploy Ray, k8s etc, but it does not have gpus.

Concurrency: Use async/await for the weight loading process to ensure the event loop isn't blocked while waiting for I/O.

Error Handling: Implement robust handling for ModelNotFound and CudaOutOfMemory errors. If a model fails to load, the worker should report health as "unhealthy" so Ray can restart it.

The Multiplexed Decorator: Specifically use:

Python

@serve.multiplexed(max_num_models_per_replica=3)
async def get_model(self, model_id: str):
    # Implementation here