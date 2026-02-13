"""
Triton gRPC client for inference.

This is the CORE component for learning Triton integration and dynamic batching!

KEY CONCEPTS:
=============

1. Dynamic Batching (Triton's killer feature):
   - When you send a single request to Triton, it doesn't process it immediately
   - Instead, Triton waits up to `max_queue_delay` (e.g., 5ms) for more requests
   - Once the delay expires OR batch reaches `max_batch_size`, Triton executes
   - This dramatically improves throughput without changing client code!

2. Why gRPC?
   - Binary protocol (efficient for tensors)
   - HTTP/2 multiplexing (multiple requests on one connection)
   - Strong typing via protobuf
   - Industry standard for ML serving

3. Throughput vs. Latency Trade-off:
   - Single request: Low throughput, lowest latency
   - Dynamic batching: High throughput, slightly higher latency (+max_queue_delay)
   - Example: 1 req/s = 1 QPS, but 10 reqs batched = ~10 QPS with same GPU time

4. How to observe batching:
   - Check Prometheus metrics: inference_batch_size histogram
   - At low QPS: batch_size ≈ 1 (requests arrive slowly)
   - At high QPS: batch_size → max_batch_size (requests accumulate quickly)
"""

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import numpy as np
from typing import List, Tuple
import time


class TritonInferenceClient:
    """
    Triton gRPC client for inference requests.

    Handles communication with Triton via gRPC protocol.
    """

    def __init__(self, triton_url: str = "triton:8001"):
        """
        Initialize Triton gRPC client.

        Args:
            triton_url: Triton server URL (host:port for gRPC, default is 8001)
        """
        self.triton_url = triton_url
        self.client = grpcclient.InferenceServerClient(
            url=triton_url,
            verbose=False
        )

    def is_server_ready(self) -> bool:
        """Check if Triton server is ready."""
        try:
            return self.client.is_server_ready()
        except Exception as e:
            print(f"Triton server not ready: {e}")
            return False

    async def infer(
        self,
        model_name: str,
        user_ids: List[int],
        item_ids: List[int],
        model_version: str = "1"
    ) -> Tuple[List[float], int]:
        """
        Send inference request to Triton.

        IMPORTANT: This method sends a SINGLE request. Dynamic batching happens
        on the Triton side, transparent to this client!

        How batching works:
        1. This client sends request with N examples (e.g., 4 user-item pairs)
        2. Triton receives it and adds to queue
        3. Triton waits up to max_queue_delay for MORE requests from OTHER clients
        4. When timeout expires or queue fills, Triton processes ALL requests together
        5. Each client gets back their own results

        Example timeline:
        - t=0ms:  Client A sends request with 2 examples
        - t=1ms:  Client B sends request with 3 examples
        - t=2ms:  Client C sends request with 5 examples
        - t=5ms:  Timeout expires, Triton processes batch of 10 examples (2+3+5)
        - t=15ms: Triton returns 2 results to A, 3 to B, 5 to C

        Args:
            model_name: Model name in Triton
            user_ids: List of user IDs
            item_ids: List of item IDs
            model_version: Model version

        Returns:
            Tuple of (scores, batch_size_used)
        """
        try:
            # Convert to numpy arrays (Triton expects numpy)
            user_ids_np = np.array(user_ids, dtype=np.int64)
            item_ids_np = np.array(item_ids, dtype=np.int64)

            request_batch_size = len(user_ids)

            # Create input tensors
            # InferInput wraps numpy arrays with metadata (name, shape, datatype)
            inputs = [
                grpcclient.InferInput(
                    "user_ids",
                    user_ids_np.shape,
                    "INT64"
                ),
                grpcclient.InferInput(
                    "item_ids",
                    item_ids_np.shape,
                    "INT64"
                ),
            ]

            # Set the actual data
            inputs[0].set_data_from_numpy(user_ids_np)
            inputs[1].set_data_from_numpy(item_ids_np)

            # Specify which outputs we want
            outputs = [grpcclient.InferRequestedOutput("scores")]

            # Send inference request via gRPC
            # This is where dynamic batching happens (on Triton side)!
            start_time = time.time()
            result = self.client.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                outputs=outputs
            )
            latency_ms = (time.time() - start_time) * 1000

            # Extract output
            scores_np = result.as_numpy("scores")
            scores = scores_np.squeeze().tolist()

            # Ensure scores is a list
            if isinstance(scores, float):
                scores = [scores]

            print(f"[Triton Inference] model={model_name}, "
                  f"request_size={request_batch_size}, "
                  f"latency={latency_ms:.2f}ms")

            # Note: We return request_batch_size, not the actual batch size Triton used
            # Triton's internal batch size is opaque to clients (by design)
            return scores, request_batch_size

        except InferenceServerException as e:
            print(f"Inference failed: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during inference: {e}")
            raise

    def get_model_metadata(self, model_name: str, model_version: str = "1") -> dict:
        """
        Get model metadata from Triton.

        Useful for debugging and understanding model inputs/outputs.

        Args:
            model_name: Model name
            model_version: Model version

        Returns:
            Model metadata dictionary
        """
        try:
            metadata = self.client.get_model_metadata(model_name, model_version)
            return {
                "name": metadata.name,
                "versions": metadata.versions,
                "platform": metadata.platform,
                "inputs": [
                    {
                        "name": inp.name,
                        "datatype": inp.datatype,
                        "shape": inp.shape
                    }
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "datatype": out.datatype,
                        "shape": out.shape
                    }
                    for out in metadata.outputs
                ]
            }
        except InferenceServerException as e:
            print(f"Failed to get metadata for {model_name}: {e}")
            return {}
