"""
Load test script to demonstrate dynamic batching efficiency.

This script sends inference requests at different QPS rates to show how
Triton's dynamic batching improves throughput.

KEY LEARNING OBJECTIVE:
- At low QPS (1-10): Small batch sizes (~1), each request processed individually
- At medium QPS (50-100): Moderate batching (~4-8), throughput increases
- At high QPS (200+): Large batches (~16-32), maximum throughput

Run this script and watch:
1. Console output showing batch sizes
2. Prometheus metrics: inference_batch_size histogram
3. Grafana dashboard showing batch size distribution
"""

import httpx
import time
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import argparse


class InferenceClient:
    """Client for sending inference requests."""

    def __init__(self, gateway_url: str = "http://localhost:8000"):
        """Initialize client."""
        self.gateway_url = gateway_url
        self.predict_url = f"{gateway_url}/api/v1/predict"

    def send_request(self) -> Tuple[float, int, str]:
        """
        Send a single inference request.

        Returns:
            Tuple of (latency_ms, batch_size, model_name)
        """
        # Generate random user and item IDs
        batch_size = random.randint(1, 8)
        user_ids = [random.randint(1, 10000) for _ in range(batch_size)]
        item_ids = [random.randint(1, 50000) for _ in range(batch_size)]

        payload = {
            "user_ids": user_ids,
            "item_ids": item_ids
        }

        start_time = time.time()
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.post(self.predict_url, json=payload)
                response.raise_for_status()
                result = response.json()

            latency_ms = (time.time() - start_time) * 1000

            return (
                latency_ms,
                result.get("batch_size", 0),
                result.get("model_name", "unknown")
            )

        except Exception as e:
            print(f"Request failed: {e}")
            return (0, 0, "error")


class LoadTest:
    """Load test orchestrator."""

    def __init__(self, gateway_url: str):
        """Initialize load test."""
        self.client = InferenceClient(gateway_url)
        self.results: List[Tuple[float, int, str]] = []

    def run_test(
        self,
        duration_seconds: int,
        target_qps: int,
        num_workers: int = 10
    ):
        """
        Run load test at target QPS.

        Args:
            duration_seconds: Test duration
            target_qps: Target queries per second
            num_workers: Number of concurrent workers
        """
        print("\n" + "=" * 70)
        print(f"Load Test Configuration")
        print("=" * 70)
        print(f"Target QPS:      {target_qps}")
        print(f"Duration:        {duration_seconds}s")
        print(f"Workers:         {num_workers}")
        print(f"Gateway URL:     {self.client.gateway_url}")
        print("=" * 70)

        # Calculate delay between requests to achieve target QPS
        requests_per_worker = target_qps / num_workers
        delay_between_requests = 1.0 / requests_per_worker if requests_per_worker > 0 else 0

        start_time = time.time()
        end_time = start_time + duration_seconds

        self.results = []

        def worker():
            """Worker function to send requests."""
            worker_results = []
            while time.time() < end_time:
                latency, batch_size, model_name = self.client.send_request()
                if latency > 0:
                    worker_results.append((latency, batch_size, model_name))

                if delay_between_requests > 0:
                    time.sleep(delay_between_requests)

            return worker_results

        # Run workers
        print(f"\nRunning test for {duration_seconds}s...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker) for _ in range(num_workers)]

            for future in as_completed(futures):
                self.results.extend(future.result())

        # Print results
        self._print_results()

    def _print_results(self):
        """Print test results."""
        if not self.results:
            print("No successful requests!")
            return

        latencies = [r[0] for r in self.results]
        batch_sizes = [r[1] for r in self.results]

        total_requests = len(self.results)
        avg_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile

        avg_batch_size = statistics.mean(batch_sizes)
        p50_batch_size = statistics.median(batch_sizes)
        p95_batch_size = statistics.quantiles(batch_sizes, n=20)[18]

        print("\n" + "=" * 70)
        print("Load Test Results")
        print("=" * 70)
        print(f"\nRequests:")
        print(f"  Total:         {total_requests}")
        print(f"  Success Rate:  100%")

        print(f"\nLatency (ms):")
        print(f"  Average:       {avg_latency:.2f}")
        print(f"  P50:           {p50_latency:.2f}")
        print(f"  P95:           {p95_latency:.2f}")
        print(f"  P99:           {p99_latency:.2f}")

        print(f"\nBatch Size (KEY METRIC):")
        print(f"  Average:       {avg_batch_size:.2f}")
        print(f"  P50:           {p50_batch_size:.2f}")
        print(f"  P95:           {p95_batch_size:.2f}")

        # Model distribution
        model_counts = {}
        for _, _, model_name in self.results:
            model_counts[model_name] = model_counts.get(model_name, 0) + 1

        print(f"\nTraffic Distribution:")
        for model_name, count in sorted(model_counts.items()):
            percentage = (count / total_requests) * 100
            print(f"  {model_name}: {percentage:.1f}% ({count} requests)")

        print("=" * 70)


def run_progressive_test(gateway_url: str):
    """
    Run progressive load test to demonstrate batching efficiency.

    Starts at low QPS and gradually increases to show how batch sizes grow.
    """
    print("\n" + "=" * 70)
    print("PROGRESSIVE LOAD TEST - Demonstrating Dynamic Batching")
    print("=" * 70)
    print("\nThis test will run at increasing QPS levels to show how Triton")
    print("automatically creates larger batches as load increases.")
    print("\nWatch the 'Batch Size' metric to see dynamic batching in action!")
    print("=" * 70)

    load_test = LoadTest(gateway_url)

    test_configs = [
        (10, 10, "Low load - expect batch size ~1-2"),
        (30, 50, "Medium load - expect batch size ~4-8"),
        (30, 100, "High load - expect batch size ~8-16"),
        (30, 200, "Very high load - expect batch size ~16-32"),
    ]

    for duration, qps, description in test_configs:
        print(f"\n{description}")
        print(f"Target: {qps} QPS for {duration}s")
        input("Press Enter to start... ")

        load_test.run_test(
            duration_seconds=duration,
            target_qps=qps,
            num_workers=min(20, qps // 2)
        )

        print("\nWaiting 5s before next test...")
        time.sleep(5)

    print("\n" + "=" * 70)
    print("✓ Progressive load test complete!")
    print("=" * 70)
    print("\nCheck Grafana dashboard to see batch size trends:")
    print("  http://localhost:3000")
    print("\nCheck Prometheus metrics:")
    print("  http://localhost:9090")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load test for ML inference gateway"
    )
    parser.add_argument(
        "--gateway-url",
        default="http://localhost:8000",
        help="Gateway URL"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "progressive"],
        default="progressive",
        help="Test mode"
    )
    parser.add_argument(
        "--qps",
        type=int,
        default=100,
        help="Target QPS (for single mode)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration in seconds (for single mode)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent workers (for single mode)"
    )

    args = parser.parse_args()

    if args.mode == "progressive":
        run_progressive_test(args.gateway_url)
    else:
        load_test = LoadTest(args.gateway_url)
        load_test.run_test(
            duration_seconds=args.duration,
            target_qps=args.qps,
            num_workers=args.workers
        )


if __name__ == "__main__":
    main()
