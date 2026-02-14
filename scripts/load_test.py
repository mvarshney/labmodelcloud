#!/usr/bin/env python3
"""Concurrent load test to observe cold-start vs cached latency and eviction."""

import asyncio
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL = "http://127.0.0.1:8000"

# All 35 model IDs
MODEL_IDS = (
    [f"text-classifier-v{i}" for i in range(10)]
    + [f"embedding-model-v{i}" for i in range(10)]
    + [f"summarizer-v{i}" for i in range(10)]
    + [f"sentiment-analyzer-v{i}" for i in range(5)]
)

NUM_REQUESTS = 100
CONCURRENCY = 10


async def send_request(
    client: httpx.AsyncClient, model_id: str
) -> tuple[str, float, int]:
    """Send a predict request and return (model_id, latency, status_code)."""
    payload = {"input": [[random.random() for _ in range(10)]]}
    start = time.perf_counter()
    resp = await client.post(
        f"{BASE_URL}/v1/predict/{model_id}",
        json=payload,
        timeout=60.0,
    )
    elapsed = time.perf_counter() - start
    return model_id, elapsed, resp.status_code


async def run_load_test():
    print(f"Load test: {NUM_REQUESTS} requests, concurrency={CONCURRENCY}")
    print(f"Targeting {len(MODEL_IDS)} models\n")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    results: list[tuple[str, float, int]] = []

    async with httpx.AsyncClient() as client:
        # Verify server is up
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=5.0)
            resp.raise_for_status()
        except Exception as e:
            print(f"Server not reachable at {BASE_URL}: {e}")
            print("Start the server first: python scripts/run.py")
            sys.exit(1)

        async def bounded_request(model_id: str):
            async with semaphore:
                result = await send_request(client, model_id)
                results.append(result)

        # Send requests with random model selection
        tasks = []
        for _ in range(NUM_REQUESTS):
            model_id = random.choice(MODEL_IDS)
            tasks.append(asyncio.create_task(bounded_request(model_id)))

        await asyncio.gather(*tasks)

    # Analyze results
    latencies = [r[1] for r in results]
    errors = [r for r in results if r[2] != 200]

    print(f"Results ({len(results)} requests):")
    print(f"  Success: {len(results) - len(errors)}/{len(results)}")
    print(f"  Errors:  {len(errors)}")
    print(f"\nLatency (seconds):")
    print(f"  Min:    {min(latencies):.3f}")
    print(f"  Max:    {max(latencies):.3f}")
    print(f"  Mean:   {statistics.mean(latencies):.3f}")
    print(f"  Median: {statistics.median(latencies):.3f}")
    print(f"  P95:    {sorted(latencies)[int(len(latencies) * 0.95)]:.3f}")
    print(f"  P99:    {sorted(latencies)[int(len(latencies) * 0.99)]:.3f}")

    # Breakdown: fast (likely cached) vs slow (likely cold-start)
    fast = [l for l in latencies if l < 0.1]
    slow = [l for l in latencies if l >= 0.1]
    print(f"\nCache analysis:")
    print(f"  Fast (<100ms, likely cached): {len(fast)}")
    print(f"  Slow (>=100ms, likely cold):  {len(slow)}")

    if errors:
        print(f"\nError details:")
        for model_id, latency, status in errors[:10]:
            print(f"  {model_id}: HTTP {status} ({latency:.3f}s)")


if __name__ == "__main__":
    asyncio.run(run_load_test())
