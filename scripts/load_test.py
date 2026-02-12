# Load Test Script for ML Model Serving Platform

import requests
import time
import json

# Configuration
SERVER_URL = "http://localhost:8000/predict"  # Update with your server URL
NUM_REQUESTS = 100  # Total number of requests to send
BATCH_SIZE = 10  # Number of requests to send in each batch
DELAY_BETWEEN_BATCHES = 1  # Delay in seconds between batches

def load_test():
    for i in range(0, NUM_REQUESTS, BATCH_SIZE):
        batch_requests = []
        for j in range(BATCH_SIZE):
            if i + j < NUM_REQUESTS:
                # Create a sample request payload
                payload = {
                    "tensor": [[0.1, 0.2, 0.3, 0.4, 0.5]]  # Example tensor input
                }
                batch_requests.append(payload)

        # Send requests in batch
        responses = []
        for request in batch_requests:
            start_time = time.time()
            response = requests.post(SERVER_URL, json=request)
            latency = time.time() - start_time
            responses.append((response.json(), latency))

        # Print responses and latencies
        for response, latency in responses:
            print(f"Response: {response}, Latency: {latency:.4f} seconds")

        # Delay before sending the next batch
        time.sleep(DELAY_BETWEEN_BATCHES)

if __name__ == "__main__":
    load_test()