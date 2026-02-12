# test_e2e.py

import pytest
import httpx

BASE_URL = "http://localhost:8000"  # Adjust the base URL as needed for your setup

@pytest.fixture(scope="module")
def client():
    with httpx.Client() as client:
        yield client

def test_model_registration(client):
    response = client.post(f"{BASE_URL}/register", json={
        "model_name": "recommendation_v1",
        "version": "1",
        "metadata": {
            "description": "First version of the recommendation model"
        }
    })
    assert response.status_code == 201
    assert response.json()["model_name"] == "recommendation_v1"

def test_model_deployment(client):
    response = client.post(f"{BASE_URL}/deploy", json={
        "model_name": "recommendation_v1",
        "version": "1"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "deployed"

def test_prediction(client):
    response = client.post(f"{BASE_URL}/predict", json={
        "tensor": [[0.1, 0.2, 0.3, 0.4, 0.5]]
    })
    assert response.status_code == 200
    assert "score" in response.json()

def test_dynamic_routing(client):
    response = client.post(f"{BASE_URL}/predict", json={
        "tensor": [[0.1, 0.2, 0.3, 0.4, 0.5]]
    })
    assert response.status_code == 200
    assert "score" in response.json()

def test_zero_downtime_deployment(client):
    response = client.post(f"{BASE_URL}/deploy", json={
        "model_name": "recommendation_v2",
        "version": "1"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "deployed"