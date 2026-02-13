#!/bin/bash

# Deploy all Kubernetes resources in the correct order

set -e

echo "=================================="
echo "Deploying ML Serving Platform to K8s"
echo "=================================="

# Function to wait for deployment
wait_for_deployment() {
    local name=$1
    local namespace=${2:-default}
    echo "Waiting for deployment/$name to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/$name -n $namespace
    echo "✓ $name is ready"
}

# Function to wait for job
wait_for_job() {
    local name=$1
    local namespace=${2:-default}
    echo "Waiting for job/$name to complete..."
    kubectl wait --for=condition=complete --timeout=300s job/$name -n $namespace
    echo "✓ $name completed"
}

echo ""
echo "[1/7] Deploying Storage Layer..."
kubectl apply -f storage/minio.yaml
wait_for_deployment minio
wait_for_job minio-init

kubectl apply -f storage/mlflow.yaml
wait_for_deployment mlflow

echo ""
echo "[2/7] Deploying Triton Inference Server..."
kubectl apply -f data-plane/triton.yaml
wait_for_deployment triton

echo ""
echo "[3/7] Deploying Config Service..."
kubectl apply -f control-plane/config-service.yaml
wait_for_deployment config-service

echo ""
echo "[4/7] Deploying Model Registry..."
kubectl apply -f control-plane/model-registry.yaml
wait_for_deployment model-registry

echo ""
echo "[5/7] Deploying Deployment Service..."
kubectl apply -f control-plane/deployment-service.yaml
wait_for_deployment deployment-service

echo ""
echo "[6/7] Deploying Gateway..."
kubectl apply -f data-plane/gateway.yaml
wait_for_deployment gateway

echo ""
echo "[7/7] Deploying Monitoring..."
kubectl apply -f monitoring/prometheus.yaml
wait_for_deployment prometheus

kubectl apply -f monitoring/grafana.yaml
wait_for_deployment grafana

echo ""
echo "=================================="
echo "✓ Deployment Complete!"
echo "=================================="
echo ""
echo "Access services using port-forward:"
echo "  kubectl port-forward svc/gateway 8000:8000"
echo "  kubectl port-forward svc/grafana 3000:3000"
echo "  kubectl port-forward svc/prometheus 9090:9090"
echo "  kubectl port-forward svc/minio 9001:9001"
echo ""
echo "Check pod status:"
echo "  kubectl get pods"
echo ""
