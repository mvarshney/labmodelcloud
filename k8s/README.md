# Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the ML serving platform on k3s.

## Prerequisites

- k3s installed and running
- kubectl configured to access your k3s cluster
- Docker images built for all services

## Deployment Order

Deploy in this order to handle dependencies:

```bash
# 1. Storage (MinIO)
kubectl apply -f storage/minio.yaml

# 2. MLflow
kubectl apply -f storage/mlflow.yaml

# 3. Model serving (Triton)
kubectl apply -f data-plane/triton.yaml

# 4. Control plane services
kubectl apply -f control-plane/config-service.yaml
kubectl apply -f control-plane/model-registry.yaml
kubectl apply -f control-plane/deployment-service.yaml

# 5. Data plane (Gateway)
kubectl apply -f data-plane/gateway.yaml

# 6. Monitoring
kubectl apply -f monitoring/prometheus.yaml
kubectl apply -f monitoring/grafana.yaml
```

## Quick Deploy All

```bash
./deploy-all.sh
```

## Access Services

```bash
# Port forward to access services locally
kubectl port-forward svc/gateway 8000:8000
kubectl port-forward svc/grafana 3000:3000
kubectl port-forward svc/prometheus 9090:9090
kubectl port-forward svc/minio 9001:9001
```

## Notes

- All manifests use `hostPath` volumes for simplicity on single-node k3s
- In production, use proper persistent volume provisioning
- Resource limits are set conservatively for single-node deployment
