#!/bin/bash

# Quickstart script for ML Model Serving Platform
# This script sets up everything needed to run the platform

set -e

echo "======================================================================"
echo "ML Model Serving Platform - Quickstart"
echo "======================================================================"
echo ""

# Check prerequisites
echo "[1/6] Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "✗ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "✗ docker-compose not found. Please install docker-compose first."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found. Please install Python 3.11+ first."
    exit 1
fi

echo "✓ Prerequisites OK"

# Install Python dependencies
echo ""
echo "[2/6] Installing Python dependencies..."
pip install -q -r requirements-dev.txt
echo "✓ Dependencies installed"

# Generate models
echo ""
echo "[3/6] Generating dummy PyTorch models..."
python scripts/generate_models.py
python scripts/create_triton_configs.py
echo "✓ Models generated"

# Start services
echo ""
echo "[4/6] Starting services with docker-compose..."
echo "This may take a few minutes on first run (downloading images)..."
docker-compose up -d
echo "✓ Services started"

# Wait for services to be ready
echo ""
echo "[5/6] Waiting for services to be healthy..."
echo "This may take 30-60 seconds..."

wait_for_service() {
    local name=$1
    local url=$2
    local max_retries=30

    for i in $(seq 1 $max_retries); do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo "✓ $name is ready"
            return 0
        fi
        sleep 2
    done

    echo "⚠ $name did not become ready in time"
    return 1
}

wait_for_service "Gateway" "http://localhost:8000/health"
wait_for_service "Model Registry" "http://localhost:8001/health"
wait_for_service "Config Service" "http://localhost:8002/health"
wait_for_service "Deployment Service" "http://localhost:8003/health"
wait_for_service "Triton" "http://localhost:8000/v2/health/ready"
wait_for_service "MLflow" "http://localhost:5000/health"

# Upload and deploy models
echo ""
echo "[6/6] Uploading and deploying models..."

# Upload to MinIO
python scripts/upload_to_minio.py

# Deploy recommendation_v1
python scripts/deploy_model.py --model-name recommendation_v1 --version 1

echo ""
echo "======================================================================"
echo "✓ Setup Complete!"
echo "======================================================================"
echo ""
echo "Services are running at:"
echo "  • Gateway:          http://localhost:8000"
echo "  • Model Registry:   http://localhost:8001"
echo "  • Config Service:   http://localhost:8002"
echo "  • Deployment:       http://localhost:8003"
echo "  • MLflow:           http://localhost:5000"
echo "  • MinIO Console:    http://localhost:9001 (minioadmin/minioadmin)"
echo "  • Prometheus:       http://localhost:9090"
echo "  • Grafana:          http://localhost:3000 (admin/admin)"
echo ""
echo "======================================================================"
echo "Quick Test:"
echo "======================================================================"
echo ""
echo "Send an inference request:"
echo ""
echo "  curl -X POST http://localhost:8000/api/v1/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"user_ids\": [123, 456], \"item_ids\": [1001, 2002]}'"
echo ""
echo "======================================================================"
echo "Next Steps:"
echo "======================================================================"
echo ""
echo "1. Try the load test to see dynamic batching in action:"
echo "   python scripts/load_test.py --mode progressive"
echo ""
echo "2. Deploy the second model for A/B testing:"
echo "   python scripts/deploy_model.py --model-name recommendation_v2 --version 1"
echo ""
echo "3. Update routing weights (70% v1, 30% v2):"
echo "   curl -X PUT http://localhost:8002/api/v1/config/routing \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"weights\": {\"recommendation_v1\": 0.7, \"recommendation_v2\": 0.3}}'"
echo ""
echo "4. View metrics in Grafana:"
echo "   open http://localhost:3000"
echo ""
echo "======================================================================"
echo "To stop all services:"
echo "  docker-compose down"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f gateway"
echo "======================================================================"
