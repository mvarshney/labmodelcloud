# ML Model Serving Platform

## Overview
The ML Model Serving Platform is designed to serve PyTorch models efficiently with high throughput and low latency. It supports multiple model variants and provides a control plane for model lifecycle management, enabling dynamic batching for GPU efficiency.

## Project Structure
```
ml-model-serving-platform/
├── control-plane/
│   ├── model-registry/          # Service for managing model metadata
│   ├── deployment-service/       # Service for deploying models to Triton
│   └── config-service/           # Service for managing routing configurations
├── data-plane/
│   └── gateway/                  # Main entry point for client requests
├── triton/                       # Contains model files and configuration for Triton
├── k8s/                          # Kubernetes configuration files for deployment
├── scripts/                      # Utility scripts for deployment and testing
├── tests/                        # Test cases for the project
├── .gitignore                    # Files and directories to be ignored by version control
├── requirements-dev.txt          # Development dependencies for the project
└── docker-compose.yml            # Docker Compose configuration for local development
```

## Features
- Serve PyTorch models on GPUs with high throughput and low latency.
- Support multiple model variants with configurable traffic routing.
- Provide control plane for model lifecycle management.
- Enable dynamic batching for GPU efficiency.
- Observable via metrics (Prometheus).

## Getting Started
1. Clone the repository:
   ```
   git clone <repository-url>
   cd ml-model-serving-platform
   ```

2. Install dependencies:
   ```
   pip install -r requirements-dev.txt
   ```

3. Run the application using Docker Compose:
   ```
   docker-compose up
   ```

4. Access the API at `http://localhost:8000`.

## Testing
To run the tests, use:
```
pytest tests/
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.