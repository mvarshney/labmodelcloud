# deploy_model.py

import os
import sys
import json
import requests

def deploy_model(model_name, model_version, model_path, triton_url):
    # Prepare the model configuration
    model_config = {
        "name": model_name,
        "version": model_version,
        "platform": "pytorch_libtorch",
        "input": [
            {
                "name": "input",
                "data_type": "TYPE_FP32",
                "dims": [1, 3, 224, 224]  # Example input shape
            }
        ],
        "output": [
            {
                "name": "output",
                "data_type": "TYPE_FP32",
                "dims": [1, 1000]  # Example output shape
            }
        ]
    }

    # Save the model configuration to a file
    config_path = os.path.join(model_path, "config.pbtxt")
    with open(config_path, 'w') as config_file:
        config_file.write(json.dumps(model_config, indent=2))

    # Deploy the model to Triton
    response = requests.post(f"{triton_url}/v2/repository/models/{model_name}/versions/{model_version}",
                             json={"model": model_config})

    if response.status_code == 200:
        print(f"Model {model_name} version {model_version} deployed successfully.")
    else:
        print(f"Failed to deploy model {model_name}: {response.text}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: deploy_model.py <model_name> <model_version> <model_path> <triton_url>")
        sys.exit(1)

    model_name = sys.argv[1]
    model_version = sys.argv[2]
    model_path = sys.argv[3]
    triton_url = sys.argv[4]

    deploy_model(model_name, model_version, model_path, triton_url)