"""
Upload model files to MinIO (S3-compatible storage).

This script uploads the generated PyTorch models to MinIO so they can be
deployed by the deployment service.
"""

import boto3
from botocore.exceptions import ClientError
from pathlib import Path
import sys


def upload_models(
    s3_endpoint: str = "http://localhost:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
    bucket: str = "models"
):
    """
    Upload all models from triton/models to MinIO.

    Args:
        s3_endpoint: MinIO endpoint URL
        access_key: MinIO access key
        secret_key: MinIO secret key
        bucket: S3 bucket name
    """
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        endpoint_url=s3_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    # Create bucket if it doesn't exist
    try:
        s3_client.head_bucket(Bucket=bucket)
        print(f"✓ Bucket '{bucket}' exists")
    except ClientError:
        print(f"Creating bucket '{bucket}'...")
        s3_client.create_bucket(Bucket=bucket)
        print(f"✓ Bucket '{bucket}' created")

    # Find all model.pt files
    models_dir = Path(__file__).parent.parent / "triton" / "models"
    model_files = list(models_dir.glob("*/1/model.pt"))

    if not model_files:
        print("⚠ No model files found!")
        print(f"  Expected location: {models_dir}")
        print("  Run generate_models.py first to create models")
        return

    print(f"\nFound {len(model_files)} model file(s) to upload")
    print("=" * 60)

    # Upload each model
    for model_file in model_files:
        # Extract model name from path
        model_name = model_file.parent.parent.name
        version = model_file.parent.name

        # S3 key (path in bucket)
        s3_key = f"{model_name}/{version}/model.pt"

        print(f"\nUploading {model_name}/{version}...")
        print(f"  Local: {model_file}")
        print(f"  S3:    s3://{bucket}/{s3_key}")

        try:
            # Upload file
            s3_client.upload_file(
                str(model_file),
                bucket,
                s3_key
            )

            # Verify upload
            response = s3_client.head_object(Bucket=bucket, Key=s3_key)
            size_mb = response['ContentLength'] / (1024 * 1024)

            print(f"  ✓ Uploaded successfully ({size_mb:.2f} MB)")

        except ClientError as e:
            print(f"  ✗ Upload failed: {e}")

    print("\n" + "=" * 60)
    print("✓ Upload complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Register models via Model Registry API")
    print("2. Deploy models via Deployment Service API")
    print("3. Use deploy_model.py script for convenience")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload models to MinIO")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:9000",
        help="MinIO endpoint URL"
    )
    parser.add_argument(
        "--access-key",
        default="minioadmin",
        help="MinIO access key"
    )
    parser.add_argument(
        "--secret-key",
        default="minioadmin",
        help="MinIO secret key"
    )
    parser.add_argument(
        "--bucket",
        default="models",
        help="S3 bucket name"
    )

    args = parser.parse_args()

    upload_models(
        s3_endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket=args.bucket
    )
