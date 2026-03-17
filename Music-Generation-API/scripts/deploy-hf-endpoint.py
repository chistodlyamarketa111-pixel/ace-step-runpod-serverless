#!/usr/bin/env python3
"""
Deploy ACE-Step v1.5 to Hugging Face Inference Endpoints.

Prerequisites:
  1. Build & push Docker image:
     cd docker/ace-step-hf
     docker build --platform linux/amd64 -t YOUR_DOCKERHUB/ace-step-hf:latest .
     docker push YOUR_DOCKERHUB/ace-step-hf:latest

  2. Set environment variables:
     export HF_TOKEN=hf_...
     export DOCKER_IMAGE=YOUR_DOCKERHUB/ace-step-hf:latest

  3. Run this script:
     python scripts/deploy-hf-endpoint.py

Usage:
  python deploy-hf-endpoint.py [--name NAME] [--gpu GPU_TYPE] [--region REGION]
"""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Deploy ACE-Step to HF Inference Endpoints")
    parser.add_argument("--name", default="ace-step-v15", help="Endpoint name (default: ace-step-v15)")
    parser.add_argument("--docker-image", default=None, help="Docker image URL (default: from DOCKER_IMAGE env)")
    parser.add_argument("--gpu", default="nvidia-a10g", choices=["nvidia-t4", "nvidia-a10g", "nvidia-l4", "nvidia-a100"],
                        help="GPU type (default: nvidia-a10g, ~$1/hr)")
    parser.add_argument("--vendor", default="aws", choices=["aws", "azure", "gcp"], help="Cloud provider")
    parser.add_argument("--region", default="us-east-1", help="Region (default: us-east-1)")
    parser.add_argument("--min-replicas", type=int, default=0, help="Min replicas, 0 = scale to zero (default: 0)")
    parser.add_argument("--max-replicas", type=int, default=1, help="Max replicas (default: 1)")
    parser.add_argument("--namespace", default=None, help="HF namespace/org (default: your username)")
    parser.add_argument("--model", default="acestep-v15-turbo",
                        choices=["acestep-v15-turbo", "acestep-v15-sft", "acestep-v15-base", "acestep-v15-turbo-shift3"],
                        help="Default model to preload (default: acestep-v15-turbo)")
    parser.add_argument("--cpu-offload", action="store_true", default=True, help="Enable CPU offload for long tracks")
    parser.add_argument("--pause", action="store_true", help="Pause existing endpoint")
    parser.add_argument("--resume", action="store_true", help="Resume paused endpoint")
    parser.add_argument("--delete", action="store_true", help="Delete endpoint")
    parser.add_argument("--status", action="store_true", help="Check endpoint status")
    args = parser.parse_args()

    try:
        from huggingface_hub import (
            create_inference_endpoint,
            get_inference_endpoint,
            login,
        )
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: Set HF_TOKEN environment variable with your Hugging Face token")
        print("  Get token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    login(token=hf_token)

    if args.status:
        try:
            ep = get_inference_endpoint(args.name, namespace=args.namespace)
            print(f"Name:     {ep.name}")
            print(f"Status:   {ep.status}")
            print(f"URL:      {ep.url}")
            print(f"Created:  {ep.created_at}")
            print(f"Updated:  {ep.updated_at}")
        except Exception as e:
            print(f"Error getting endpoint: {e}")
        return

    if args.pause:
        ep = get_inference_endpoint(args.name, namespace=args.namespace)
        print(f"Pausing endpoint '{args.name}'...")
        ep.pause()
        print("Paused. No charges while paused.")
        return

    if args.resume:
        ep = get_inference_endpoint(args.name, namespace=args.namespace)
        print(f"Resuming endpoint '{args.name}'...")
        ep.resume()
        print("Resuming... Use --status to check when ready.")
        return

    if args.delete:
        ep = get_inference_endpoint(args.name, namespace=args.namespace)
        print(f"Deleting endpoint '{args.name}'...")
        ep.delete()
        print("Deleted.")
        return

    docker_image = args.docker_image or os.environ.get("DOCKER_IMAGE")
    if not docker_image:
        print("ERROR: Specify Docker image via --docker-image or DOCKER_IMAGE env var")
        print("  Example: --docker-image your-username/ace-step-hf:latest")
        sys.exit(1)

    print(f"Creating HF Inference Endpoint:")
    print(f"  Name:          {args.name}")
    print(f"  Docker image:  {docker_image}")
    print(f"  GPU:           {args.gpu}")
    print(f"  Region:        {args.vendor}/{args.region}")
    print(f"  Scale:         {args.min_replicas}-{args.max_replicas} replicas")
    print(f"  Default model: {args.model}")
    print(f"  CPU offload:   {args.cpu_offload}")
    print()

    env_vars = {
        "ACESTEP_DIT_MODEL": args.model,
        "ACESTEP_CPU_OFFLOAD": "true" if args.cpu_offload else "false",
    }

    try:
        endpoint = create_inference_endpoint(
            name=args.name,
            namespace=args.namespace,
            repository="ACE-Step/Ace-Step1.5",
            framework="custom",
            task="custom",
            accelerator="gpu",
            vendor=args.vendor,
            region=args.region,
            type="protected",
            instance_size="x1",
            instance_type=args.gpu,
            min_replica=args.min_replicas,
            max_replica=args.max_replicas,
            custom_image={
                "url": docker_image,
                "health_route": "/health",
                "env": env_vars,
            },
        )
    except Exception as e:
        print(f"Error creating endpoint: {e}")
        sys.exit(1)

    print(f"Endpoint created!")
    print(f"  Status: {endpoint.status}")
    print(f"  URL:    {endpoint.url}")
    print()
    print("Waiting for endpoint to become ready (this may take 5-15 minutes)...")
    print("  (Models need to load into GPU memory)")
    print()

    try:
        endpoint.wait(timeout=900)
        print()
        print(f"Endpoint is READY!")
        print(f"  URL: {endpoint.url}")
        print()
        print("Set this in your Replit environment:")
        print(f"  HF_ENDPOINT_URL={endpoint.url}")
        if hf_token:
            print(f"  HF_API_TOKEN={hf_token}")
        print()
        print("Test health:")
        print(f'  curl -H "Authorization: Bearer {hf_token}" {endpoint.url}/health')
        print()
        print("To pause (stop billing):")
        print(f"  python {sys.argv[0]} --pause --name {args.name}")
    except TimeoutError:
        print()
        print(f"Timeout waiting for endpoint. Check status at:")
        print(f"  https://ui.endpoints.huggingface.co/")
        print(f"  or: python {sys.argv[0]} --status --name {args.name}")


if __name__ == "__main__":
    main()
