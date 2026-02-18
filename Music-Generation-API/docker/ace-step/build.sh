#!/bin/bash
set -e

IMAGE_NAME="${1:-ace-step-serverless}"
IMAGE_TAG="${2:-latest}"

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "This will take 15-30 minutes (model download ~7GB)"

docker build \
  --progress=plain \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  .

echo ""
echo "Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Test locally:"
echo "  docker run --gpus all -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Push to Docker Hub:"
echo "  docker tag ${IMAGE_NAME}:${IMAGE_TAG} YOUR_DOCKERHUB_USER/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  docker push YOUR_DOCKERHUB_USER/${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Then create a RunPod Serverless Endpoint with this image."
