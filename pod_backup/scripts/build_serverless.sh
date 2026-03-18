#!/bin/bash
set -e

GITHUB_REPO="chistodlyamarketa111-pixel/ace-step-runpod-serverless"
IMAGE_NAME="ghcr.io/$GITHUB_REPO/ace-step-serverless:latest"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

echo "============================================"
echo " Build Docker image & create RunPod Serverless endpoint"
echo "============================================"

if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: Set RUNPOD_API_KEY env var"
    exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "ERROR: Set GITHUB_TOKEN env var (with write:packages scope)"
    exit 1
fi

echo ""
echo "[1/4] Logging into GitHub Container Registry..."
echo "$GITHUB_TOKEN" | docker login ghcr.io -u chistodlyamarketa111-pixel --password-stdin

echo ""
echo "[2/4] Building Docker image (this takes 10-20 min)..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTEXT_DIR="$SCRIPT_DIR/.."

docker build -t "$IMAGE_NAME" -f "$CONTEXT_DIR/Dockerfile.serverless" "$CONTEXT_DIR"

echo ""
echo "[3/4] Pushing Docker image to ghcr.io..."
docker push "$IMAGE_NAME"

echo ""
echo "[4/4] Creating RunPod Serverless endpoint..."
RESPONSE=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    "https://api.runpod.io/graphql" \
    -d '{
        "query": "mutation { saveEndpoint(input: { name: \"ace-step-macan-lora\", templateId: null, dockerImage: \"'"$IMAGE_NAME"'\", gpuIds: \"AMPERE_48\", workersMin: 0, workersMax: 1, idleTimeout: 5, flashBoot: true, volumeInGb: 0, env: [] }) { id name } }"
    }')

echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'errors' in d:
    print('ERROR:', d['errors'][0]['message'])
else:
    ep = d['data']['saveEndpoint']
    print(f'Endpoint created!')
    print(f'  ID: {ep[\"id\"]}')
    print(f'  Name: {ep[\"name\"]}')
    print(f'  URL: https://api.runpod.ai/v2/{ep[\"id\"]}/run')
    print()
    print('Usage:')
    print(f'  curl -X POST https://api.runpod.ai/v2/{ep[\"id\"]}/runsync \\\\')
    print(f'    -H \"Authorization: Bearer YOUR_RUNPOD_KEY\" \\\\')
    print(f'    -H \"Content-Type: application/json\" \\\\')
    print(f'    -d \'{{\"input\": {{\"caption\": \"voice_macan, Russian hip-hop/rap track\", \"lyrics\": \"...\", \"duration\": 60}}}}\'')
"

echo ""
echo "============================================"
echo " Done!"
echo "============================================"
