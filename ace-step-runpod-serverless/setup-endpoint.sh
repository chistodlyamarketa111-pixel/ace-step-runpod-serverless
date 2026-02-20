#!/bin/bash
#
# Setup RunPod Serverless Endpoint for ACE-Step v1.5
#
# Prerequisites:
#   1. Push this repo to GitHub
#   2. Connect GitHub to RunPod: https://www.runpod.io/console/user/settings
#   3. Set RUNPOD_API_KEY environment variable
#
# Usage:
#   export RUNPOD_API_KEY="your_runpod_api_key"
#   ./setup-endpoint.sh
#

set -e

if [ -z "$RUNPOD_API_KEY" ]; then
  echo "ERROR: RUNPOD_API_KEY not set"
  echo "Get your API key from: https://www.runpod.io/console/user/settings"
  exit 1
fi

ENDPOINT_NAME="${ENDPOINT_NAME:-ace-step-v15}"

echo "=== ACE-Step v1.5 RunPod Serverless Setup ==="
echo ""
echo "This endpoint requires GitHub integration with RunPod."
echo "Please create the endpoint manually via the RunPod UI:"
echo ""
echo "  1. Go to: https://www.runpod.io/console/serverless"
echo "  2. Click '+ New Endpoint'"
echo "  3. Endpoint name: ${ENDPOINT_NAME}"
echo ""
echo "  GPU Configuration:"
echo "    - Select: 48 GB GPU"
echo "    - Max workers: 2"
echo "    - Active workers: 0"
echo "    - GPU count: 1"
echo "    - Idle timeout: 1 sec"
echo "    - Execution timeout: 600 sec"
echo "    - FlashBoot: Enabled"
echo ""
echo "  Repository configuration:"
echo "    - Branch: main"
echo "    - Dockerfile Path: Dockerfile"
echo "    - Build Context: worker"
echo ""
echo "  4. Click 'Save Endpoint'"
echo ""
echo "After creation, you can test with:"
echo ""

echo "Listing existing endpoints..."
echo ""

RESPONSE=$(curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"query":"query { myself { endpoints { id name } } }"}')

echo "Your endpoints:"
echo "$RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
endpoints = data.get('data', {}).get('myself', {}).get('endpoints', [])
if not endpoints:
    print('  (no endpoints found)')
else:
    for ep in endpoints:
        print(f\"  - {ep['name']}: {ep['id']}\")
" 2>/dev/null || echo "  (could not parse response)"

echo ""
echo "Once you have the endpoint ID, test it with:"
echo ""
echo "  curl -X POST 'https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run' \\"
echo "    -H 'Authorization: Bearer ${RUNPOD_API_KEY}' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"input\": {\"prompt\": \"upbeat electronic dance music\", \"lyrics\": \"[verse]\\nTest lyrics\", \"duration\": 30}}'"
echo ""
echo "Then set in your Music Generation API:"
echo "  ACESTEP_ENDPOINT_ID=YOUR_ENDPOINT_ID"
echo "  RUNPOD_API_KEY=${RUNPOD_API_KEY}"
