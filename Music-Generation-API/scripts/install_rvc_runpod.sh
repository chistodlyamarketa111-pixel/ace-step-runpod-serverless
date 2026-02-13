#!/bin/bash
# Install RVC (Retrieval-based Voice Conversion) on RunPod pod
# Run this in the RunPod Web Terminal to enable voice conversion in post-processing

set -e

echo "=== Installing RVC on RunPod Pod ==="

source /opt/conda/etc/profile.d/conda.sh
conda activate pyenv

echo "Installing rvc-python..."
pip install rvc-python

echo "Creating RVC models directory..."
mkdir -p /workspace/models/rvc

echo ""
echo "=== RVC Installation Complete ==="
echo ""
echo "To add voice models, create a folder in /workspace/models/rvc/<model_name>/"
echo "and place the .pth file (and optionally .index file) inside."
echo ""
echo "Example:"
echo "  mkdir -p /workspace/models/rvc/my_voice"
echo "  # Copy your_model.pth and your_model.index into /workspace/models/rvc/my_voice/"
echo ""
echo "The post-processing pipeline will automatically detect available models."
echo "You can check available models via GET /health endpoint."
