#!/bin/bash
# Quick Script: Exclude FLOW Channel and Regenerate
# ================================================
# This script shows how to exclude the FLOW channel from your analysis

cd /home/boshra95/sleepfm-clinical/sleepfm/stages_cognitive_prediction

# Backup original config
cp config_finetune_cognitive.yaml config_finetune_cognitive.yaml.backup

# Update config to exclude FLOW channel
# (You can edit manually or use sed as shown below)

echo "To exclude FLOW channel, update your config:"
echo ""
echo "data:"
echo "  exclude_channels: [\"Flow\"]  # Exclude FLOW channel"
echo ""
echo "Then run:"
echo ""
echo "# 1. Delete old embeddings (optional but recommended)"
echo "rm -rf /home/boshra95/scratch/stages/sleepfm_format/embeddings/*.npy"
echo ""
echo "# 2. Regenerate embeddings without FLOW"
echo "python generate_embeddings.py --config config_finetune_cognitive.yaml"
echo ""
echo "# 3. Fine-tune model"
echo "python finetune_cognitive.py --config config_finetune_cognitive.yaml"
echo ""
echo "Note: Embedding generation will take ~1-2 hours"
