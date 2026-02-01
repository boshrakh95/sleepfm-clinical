#!/bin/bash
################################################################################
# STAGES Cognitive Prediction Fine-tuning Script
################################################################################
# 
# This script runs the complete pipeline for fine-tuning SleepFM on STAGES
# cognitive prediction tasks.
#
# Usage:
#   ./run_cognitive_finetuning.sh [--generate-embeddings] [--target TARGET]
#
# Options:
#   --generate-embeddings    Generate embeddings before training (only needed once)
#   --target TARGET          Train for specific target (default: all targets)
#
# Examples:
#   # Generate embeddings (one time only)
#   ./run_cognitive_finetuning.sh --generate-embeddings
#
#   # Train for single target
#   ./run_cognitive_finetuning.sh --target sustained_attention
#
#   # Train for all targets
#   ./run_cognitive_finetuning.sh
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
GENERATE_EMBEDDINGS=false
SPECIFIC_TARGET=""
CONFIG_FILE="config_finetune_cognitive.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --generate-embeddings)
            GENERATE_EMBEDDINGS=true
            shift
            ;;
        --target)
            SPECIFIC_TARGET="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}STAGES Cognitive Prediction${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Generate embeddings if requested
if [ "$GENERATE_EMBEDDINGS" = true ]; then
    echo -e "${YELLOW}Step 1: Generating Embeddings${NC}"
    echo "This will take 1-2 hours for ~1500 subjects..."
    echo ""
    
    python generate_embeddings.py --config "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Embeddings generated successfully${NC}"
        echo ""
    else
        echo -e "${RED}✗ Embedding generation failed${NC}"
        exit 1
    fi
fi

# Define all cognitive targets
ALL_TARGETS=(
    "sustained_attention"
    "working_memory"
    "episodic_memory"
    "executive_functioning"
    "CPF_A.CPF_CR"
    "CPF_A.CPF_FP"
    "CPF_A.CPF_TPRT"
)

# Determine which targets to train
if [ -n "$SPECIFIC_TARGET" ]; then
    TARGETS=("$SPECIFIC_TARGET")
    echo -e "${YELLOW}Training for single target: $SPECIFIC_TARGET${NC}"
else
    TARGETS=("${ALL_TARGETS[@]}")
    echo -e "${YELLOW}Training for all ${#TARGETS[@]} targets${NC}"
fi

echo ""

# Train for each target
for i in "${!TARGETS[@]}"; do
    TARGET="${TARGETS[$i]}"
    TARGET_NUM=$((i + 1))
    TOTAL_TARGETS=${#TARGETS[@]}
    
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Target $TARGET_NUM/$TOTAL_TARGETS: $TARGET${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    
    # Create a temporary config with updated target and split path
    TEMP_CONFIG="config_temp_${TARGET}.yaml"
    cp "$CONFIG_FILE" "$TEMP_CONFIG"
    
    # Update target in config
    sed -i "s/target: .*/target: '$TARGET'/" "$TEMP_CONFIG"
    
    # Update split path for this target
    SPLIT_PATH="/home/boshra95/scratch/stages/sleepfm_format/splits/dataset_split_${TARGET}.json"
    sed -i "s|split_path: .*|split_path: '$SPLIT_PATH'|" "$TEMP_CONFIG"
    
    # Check if split file exists
    if [ ! -f "$SPLIT_PATH" ]; then
        echo -e "${YELLOW}⚠ Warning: Split file not found for $TARGET: $SPLIT_PATH${NC}"
        echo "Skipping this target..."
        rm "$TEMP_CONFIG"
        continue
    fi
    
    # Run fine-tuning
    echo "Starting training..."
    python finetune_cognitive.py --config "$TEMP_CONFIG"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training completed for $TARGET${NC}"
        echo ""
    else
        echo -e "${RED}✗ Training failed for $TARGET${NC}"
        echo ""
    fi
    
    # Cleanup temporary config
    rm "$TEMP_CONFIG"
    
    echo ""
done

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}All training complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Results saved to: /home/boshra95/scratch/stages/sleepfm_format/cognitive_models/"
echo ""
