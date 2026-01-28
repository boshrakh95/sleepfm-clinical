#!/bin/bash
# Run STAGES Data Preparation Pipeline
# =====================================
#
# This script runs all data preparation steps in sequence:
# 1. Convert NumPy arrays to HDF5
# 2. Prepare labels and demographics
# 3. Create train/val/test splits
# 4. Validate converted data
#
# Usage:
#   bash run_pipeline.sh [--pilot]
#
# Options:
#   --pilot: Run in pilot mode (process only first 10 subjects)

set -e  # Exit on error

# Configuration file
CONFIG="config_stages_conversion.yaml"

# Check if pilot mode
PILOT_MODE=false
if [[ "$1" == "--pilot" ]]; then
    PILOT_MODE=true
    echo "Running in PILOT MODE (first 10 subjects only)"
fi

# Activate environment if needed
# conda activate sleepfm_env  # Uncomment if using conda

echo "=============================================="
echo "STAGES Data Preparation Pipeline"
echo "=============================================="
echo ""

# Step 1: Convert to HDF5
echo "Step 1: Converting NumPy arrays to HDF5..."
echo "----------------------------------------------"
python convert_to_hdf5.py --config $CONFIG
echo ""

# Step 2: Prepare labels
echo "Step 2: Preparing labels and demographics..."
echo "----------------------------------------------"
python prepare_labels.py --config $CONFIG
echo ""

# Step 3: Create splits
echo "Step 3: Creating train/val/test splits..."
echo "----------------------------------------------"
python create_splits.py --config $CONFIG
echo ""

# Step 4: Validate data
echo "Step 4: Validating converted data..."
echo "----------------------------------------------"
python validate_data.py --config $CONFIG --detailed 5
echo ""

echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Review validation report in: output/validation/"
echo "2. Generate embeddings using: sleepfm/pipeline/generate_embeddings.py"
echo "3. Train cognitive prediction model"
echo ""
