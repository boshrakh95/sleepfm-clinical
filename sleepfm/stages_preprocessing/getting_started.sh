#!/bin/bash
# STAGES Data Preparation - Getting Started
# ==========================================
#
# Quick setup and first test
#
# Usage: bash getting_started.sh

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   STAGES Data Preparation for SleepFM - Getting Started   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to preprocessing directory
cd "$(dirname "$0")"

echo "📁 Current directory: $(pwd)"
echo ""

# Step 1: Check if dependencies are installed
echo "═══════════════════════════════════════════════════════════"
echo "Step 1: Checking dependencies"
echo "═══════════════════════════════════════════════════════════"
echo ""

if python -c "import loguru" 2>/dev/null; then
    echo "✓ Dependencies already installed"
else
    echo "⚠️  Missing dependencies. Installing..."
    echo ""
    bash setup_environment.sh
fi

echo ""

# Step 2: Verify configuration
echo "═══════════════════════════════════════════════════════════"
echo "Step 2: Verifying configuration"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "Input directory:"
INPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('config_stages_conversion.yaml'))['input']['base_dir'])")
echo "  $INPUT_DIR"

if [ -d "$INPUT_DIR/eeg_segmented" ]; then
    NUM_SUBJECTS=$(ls "$INPUT_DIR/eeg_segmented" | wc -l)
    echo "  ✓ Found $NUM_SUBJECTS subjects"
else
    echo "  ✗ Directory not found!"
    echo "  Please edit config_stages_conversion.yaml with correct paths"
    exit 1
fi

echo ""
echo "Output directory:"
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('config_stages_conversion.yaml'))['output']['base_dir'])")
echo "  $OUTPUT_DIR"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "  Creating output directory..."
    mkdir -p "$OUTPUT_DIR"
fi

echo ""

# Step 3: Test single subject
echo "═══════════════════════════════════════════════════════════"
echo "Step 3: Testing single subject conversion"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "This will convert one subject to verify everything works..."
echo ""

read -p "Continue with test? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python test_single_subject.py --auto
    
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "✓ Test successful!"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Review the test output above"
    echo "2. Check the converted file in:"
    echo "   $OUTPUT_DIR/hdf5_data/"
    echo ""
    echo "3. For pilot run (10 subjects):"
    echo "   - Edit config_stages_conversion.yaml"
    echo "   - Set: pilot_mode: true, pilot_count: 10"
    echo "   - Run: bash run_pipeline.sh"
    echo ""
    echo "4. For full run (~1500 subjects):"
    echo "   - Edit config_stages_conversion.yaml"
    echo "   - Set: pilot_mode: false"
    echo "   - Run: bash run_pipeline.sh"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
else
    echo "Test cancelled. You can run it manually with:"
    echo "  python test_single_subject.py --auto"
fi

echo ""
