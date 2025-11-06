#!/bin/bash

echo "========================================="
echo "QUICK DEBUG TEST - Path D + x->c->y"
echo "Runtime: ~10 minutes"
echo "========================================="
echo ""

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Create output directories
mkdir -p results

echo "Running debug tests on PH2 split 0 (20 samples)..."
echo ""

# Run the quick test
python debug_pathd_quick.py --test all --max_samples 20

echo ""
echo "========================================="
echo "DEBUG COMPLETE!"
echo "========================================="
echo ""
echo "If no errors, you can run full experiments:"
echo "  sbatch scripts/run_pathd_rulebased.sh  (24h)"
echo "  sbatch scripts/run_pathd_llm.sh        (48h)"
echo "  sbatch scripts/main.sh                 (3h)"
