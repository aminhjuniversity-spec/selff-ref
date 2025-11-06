#!/bin/bash
#SBATCH --job-name=debug_ph2_full
#SBATCH --account=def-arashmoh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G 
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

echo "========================================="
echo "🔍 DEBUG MODE - Full PH2 Dataset"
echo "Testing Path D on ALL 5 PH2 splits"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

# Create directories
mkdir -p logs
mkdir -p results/path_d_full_evaluation

# Activate environment
source /project/def-arashmoh/shahab33/Medsam/self/bin/activate
echo "Activated environment: $VIRTUAL_ENV"

# Move to project directory
cd /project/def-arashmoh/shahab33/Medsam/selff-ref || {
    echo "❌ Failed to cd into project directory!"
    exit 1
}

# Print debug info
echo "Python executable: $(which python)"
python -V
echo "----------------------------------------"

# Set environment
export CUDA_VISIBLE_DEVICES=0
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

echo ""
echo "Testing on FULL PH2 dataset (all 5 splits, ~200 samples total)..."
echo "This will take ~2-3 hours instead of 24+ hours for all datasets"
echo ""

# Test Rule-Based on all PH2 splits
echo "🔧 [1/2] Testing RULE-BASED refinement on all PH2 splits..."
echo "========================================="

for split in 0 1 2 3 4; do
    echo ""
    echo "--- Processing PH2 Split $split (Rule-Based) ---"
    python full_datasets_violation_comparision_with_llm.py \
      --dataset PH2 \
      --split $split \
      --data_path $DATA_PATH \
      2>&1 | tee -a logs/debug_ph2_rulebased_full.log
    
    echo "Split $split completed at $(date)"
done

echo ""
echo "========================================="
echo ""

# Test LLM-Based on all PH2 splits
echo "🤖 [2/2] Testing LLM-BASED refinement on all PH2 splits..."
echo "========================================="

for split in 0 1 2 3 4; do
    echo ""
    echo "--- Processing PH2 Split $split (LLM-Based) ---"
    python full_datasets_violation_comparision_with_llm.py \
      --dataset PH2 \
      --split $split \
      --data_path $DATA_PATH \
      --use_llm \
      --llm_model MMed \
      2>&1 | tee -a logs/debug_ph2_llm_full.log
    
    echo "Split $split completed at $(date)"
done

echo ""
echo "========================================="
echo "✅ DEBUG TEST COMPLETE - FULL PH2 DATASET"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  Rule-Based:"
for split in 0 1 2 3 4; do
    echo "    - results/path_d_full_evaluation/PH2_split_${split}_rulebased_full_results.json"
done
echo ""
echo "  LLM-Based:"
for split in 0 1 2 3 4; do
    echo "    - results/path_d_full_evaluation/PH2_split_${split}_mmed_full_results.json"
done
echo ""
echo "Logs:"
echo "  - logs/debug_ph2_rulebased_full.log"
echo "  - logs/debug_ph2_llm_full.log"
echo "  - logs/slurm-${SLURM_JOB_ID}.out"
echo ""
echo "========================================="
echo "📊 NEXT STEPS:"
echo "========================================="
echo "1. Check logs for bugs (oscillation, format errors)"
echo "2. If results look good (>0% reduction), run on Derm7pt & HAM10000"
echo "3. Use compare_rulebased_vs_llm.py to generate figures"
echo ""
echo "To run on all datasets:"
echo "  sbatch scripts/run_pathd_rulebased.sh"
echo "  sbatch scripts/run_pathd_llm.sh"
echo "========================================="
