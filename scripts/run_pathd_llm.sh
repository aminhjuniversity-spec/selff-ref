#!/bin/bash
#SBATCH --job-name=pathd_llm
#SBATCH --account=def-arashmoh
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1 
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/pathd_llm_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/pathd_llm_%j.err

echo "========================================="
echo "PATH D: LLM-BASED REFINEMENT (MMed-Llama-3-8B)"
echo "All Datasets - Full Test Sets"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "========================================="

# Load modules
module load gcc
module load python/3.11
module load cuda/12.6
module load opencv/4.12.0
module load scipy-stack

# Activate environment
source /project/def-arashmoh/shahab33/Medsam/self/bin/activate

# Change to project directory
cd /project/def-arashmoh/shahab33/Medsam/selff-ref

# Create directories
mkdir -p results/path_d_full_evaluation
mkdir -p logs

# Set data path
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

echo ""
echo "Running Path D with LLM-BASED refinement (MMed-Llama-3-8B)..."
echo "This uses MMed LLM running on cluster (FREE - no API costs)"
echo ""

# Run evaluation with LLM-BASED refinement using MMed
python full_datasets_violation_comparision_with_llm.py \
    --dataset all \
    --data_path $DATA_PATH \
    --use_llm \
    --llm_model MMed \
    2>&1 | tee logs/pathd_llm.log

echo ""
echo "========================================="
echo "LLM-BASED EVALUATION COMPLETED"
echo "Completed: $(date)"
echo "========================================="
echo ""
echo "Results location:"
echo "  Individual: results/path_d_full_evaluation/*_mmed_full_results.json"
echo "  Consolidated: results/path_d_full_evaluation/CONSOLIDATED_RESULTS_mmed.json"
echo ""
echo "Next: Run compare_rulebased_vs_llm.py to generate comparison figures!"
