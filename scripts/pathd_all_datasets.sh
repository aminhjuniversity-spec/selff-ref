#!/bin/bash
#SBATCH --job-name=pathd_all_datasets
#SBATCH --account=def-arashmoh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1 
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/pathd_all_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/pathd_all_%j.err

echo "========================================="
echo "PATH D: COMPREHENSIVE EVALUATION"
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
echo "This script will evaluate Path D on ALL datasets:"
echo "  - PH2 (5 splits)"
echo "  - Derm7pt (1 dataset)"
echo "  - HAM10000 (1 dataset)"
echo ""
echo "Total: 7 evaluations on COMPLETE test sets"
echo ""

# Run the comprehensive evaluation script
python full_dataset_violation_comparison.py \
    --dataset all \
    --data_path $DATA_PATH \
    2>&1 | tee logs/pathd_comprehensive.log

echo ""
echo "========================================="
echo "COMPREHENSIVE EVALUATION COMPLETED"
echo "Completed: $(date)"
echo "========================================="
echo ""
echo "Results location:"
echo "  Individual: results/path_d_full_evaluation/*_full_results.json"
echo "  Consolidated: results/path_d_full_evaluation/CONSOLIDATED_RESULTS.json"
echo ""
echo "Logs location:"
echo "  logs/pathd_comprehensive.log"
echo ""
echo "Next steps:"
echo "  1. Download results for analysis"
echo "  2. Generate figures for paper"
echo "  3. Write Results section"
