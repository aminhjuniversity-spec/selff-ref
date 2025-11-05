#!/bin/bash
#SBATCH --job-name=pathd_derm7pt
#SBATCH --account=def-arashmoh
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1 
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/pathd_derm7pt_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/pathd_derm7pt_%j.err

echo "========================================="
echo "PATH D: Derm7pt Full Evaluation"
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

# Create output directory
mkdir -p results/path_d_full_evaluation
mkdir -p logs

# Set data path
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

echo ""
echo "Running Path D evaluation on Derm7pt..."
echo "Processing COMPLETE test set"
echo ""

python full_dataset_violation_comparison.py \
    --dataset Derm7pt \
    --data_path $DATA_PATH \
    2>&1 | tee logs/pathd_derm7pt.log

echo ""
echo "========================================="
echo "Derm7pt completed: $(date)"
echo "========================================="
echo ""
echo "Results saved in: results/path_d_full_evaluation/Derm7pt_full_results.json"
