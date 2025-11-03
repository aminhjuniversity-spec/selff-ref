#!/bin/bash
#SBATCH --job-name=test_selfrefine_opt1
#SBATCH --account=def-arashmoh
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/test_option1_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/test_option1_%j.err

echo "========================================="
echo "Self-Refine Test - Option 1"
echo "Job started: $(date)"
echo "========================================="

module load gcc python/3.11 cuda/12.6 opencv/4.12.0 scipy-stack
source /project/def-arashmoh/shahab33/Medsam/self/bin/activate

# CRITICAL: Enable offline mode
export HF_HOME=$HOME/scratch/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /project/def-arashmoh/shahab33/Medsam/selff-ref
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

python "violation comparison.py" --dataset PH2 --split 0 --num_samples 10

echo "========================================="
echo "Job completed: $(date)"
echo "========================================="
