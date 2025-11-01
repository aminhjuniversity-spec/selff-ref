#!/bin/bash
#SBATCH --job-name=test_selfrefine_quick
#SBATCH --account=def-arashmoh
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/test_selfrefine_quick_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/test_selfrefine_quick_%j.err

# Load modules (as per cluster documentation)
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
mkdir -p logs
mkdir -p results/self_refine_comparison

echo "========================================="
echo "OPTION 1: Quick Self-Refine Test"
echo "Testing on 10 samples per split"
echo "Job started: $(date)"
echo "========================================="

# Set data path
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

# Test on split 0 only (or modify to test all splits)
for split in 0; do
    echo "----------------------------------------"
    echo "Testing Self-Refine on PH2 split $split"
    echo "----------------------------------------"
    
    python test_option1_violation_comparison.py \
        --dataset PH2 \
        --split $split \
        --num_samples 10 \
        2>&1 | tee -a logs/test_selfrefine_split_${split}.log
    
    echo "Split $split test completed at $(date)"
    echo ""
done

echo "========================================="
echo "SUMMARY: Check the output above"
echo "If violation reduction > 30%, proceed with full pipeline"
echo "Job completed: $(date)"
echo "========================================="
