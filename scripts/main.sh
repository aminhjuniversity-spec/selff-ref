#!/bin/bash
#SBATCH --job-name=selfrefine_ph2
#SBATCH --account=def-arashmoh
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/selfrefine_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/selfrefine_%j.err

# Load modules
module load opencv/4.12.0
module load rust
module load python/3.11
module load cuda/12.6 

# Activate environment
source /project/def-arashmoh/shahab33/Medsam/self/bin/activate

 
# Change to project directory
cd /project/def-arashmoh/shahab33/Medsam/selff-ref

# Create directories
mkdir -p logs
mkdir -p results/concept_prediction
mkdir -p results/label_prediction
mkdir -p results/x_to_c_to_y

echo "========================================="
echo "ExpLICD WITH Self-Refine on PH2"
echo "Job started: $(date)"
echo "========================================="

# Set data path
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

# IMPORTANT: Self-refine is controlled in run_x_to_c_to_y.py line 57
# Make sure use_self_refine=True in the code before running

# Run all 5 splits
for split in 0 1 2 3 4; do
    echo "----------------------------------------"
    echo "Processing PH2 split $split (WITH Self-Refine)"
    echo "----------------------------------------"
    
    python run_x_to_c_to_y.py \
        --dataset PH2 \
        --split $split \
        --llm MMed \
        --ckpt Henrychur/MMed-Llama-3-8B \
        --generate_concepts \
        --n_demos 0 \
        --data_path $DATA_PATH \
        2>&1 | tee -a logs/selfrefine_split_${split}.log
    
    echo "Split $split completed at $(date)"
    echo ""
done

echo "========================================="
echo "Job completed: $(date)"
echo "========================================="
