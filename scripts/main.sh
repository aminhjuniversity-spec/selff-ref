#!/bin/bash
#SBATCH --job-name=baseline_explicd
#SBATCH --account=def-arashmoh
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/baseline_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/baseline_%j.err

# Load modules
module load python/3.10
module load cuda/11.8

# Activate environment
source /project/def-arashmoh/shahab33/Medsam/selff-ref/selfrefine_env/bin/activate

# Change to project directory
cd /project/def-arashmoh/shahab33/Medsam/selff-ref

# Create directories
mkdir -p logs
mkdir -p results/baseline_ph2

echo "========================================="
echo "Baseline ExpLICD (No Self-Refine) on PH2"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "========================================="

# Set data path
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

# IMPORTANT: Temporarily disable Self-Refine in the code
# You need to set use_self_refine = False in run_x_to_c_to_y.py

# Run all 5 splits of PH2
for split in 0 1 2 3 4; do
    echo "----------------------------------------"
    echo "Processing PH2 split $split (Baseline)"
    echo "----------------------------------------"
    
    python run_x_to_c_to_y.py \
        --model Explicd \
        --dataset PH2 \
        --split $split \
        --concept_extractor Explicd \
        --n_demos 0 \
        --data_path $DATA_PATH \
        --no_self_refine \
        --output_dir results/baseline_ph2/split_${split} \
        2>&1 | tee -a logs/baseline_split_${split}.log
    
    echo "Split $split completed at $(date)"
done

echo "========================================="
echo "Computing baseline metrics..."
echo "========================================="

# Calculate average metrics
python calculate_metrics.py \
    --results_dir results/baseline_ph2 \
    --dataset PH2

echo "========================================="
echo "Job completed: $(date)"
echo "Total time: $SECONDS seconds"
echo "========================================="
