#!/bin/bash
#SBATCH --job-name=test_selfrefine_full
#SBATCH --account=def-arashmoh
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/test_selfrefine_full_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/test_selfrefine_full_%j.err

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
echo "OPTION 2: Full Self-Refine Report Generation"
echo "Generating complete reports for entire dataset"
echo "Job started: $(date)"
echo "========================================="

# Set data path
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

# Generate reports for split 0 (modify to test all splits if needed)
for split in 0; do
    echo "----------------------------------------"
    echo "Generating Self-Refine Reports for PH2 split $split"
    echo "----------------------------------------"
    
    python test_option2_report_comparison.py \
        --dataset PH2 \
        --split $split \
        --data_path $DATA_PATH \
        2>&1 | tee -a logs/test_selfrefine_reports_split_${split}.log
    
    echo "Split $split report generation completed at $(date)"
    echo ""
    
    # Show results location
    echo "Results saved to:"
    echo "  - results/self_refine_comparison/PH2_split_${split}/concepts_baseline.csv"
    echo "  - results/self_refine_comparison/PH2_split_${split}/concepts_refined.csv"
    echo "  - results/self_refine_comparison/PH2_split_${split}/concepts_comparison.csv"
    echo ""
done

echo "========================================="
echo "Job completed: $(date)"
echo "Check the CSV files for detailed comparison"
echo "========================================="
