#!/bin/bash
#SBATCH --job-name=test_selfrefine_opt2
#SBATCH --account=def-arashmoh
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/test_option2_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/test_option2_%j.err

echo "========================================="
echo "Self-Refine Test - Option 2: Full Report Generation"
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

# Verify environment
echo ""
echo "Environment Check:"
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "⚠ PyTorch not found"
python -c "import pandas; print('✓ Pandas: OK')" || echo "⚠ Pandas not found"
echo ""

# Change to project directory
cd /project/def-arashmoh/shahab33/Medsam/selff-ref

# Create directories
mkdir -p logs
mkdir -p results/comparison_reports

# Set data path
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

echo "========================================="
echo "Generating Full Concept Reports"
echo "Comparing: Baseline vs Self-Refine"
echo "Dataset: PH2, Split: 0"
echo "========================================="
echo ""

# Run Option 2 test using existing script name
python "comparison.py" \
    --dataset PH2 \
    --split 0 \
    --data_path $DATA_PATH \
    --output_dir results/comparison_reports

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Report Generation Completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Reports generated successfully!"
    echo ""
    echo "OUTPUT FILES:"
    echo "  - results/comparison_reports/baseline_concepts.csv"
    echo "  - results/comparison_reports/refined_concepts.csv"
    echo "  - results/comparison_reports/comparison_summary.txt"
    echo ""
    echo "NEXT STEPS:"
    echo "  1. Download the CSV files to analyze differences"
    echo "  2. Check comparison_summary.txt for statistics"
    echo "  3. Look for patterns in which concepts improved"
else
    echo "✗ Report generation failed with exit code: $EXIT_CODE"
    echo "Check the error messages above for details"
fi
