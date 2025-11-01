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
echo "Self-Refine Test - Option 1: Quick Violation Comparison"
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

# Create logs directory
mkdir -p logs

# Set data path
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

echo "========================================="
echo "Testing Self-Refine on PH2 Dataset"
echo "Testing 10 samples to compare violations"
echo "========================================="
echo ""

# Run Option 1 test using existing script name
python "violation comparison.py" \
    --dataset PH2 \
    --split 0 \
    --num_samples 10

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Test completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully!"
    echo ""
    echo "RESULTS INTERPRETATION:"
    echo "- If violations decreased by >30%, self-refine is working well"
    echo "- If violations decreased by 10-30%, moderate improvement"
    echo "- If violations decreased by <10%, minimal improvement"
    echo ""
    echo "Next steps:"
    echo "  1. Check the output above for violation reduction percentage"
    echo "  2. If reduction >30%, run Option 2: sbatch test_option2_selfrefine.sh"
    echo "  3. If reduction <10%, consider adjusting consistency rules"
else
    echo "✗ Test failed with exit code: $EXIT_CODE"
    echo "Check the error messages above for details"
fi
