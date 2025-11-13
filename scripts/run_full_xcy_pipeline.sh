#!/bin/bash
#SBATCH --job-name=xcy_selfrefine
#SBATCH --account=def-arashmoh
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/xcy_pipeline_%j.out
#SBATCH --error=logs/xcy_pipeline_%j.err

echo "========================================="
echo "🚀 FULL x→c→y PIPELINE WITH SELF-REFINE"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "========================================="

# Load modules
module load gcc python/3.11 cuda/12.6 opencv scipy-stack

# Activate environment
source /project/def-arashmoh/shahab33/Medsam/self/bin/activate
echo "✓ Environment activated: $VIRTUAL_ENV"

# Go to project directory
cd /project/def-arashmoh/shahab33/Medsam/selff-ref || exit 1

# Create directories
mkdir -p logs results/concept_prediction results/label_prediction

# Set paths
export CUDA_VISIBLE_DEVICES=0
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

echo ""
echo "📊 Processing all datasets with self-refine..."
echo ""

# ============================================
# DATASET 1: PH2 (5 splits)
# ============================================
echo "========================================="
echo "[1/3] PH2 Dataset - 5 Splits"
echo "========================================="

for split in 0 1 2 3 4; do
    echo ""
    echo "--- PH2 Split $split ---"
    
    # Step 1: x→c (generate refined concepts)
    echo "  [x→c] Generating refined concepts..."
    python run_x_to_c_to_y.py \
        --dataset PH2 \
        --split $split \
        --generate_concepts \
        --data_path $DATA_PATH \
        2>&1 | tee -a logs/ph2_split_${split}_xc.log
    
    # Step 2: c→y (concepts to diagnosis)
    echo "  [c→y] Predicting diagnosis..."
    python run_x_to_c_to_y.py \
        --dataset PH2 \
        --split $split \
        --llm MMed \
        --ckpt Henrychur/MMed-Llama-3-8B \
        --n_demos 0 \
        --data_path $DATA_PATH \
        2>&1 | tee -a logs/ph2_split_${split}_cy.log
    
    echo "  ✓ Split $split completed at $(date)"
done

echo ""
echo "✓ PH2 completed!"
echo ""

# ============================================
# DATASET 2: Derm7pt
# ============================================
echo "========================================="
echo "[2/3] Derm7pt Dataset"
echo "========================================="

# Step 1: x→c
echo "  [x→c] Generating refined concepts..."
python run_x_to_c_to_y.py \
    --dataset Derm7pt \
    --generate_concepts \
    --data_path $DATA_PATH \
    2>&1 | tee logs/derm7pt_xc.log

# Step 2: c→y
echo "  [c→y] Predicting diagnosis..."
python run_x_to_c_to_y.py \
    --dataset Derm7pt \
    --llm MMed \
    --ckpt Henrychur/MMed-Llama-3-8B \
    --n_demos 0 \
    --data_path $DATA_PATH \
    2>&1 | tee logs/derm7pt_cy.log

echo ""
echo "✓ Derm7pt completed!"
echo ""

# ============================================
# DATASET 3: HAM10000
# ============================================
echo "========================================="
echo "[3/3] HAM10000 Dataset"
echo "========================================="

# Step 1: x→c
echo "  [x→c] Generating refined concepts..."
python run_x_to_c_to_y.py \
    --dataset HAM10000 \
    --generate_concepts \
    --data_path $DATA_PATH \
    2>&1 | tee logs/ham10000_xc.log

# Step 2: c→y
echo "  [c→y] Predicting diagnosis..."
python run_x_to_c_to_y.py \
    --dataset HAM10000 \
    --llm MMed \
    --ckpt Henrychur/MMed-Llama-3-8B \
    --n_demos 0 \
    --data_path $DATA_PATH \
    2>&1 | tee logs/ham10000_cy.log

echo ""
echo "✓ HAM10000 completed!"
echo ""

# ============================================
# SUMMARY
# ============================================
echo "========================================="
echo "✅ COMPLETE PIPELINE FINISHED"
echo "========================================="
echo "Completed: $(date)"
echo ""
echo "📁 Results saved in:"
echo "  Concepts: results/concept_prediction/"
echo "  Diagnosis: results/label_prediction/"
echo ""
echo "📊 To view results:"
echo "  cd results/label_prediction"
echo "  ls -lh"
echo ""
echo "📈 Next: Calculate metrics with:"
echo "  python calculate_metrics.py --model=MMed --task=PH2_eval ..."
echo "========================================="
