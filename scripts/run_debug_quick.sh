#!/bin/bash
#SBATCH --job-name=debug_pathd
#SBATCH --account=def-arashmoh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G 
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

echo "========================================="
echo "QUICK DEBUG TEST - Path D + x->c->y"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

# Create logs and results directories (safe even if they exist)
mkdir -p logs
mkdir -p results/job_${SLURM_JOB_ID}

# --------------------------------------------
# 1️⃣ Activate your virtual environment
# --------------------------------------------
source /project/def-arashmoh/shahab33/Medsam/self/bin/activate
echo "Activated environment: $VIRTUAL_ENV"

# --------------------------------------------
# 2️⃣ Move into your project directory
# --------------------------------------------
cd /project/def-arashmoh/shahab33/Medsam/selff-ref || {
    echo "❌ Failed to cd into project directory!"
    exit 1
}

# --------------------------------------------
# 3️⃣ Print debug info (optional but useful)
# --------------------------------------------
echo "Python executable: $(which python)"
python -V
python -m site
echo "----------------------------------------"
python -m pip show filelock || echo "⚠️ filelock not found"
echo "----------------------------------------"

# --------------------------------------------
# 4️⃣ Run your Python script
# --------------------------------------------
export CUDA_VISIBLE_DEVICES=0
echo "Running debug tests on PH2 split 0 (20 samples)..."
python debug_pathd_quick.py \
  --test all \
  --max_samples 20

echo "Python exit code: $?"
echo "========================================="
echo "DEBUG COMPLETE for Job ${SLURM_JOB_ID}"
echo "Results in: results/job_${SLURM_JOB_ID}"
echo "Logs in: logs/slurm-${SLURM_JOB_ID}.out / .err"
echo "========================================="
