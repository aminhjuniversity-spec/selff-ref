#!/bin/bash
#SBATCH --job-name=debug_pathd
#SBATCH --account=def-arashmoh       # Replace with your account if needed
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00              # Adjust as needed
#SBATCH --mem=16G                    # Adjust based on model size
#SBATCH --output=logs/slurm-%j.out   # Save stdout using job number
#SBATCH --error=logs/slurm-%j.err    # Save stderr using job number

# ------------------- SETUP -------------------
echo "========================================="
echo "QUICK DEBUG TEST - Path D + x->c->y"
echo "Runtime: ~10 minutes"
echo "========================================="
echo ""

# Create necessary directories
mkdir -p logs
mkdir -p results/job_${SLURM_JOB_ID}

# Activate environment if needed
# module load python/3.10 cuda/12.1
# source ~/envs/myenv/bin/activate

export CUDA_VISIBLE_DEVICES=0

# ------------------- RUN ---------------------
echo "Running debug tests on PH2 split 0 (20 samples)..."
echo "Saving results to results/job_${SLURM_JOB_ID}"
echo ""

python debug_pathd_quick.py \
  --test all \
  --max_samples 20 \
  --output_dir results/job_${SLURM_JOB_ID}

# ------------------- COMPLETE ----------------
echo ""
echo "========================================="
echo "DEBUG COMPLETE for Job ${SLURM_JOB_ID}!"
echo "Results saved in: results/job_${SLURM_JOB_ID}"
echo "Logs saved in: logs/slurm-${SLURM_JOB_ID}.out / .err"
echo "========================================="
echo ""
echo "Next runs:"
echo "  sbatch scripts/run_pathd_rulebased.sh  (24h)"
echo "  sbatch scripts/run_pathd_llm.sh        (48h)"
echo "  sbatch scripts/main.sh                 (3h)"
