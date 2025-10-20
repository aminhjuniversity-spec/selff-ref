#!/bin/bash
#SBATCH --job-name=selfrefine_test
#SBATCH --account=def-arashmoh
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/selfrefine_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Medsam/selff-ref/logs/selfrefine_%j.err

# Load modules
module load python/3.10
module load cuda/11.8

# Activate environment
source /project/def-arashmoh/shahab33/Medsam/selff-ref/selfrefine_env/bin/activate

# Change to project directory
cd /project/def-arashmoh/shahab33/Medsam/selff-ref

# Create directories
mkdir -p logs
mkdir -p results/selfrefine_ph2

echo "========================================="
echo "ExpLICD WITH Self-Refine on PH2"
echo "Job started: $(date)"
echo "========================================="

# Set data path
export DATA_PATH="/project/def-arashmoh/shahab33/Medsam/selff-ref/data"

# IMPORTANT: Keep use_self_refine = True in run_x_to_c_to_y.py (current setting)

# Run all 5 splits of PH2
for split in 0 1 2 3 4; do
    echo "----------------------------------------"
    echo "Processing PH2 split $split (WITH Self-Refine)"
    echo "----------------------------------------"
    
    python run_x_to_c_to_y.py \
        --model Explicd \
        --dataset PH2 \
        --split $split \
        --concept_extractor Explicd \
        --concept_reference_dict PH2 \
        --n_demos 0 \
        --data_path $DATA_PATH \
        2>&1 | tee -a logs/selfrefine_split_${split}.log
    
    echo "Split $split completed at $(date)"
done

echo "========================================="
echo "Job completed: $(date)"
echo "========================================="
```

### **Option 2: Baseline WITHOUT Self-Refine**
This is for comparison to see how much Self-Refine improves results.

## 📊 **Why Test Both?**

For your paper, you'll want to show:
1. **Baseline performance** (ExpLICD alone)
2. **Improved performance** (ExpLICD + Self-Refine)
3. **The improvement** (how much Self-Refine helps)

## ✅ **What You Should Do:**

Since you want to test your Self-Refine implementation:

1. **Keep** `use_self_refine = True` in `run_x_to_c_to_y.py` (lines 149 & 233)
2. **Use the "WITH Self-Refine" script above**
3. **Save it as `main.sh`**
4. **Run it**: `sbatch main.sh`

This will test your Self-Refine improvements! You'll see output like:
```
Image IMD001: 2 → 0 violations
Image IMD002: 1 → 0 violations
