#!/bin/bash
#SBATCH --job-name=snm_g3var
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# Is the 2.9-point SST-2 loss at g <= 3 partly the single untuned conversion recipe? Variants, all g_max 3.0, gains at 1e-3,
# 3 seeds each (baseline recipe: bert_qknorm_sst2_gmax30.csv, mean 89.95 +- 0.63):
#   A  10 epochs                      B  KD from the vanilla teacher, 3 epochs
#   C  KD, 10 epochs                  D  body lr 5e-5, 3 epochs
# NB: SST-2 validation is the evaluation set, so these are a sensitivity analysis, not a tuned result.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=${HF_CACHE_DIR:-$HOME/huggingface}

OUT=results/bert_qknorm_sst2_g30_variants.csv
COMMON="--model bert-base-uncased --task sst2 --data glue/sst2 --variant qknorm --g_max 3.0 --g_lr 1e-3 --out $OUT"
for SEED in 0 1 2; do
  python3 bert_qknorm_glue.py $COMMON --epochs 3  --lr 2e-5 --teacher ckpt/bert_vanilla_sst2_hf --seed $SEED   # B
  python3 bert_qknorm_glue.py $COMMON --epochs 3  --lr 5e-5 --seed $SEED                                      # D
  python3 bert_qknorm_glue.py $COMMON --epochs 10 --lr 2e-5 --seed $SEED                                      # A
  python3 bert_qknorm_glue.py $COMMON --epochs 10 --lr 2e-5 --teacher ckpt/bert_vanilla_sst2_hf --seed $SEED  # C
done
cat $OUT
echo "[job] DONE"
