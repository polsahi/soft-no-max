#!/bin/bash
#SBATCH --job-name=snm_gsweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# Accuracy against the gain cap g_max (reviewers asked for a sweep; g_max 3.0 and 5.5 exist already, baseline recipe: 3 epochs,
# lr 2e-5, gains at 1e-3). SST-2: g_max 4, 8, 12 x 3 seeds. QNLI: g_max 5.5, 8 x seed 0.
# NB: g_max above 3 is NOT fixed-point safe for CrypTen's 16-bit reciprocal; this only measures how much accuracy the cap costs.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=${HF_CACHE_DIR:-$HOME/huggingface}

OUT=results/bert_qknorm_g_sweep.csv
for G in 4.0 8.0 12.0; do
  for SEED in 0 1 2; do
    python3 bert_qknorm_glue.py --model bert-base-uncased --task sst2 --data glue/sst2 --variant qknorm --g_max $G --g_lr 1e-3 \
        --epochs 3 --lr 2e-5 --seed $SEED --out $OUT
  done
done
for G in 5.5 8.0; do
  python3 bert_qknorm_glue.py --model bert-base-uncased --task qnli --data glue/qnli --variant qknorm --g_max $G --g_lr 1e-3 \
      --epochs 3 --lr 2e-5 --seed 0 --out $OUT
done
cat $OUT
echo "[job] DONE"
