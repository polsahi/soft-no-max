#!/bin/bash
#SBATCH --job-name=snm_g4train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# Encoder-correct cap: with full-visibility rows the 16-bit window needs 2g + ln c <= 6.75 + ln k_min, so g = 4 (k_min >= 5)
# and g = 4.5 (k_min >= 12) are candidates. Train QK-norm BERT-base at g_max 4.0 and 4.5 on SST-2 and QNLI (baseline recipe,
# seed 0), save, and export the secure-inference sets (200 SST-2 sentences, 200 QNLI pairs, 48 long QNLI pairs).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results ckpt export
module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=/gpfs/scratch/bsc28/$USER/huggingface

for G in 4.0 4.5; do
  T=$(echo $G | tr -d .)
  python3 bert_qknorm_glue.py --model bert-base-uncased --task sst2 --data glue/sst2 --variant qknorm --g_max $G --g_lr 1e-3 \
      --epochs 3 --lr 2e-5 --seed 0 --out results/bert_g4_sst2_ckpt.csv --save ckpt/bert_qknorm_sst2_g$T
  python3 bert_qknorm_glue.py --model bert-base-uncased --task qnli --data glue/qnli --variant qknorm --g_max $G --g_lr 1e-3 \
      --epochs 3 --lr 2e-5 --seed 0 --out results/bert_g4_qnli_ckpt.csv --save ckpt/bert_qknorm_qnli_g$T
  python3 mpc_bert_export.py --task sst2 --data glue/sst2 --ckpt ckpt/bert_qknorm_sst2_g$T --variant qknorm --g_max $G --n 200 \
      --out export/bert_qknorm_g${T}_sst2.pt
  python3 mpc_bert_export.py --task qnli --data glue/qnli --ckpt ckpt/bert_qknorm_qnli_g$T --variant qknorm --g_max $G --n 200 \
      --out export/bert_qknorm_g${T}_qnli.pt
  python3 mpc_bert_export.py --task qnli --data glue/qnli --ckpt ckpt/bert_qknorm_qnli_g$T --variant qknorm --g_max $G --n 48 --min_tokens 90 \
      --out export/bert_qknorm_g${T}_qnli_long.pt
done
cat results/bert_g4_sst2_ckpt.csv results/bert_g4_qnli_ckpt.csv
echo "[job] DONE"
