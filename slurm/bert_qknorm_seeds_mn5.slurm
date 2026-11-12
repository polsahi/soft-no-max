#!/bin/bash
#SBATCH --job-name=snm_bertseeds
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# Is the 0.57-point SST-2 drop of scalar QK-norm BERT (g <= 5.5) real or seed noise? Three seeds
# each of vanilla and QK-norm, with the QK-norm gains on their own learning rate (1e-3), since at
# 2e-5 they never moved from their initial value. SST-2 validation has only 872 sentences.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=${HF_CACHE_DIR:-$HOME/huggingface}

OUT=results/bert_qknorm_sst2_seeds.csv
for SEED in 0 1 2; do
  python3 bert_qknorm_glue.py --model bert-base-uncased --task sst2 --data glue/sst2 \
      --variant vanilla --epochs 3 --seed $SEED --out $OUT
  python3 bert_qknorm_glue.py --model bert-base-uncased --task sst2 --data glue/sst2 \
      --variant qknorm --g_max 5.5 --g_lr 1e-3 --epochs 3 --seed $SEED --out $OUT
done
cat $OUT
echo "[job] DONE"
