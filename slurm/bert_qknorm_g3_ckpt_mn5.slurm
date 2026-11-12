#!/bin/bash
#SBATCH --job-name=snm_g3ckpt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# Train the g <= 3.0 QK-norm SST-2 model with the same recipe as bert_qknorm_g3_mn5.slurm (seed 0) but SAVE it,
# then export it and the vanilla SST-2 checkpoint (200 validation sentences, same indices) for secure inference.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results ckpt export

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=${HF_CACHE_DIR:-$HOME/huggingface}

python3 bert_qknorm_glue.py --model bert-base-uncased --task sst2 --data glue/sst2 \
    --variant qknorm --g_max 3.0 --g_lr 1e-3 --epochs 3 --seed 0 \
    --out results/bert_qknorm_sst2_gmax30_ckpt.csv --save ckpt/bert_qknorm_sst2_g30
python3 mpc_bert_export.py --ckpt ckpt/bert_qknorm_sst2_g30 --variant qknorm --g_max 3.0 --n 200 \
    --out export/bert_qknorm_g30_sst2.pt
python3 mpc_bert_export.py --ckpt ckpt/bert_vanilla_sst2 --variant vanilla --n 200 \
    --out export/bert_vanilla_sst2.pt
echo "[job] DONE"
