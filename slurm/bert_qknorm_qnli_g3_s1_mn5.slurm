#!/bin/bash
#SBATCH --job-name=snm_qnli_s1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# QNLI (question + sentence pairs, longer than SST-2): train a second seed (seed 1) of the g<=3 QK-norm BERT-base with the Table-GLUE
# recipe (3 epochs, lr 2e-5, gains 1e-3, seed 0), save both, and export 200 validation pairs for secure inference.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results ckpt export
module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=${HF_CACHE_DIR:-$HOME/huggingface}

python3 bert_qknorm_glue.py --model bert-base-uncased --task qnli --data glue/qnli \
    --variant qknorm --g_max 3.0 --g_lr 1e-3 --epochs 3 --seed 1 --out results/bert_qnli_qknorm_g30_s1_ckpt.csv --save ckpt/bert_qknorm_qnli_g30_s1
python3 mpc_bert_export.py --task qnli --data glue/qnli --ckpt ckpt/bert_qknorm_qnli_g30_s1 --variant qknorm --g_max 3.0 --n 200 \
    --out export/bert_qknorm_g30_qnli_s1.pt
echo "[job] DONE"
