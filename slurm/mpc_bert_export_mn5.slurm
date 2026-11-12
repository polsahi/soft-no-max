#!/bin/bash
#SBATCH --job-name=snm_export
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# Re-export the SST-2 checkpoints (QK-norm g<=3 and vanilla) with calibrated public inverse-sqrt scales.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs export
module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=${HF_CACHE_DIR:-$HOME/huggingface}
python3 mpc_bert_export.py --ckpt ckpt/bert_qknorm_sst2_g30 --variant qknorm --g_max 3.0 --n 200 --out export/bert_qknorm_g30_sst2.pt
python3 mpc_bert_export.py --ckpt ckpt/bert_vanilla_sst2 --variant vanilla --n 200 --out export/bert_vanilla_sst2.pt
echo "[job] DONE"
