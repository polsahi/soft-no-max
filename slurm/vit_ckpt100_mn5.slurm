#!/bin/bash
#SBATCH --job-name=snm_vit100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# Retrain the CIFAR-10 ViTs (vanilla and scalar-gain QK-norm) and keep their weights: the
# secure-inference experiment (mpc_vit_private.py) needs trained checkpoints, and the per-head
# QK-norm gains g are its public softmax shifts. Same recipe as the original vit_qknorm run.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/ckpt100

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)

python3 -c "import torch; print('cuda_ok=', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
VIT_CKPT_DIR=results/ckpt100 python3 vit_qknorm_cifar.py data/cifar10.npz 100 results/vit_qknorm_ckpt100.csv
ls -la results/ckpt100

echo "[job] DONE"
