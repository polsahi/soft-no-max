#!/bin/bash
#SBATCH --job-name=snm_vitg3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# Does a provably bounded attention (hard cap g <= 3, trained in from scratch) cost accuracy?
# CIFAR-10 ViT, 100 epochs, same recipe as vit_ckpt100_mn5.slurm, vanilla vs scalar-gain QK-norm with g = 3*sigmoid(raw)
# per head. Three seeds. (The earlier g=4.48 model had an uncapped softplus gain.)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/ckpt_g3

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)

python3 -c "import torch; print('cuda_ok=', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
for SEED in 0 1 2; do
  VIT_G_MAX=3.0 VIT_SEED=$SEED VIT_TAG=_g3_s$SEED VIT_CKPT_DIR=results/ckpt_g3 \
    python3 vit_qknorm_cifar.py data/cifar10.npz 100 results/vit_qknorm_g3_s$SEED.csv
done
ls -la results/ckpt_g3
echo "[job] DONE"
