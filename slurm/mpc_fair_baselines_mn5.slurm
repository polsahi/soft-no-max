#!/bin/bash
#SBATCH --job-name=snm_fair
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_debug
#SBATCH --account=bsc28

# Reviewer-requested fairness controls for the BERT-base cost table (single node, random weights, N = 128, 256):
#   crypten_max_rescale  CrypTen secure-max softmax WITH our public row-sum rescale
#   shaft_ode_narrow     SHAFT with clip [-6, 0] and 8 steps (cost only; accuracy not validated)
#   vanilla:snm_shifted  constant shift on a vanilla model, no QK-norm (cost only; carries no bound)
# plus the three original configurations in the same job for a like-for-like comparison.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/mpc
module purge
module load impi intel hdf5 mkl python/3.11.5-gcc
unset PYTHONPATH PYTHONHOME
VENV=$SLURM_SUBMIT_DIR/venv_shaft
export PYTHONPATH="$SLURM_SUBMIT_DIR/SHAFT"
export OMP_NUM_THREADS=56 MKL_NUM_THREADS=56
cd SHAFT/examples/unit-test
"$VENV/bin/python" "$SLURM_SUBMIT_DIR/mpc_encoder_cost.py" --arch bert --layers 12 --hidden 768 --heads 12 --seq_lens 128,256 \
    --configs vanilla:crypten_max,vanilla:crypten_max_rescale,vanilla:shaft_ode,vanilla:shaft_ode_narrow,vanilla:snm_shifted,qknorm:snm_shifted \
    --out "$SLURM_SUBMIT_DIR/results/mpc/mpc_fair_baselines.csv"
echo "[job] DONE"
