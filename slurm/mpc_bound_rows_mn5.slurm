#!/bin/bash
#SBATCH --job-name=snm_boundrows
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_debug
#SBATCH --account=bsc28

# Item 3: public rescale of the row sum in the bounded ("shifted") secure softmax.
#   1. patch SHAFT (shifted softmax with softmax_shift_rescale; config key), save the diff
#   2. measure the reciprocal's window on a fine grid, then worst-case and typical rows for
#      g in {3.0, 5.5} and N in {32..512}, rescale off and on

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/mpc

module purge
module load impi intel hdf5 mkl python/3.11.5-gcc
unset PYTHONPATH PYTHONHOME
VENV=$SLURM_SUBMIT_DIR/venv_shaft
export PYTHONPATH="$SLURM_SUBMIT_DIR/SHAFT"
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28

"$VENV/bin/python" shaft_apply_patches.py SHAFT
# shaft_apply_patches.py is the record of the change (this SHAFT copy is not a git checkout, where
# "git diff a b" would compare the two files with each other)
if (cd SHAFT && git rev-parse --is-inside-work-tree >/dev/null 2>&1); then
    (cd SHAFT && git diff -- crypten/common/functions/approximations.py configs/default.yaml) > shaft_softnomax.patch
fi

cd SHAFT/examples/unit-test
"$VENV/bin/python" "$SLURM_SUBMIT_DIR/mpc_bound_rows.py" --gs 3.0,5.5 --ns 32,64,128,256,512 \
    --out_dir "$SLURM_SUBMIT_DIR/results/mpc"
echo "[job] DONE"
