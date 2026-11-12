#!/bin/bash
#SBATCH --job-name=snm_sst2mpc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_debug
#SBATCH --account=bsc28

# Secure 2-party inference of a fine-tuned BERT-base SST-2 model. One softmax variant per job:
#   sbatch --export=ALL,EXPORT=export/bert_qknorm_g30_sst2.pt,METHOD=snm_shifted,OUT=results/mpc/mpc_bert_sst2_qknorm_g30.csv,START=0,END=200 mpc_bert_sst2_mn5.slurm
# EXPORT/METHOD/OUT are required; START/END select examples (default all). Rows are appended per example.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/mpc

module purge
module load impi intel hdf5 mkl python/3.11.5-gcc
unset PYTHONPATH PYTHONHOME
VENV=$SLURM_SUBMIT_DIR/venv_shaft
export PYTHONPATH="$SLURM_SUBMIT_DIR/SHAFT"
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28

cd SHAFT/examples/unit-test
"$VENV/bin/python" "$SLURM_SUBMIT_DIR/mpc_bert_sst2.py" --export "$SLURM_SUBMIT_DIR/$EXPORT" --method "$METHOD" \
    --start "${START:-0}" --end "${END:-1000000}" ${EXTRA:-} --out "$SLURM_SUBMIT_DIR/$OUT"
echo "[job] DONE"
