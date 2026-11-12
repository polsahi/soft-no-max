#!/bin/bash
#SBATCH --job-name=snm_g4sec_a
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_debug
#SBATCH --account=bsc28

# Secure 2-party CrypTen inference of the g=4 / g=4.5 QK-norm BERT models (shifted softmax, public rescale c = max(1, N/64)).
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
P="$VENV/bin/python $SLURM_SUBMIT_DIR/mpc_bert_sst2.py"
E=$SLURM_SUBMIT_DIR/export; R=$SLURM_SUBMIT_DIR/results/mpc
$P --export $E/bert_qknorm_g40_sst2.pt --method snm_shifted --out $R/mpc_bert_sst2_qknorm_g40.csv
$P --export $E/bert_qknorm_g45_qnli.pt --method snm_shifted --end 100 --out $R/mpc_bert_qnli_qknorm_g45.csv
echo "[job] DONE"
