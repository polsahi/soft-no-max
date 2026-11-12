#!/bin/bash
#SBATCH --job-name=snm_diag2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_debug
#SBATCH --account=bsc28

# Re-run QNLI garbage sentences alone: QK-norm pos 28 (idx 3852), vanilla ODE pos 37 (idx 566); 3 repeats each.
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
run() { # export method pos out extra
  "$VENV/bin/python" "$SLURM_SUBMIT_DIR/mpc_bert_sst2.py" --export "$SLURM_SUBMIT_DIR/$1" --method "$2" --start "$3" --end "$(($3+1))" $5 --out "$SLURM_SUBMIT_DIR/$4"
}
for rep in 1 2 3; do
  echo "=== qknorm qnli pos28 rep $rep"; run export/bert_qknorm_g30_qnli.pt snm_shifted 28 results/mpc/diag_qnli_qknorm_3852.csv ""
  echo "=== vanilla ode qnli pos37 rep $rep"; run export/bert_vanilla_qnli.pt shaft_ode 37 results/mpc/diag_qnli_ode_566.csv ""
done
echo "[job] DONE"
