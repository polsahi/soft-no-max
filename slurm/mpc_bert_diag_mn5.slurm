#!/bin/bash
#SBATCH --job-name=snm_diag
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_debug
#SBATCH --account=bsc28

# Re-run the two garbage-logit sentences alone: QK-norm g<=3 sentence at position 116 (idx 483), vanilla CrypTen-max
# sentence at position 151 (idx 700). 3 repeats each (fresh randomness), first repeat with --debug activation ranges.
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
  X=""; [ $rep = 1 ] && X="--debug"
  echo "=== qknorm pos116 rep $rep"; run export/bert_qknorm_g30_sst2.pt snm_shifted 116 results/mpc/diag_qknorm_116.csv "$X"
  echo "=== vanilla max pos151 rep $rep"; run export/bert_vanilla_sst2.pt crypten_max 151 results/mpc/diag_vanilla_max_151.csv "$X"
done
echo "[job] DONE"
