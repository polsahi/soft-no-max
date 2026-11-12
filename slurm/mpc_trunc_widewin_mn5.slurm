#!/bin/bash
#SBATCH --job-name=snm_truncwin
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_debug
#SBATCH --account=bsc28

# (1) Truncation hypothesis. CrypTen's two-party truncation is probabilistic and fails with probability about |x_fixed|/2^64,
#     so raising the fixed-point precision multiplies the failure rate by 2^(bits-16). If the rare garbage logits come from
#     it, 40 SST-2 sentences at 20 and 24 bits should show garbage far more often than the 0 in 400 seen at 16 bits.
# (2) Wider reciprocal window: worst-case rows for g = 3, 5.5, 8 with 20 Newton iterations (and 24-bit precision).
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
for B in 20 24; do
  $P --export $SLURM_SUBMIT_DIR/export/bert_qknorm_g30_sst2.pt --method snm_shifted --precision_bits $B --start 0 --end 40 \
     --out $SLURM_SUBMIT_DIR/results/mpc/trunc_qknorm_p$B.csv
  $P --export $SLURM_SUBMIT_DIR/export/bert_vanilla_sst2.pt --method crypten_max --precision_bits $B --start 0 --end 40 \
     --out $SLURM_SUBMIT_DIR/results/mpc/trunc_vanilla_max_p$B.csv
done
for CFG in "20 16" "10 24"; do
  set -- $CFG
  mkdir -p $SLURM_SUBMIT_DIR/results/mpc/wide_nr$1_p$2
  $VENV/bin/python $SLURM_SUBMIT_DIR/mpc_bound_rows.py --gs 3.0,5.5,8.0 --ns 32,128,512 --reps 32 --rows 8 \
     --nr_iters $1 --precision_bits $2 --out_dir $SLURM_SUBMIT_DIR/results/mpc/wide_nr$1_p$2
done
echo "[job] DONE"
