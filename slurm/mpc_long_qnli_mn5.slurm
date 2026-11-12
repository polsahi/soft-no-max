#!/bin/bash
#SBATCH --job-name=snm_longq
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_debug
#SBATCH --account=bsc28

# Long-input secure run on a trained model: the 48 QNLI validation pairs (seeded) with >= 90 tokens, for the g<=3 seed-0
# QK-norm model and the vanilla model (CrypTen max, SHAFT ODE). Step 1 exports on CPU (HF environment), step 2 runs the
# secure inference (SHAFT environment).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/mpc export

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=/gpfs/scratch/bsc28/$USER/huggingface
python3 mpc_bert_export.py --task qnli --data glue/qnli --ckpt ckpt/bert_qknorm_qnli_g30 --variant qknorm --g_max 3.0 --n 48 --min_tokens 90 \
    --out export/bert_qknorm_g30_qnli_long.pt
python3 mpc_bert_export.py --task qnli --data glue/qnli --ckpt ckpt/bert_vanilla_qnli --variant vanilla --n 48 --min_tokens 90 \
    --out export/bert_vanilla_qnli_long.pt

module purge
module load impi intel hdf5 mkl python/3.11.5-gcc
unset PYTHONPATH PYTHONHOME
VENV=$SLURM_SUBMIT_DIR/venv_shaft
export PYTHONPATH="$SLURM_SUBMIT_DIR/SHAFT"
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28
cd SHAFT/examples/unit-test
P="$VENV/bin/python $SLURM_SUBMIT_DIR/mpc_bert_sst2.py"
$P --export $SLURM_SUBMIT_DIR/export/bert_qknorm_g30_qnli_long.pt --method snm_shifted --out $SLURM_SUBMIT_DIR/results/mpc/mpc_bert_qnlilong_qknorm_g30.csv
$P --export $SLURM_SUBMIT_DIR/export/bert_vanilla_qnli_long.pt --method crypten_max --out $SLURM_SUBMIT_DIR/results/mpc/mpc_bert_qnlilong_vanilla_max.csv
$P --export $SLURM_SUBMIT_DIR/export/bert_vanilla_qnli_long.pt --method shaft_ode --out $SLURM_SUBMIT_DIR/results/mpc/mpc_bert_qnlilong_vanilla_ode.csv
echo "[job] DONE"
