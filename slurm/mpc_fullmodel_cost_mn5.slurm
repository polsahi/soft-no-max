#!/bin/bash
#SBATCH --job-name=snm_mpcfull
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_debug
#SBATCH --account=bsc28

# Point 3: full-model secure cost and a second architecture with a causal mask.
#   bert_full  whole BERT-base classifier (private token ids one-hot against the encrypted
#              embedding table, pooler, classifier) at 128 tokens
#   gpt2       GPT-2-small causal decoder (12 layers, 768, 12 heads) at 128 and 256 tokens
# Same three softmax variants as the encoder-only run; cost is weight-independent.

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
"$VENV/bin/python" "$SLURM_SUBMIT_DIR/mpc_encoder_cost.py" --arch bert_full --layers 12 --hidden 768 --heads 12 \
    --vocab 30522 --seq_lens 128 --out "$SLURM_SUBMIT_DIR/results/mpc/mpc_fullmodel_bert_full.csv"
"$VENV/bin/python" "$SLURM_SUBMIT_DIR/mpc_encoder_cost.py" --arch gpt2 --layers 12 --hidden 768 --heads 12 \
    --seq_lens 128,256 --out "$SLURM_SUBMIT_DIR/results/mpc/mpc_fullmodel_gpt2.csv"
echo "[job] DONE"
