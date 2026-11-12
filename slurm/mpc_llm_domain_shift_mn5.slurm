#!/bin/bash
#SBATCH --job-name=snm_mpcllm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_bsccs
#SBATCH --account=bsc28

# Point 2: secure SmolLM2-135M inference under domain shift, empirical softmax constant vs secure max.
#   stage 1 (py3.12 + transformers): calibrate the per-layer constant on Wikitext, pick worst and
#            random windows from Wikitext (held out), MBPP code and Catalan, save plaintext reference
#   stage 2 (SHAFT CrypTen env): plaintext re-implementation check, then 2-party secure forwards

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/mpc

MODEL=/gpfs/scratch/bsc28/$USER/soft-no-max/offline_models/HuggingFaceTB--SmolLM2-135M-Instruct
DOMAINS=$HOME/lm-evaluation-harness/snm_check/domains
SEQ=${SEQ:-64}
DELTA=${DELTA:-1.0}
VARIANTS=${VARIANTS:-crypten_max,emp_layer,emp_head,emp_row,emp_row_nr20}
BITS=${BITS:-16,20}
PREPTAG=${PREPTAG:-v2}
EVAL=${EVAL:-wikitext,code_mbpp,multiling_ca}
WORST=${WORST:-2}
RANDOM_W=${RANDOM_W:-1}
PREP=/gpfs/scratch/bsc28/$USER/soft-no-max/llm_prep_smollm2_n${SEQ}_d${DELTA}_${PREPTAG}.pt

if [ ! -s "$PREP" ]; then
    module purge
    module load impi intel hdf5 mkl python/3.12.1-gcc
    export PYTHONPATH=$HOME/.local HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=56
    python3 llm_domain_prep.py --model "$MODEL" --domains "$DOMAINS" --calib wikitext_calib \
        --eval "$EVAL" --worst "$WORST" --random "$RANDOM_W" --seq_len "$SEQ" --delta "$DELTA" --select row \
        --stats_csv "results/mpc/llm_domain_stats_n${SEQ}_d${DELTA}_${PREPTAG}.csv" --out "$PREP"
fi

module purge
module load impi intel hdf5 mkl python/3.11.5-gcc
unset PYTHONPATH PYTHONHOME
VENV=$SLURM_SUBMIT_DIR/venv_shaft
export PYTHONPATH="$SLURM_SUBMIT_DIR/SHAFT"
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28

cd SHAFT/examples/unit-test
"$VENV/bin/python" "$SLURM_SUBMIT_DIR/mpc_llm_domain_shift.py" --prep "$PREP" --variants "$VARIANTS" \
    --precision_bits "$BITS" --out "$SLURM_SUBMIT_DIR/results/mpc/mpc_llm_domain_shift_n${SEQ}_d${DELTA}_${PREPTAG}${TAG:-}.csv"
echo "[job] DONE"
