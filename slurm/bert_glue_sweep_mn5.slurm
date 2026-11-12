#!/bin/bash
#SBATCH --job-name=snm_glue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# GLUE breadth for the bound we can claim (g <= 3.0, reciprocal-safe at 16 bits) against vanilla and
# g <= 5.5. Runs are given as SPECS="task:variant:g_max:seed ..." and spread over the node's GPUs:
# each GPU works through its share one run at a time. One CSV per run (results/glue/), merged later,
# so parallel runs never write the same file. Data: glue_data/<task> (datasets.save_to_disk).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/glue

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=${HF_CACHE_DIR:-$HOME/huggingface}
export OMP_NUM_THREADS=16

read -r -a SPEC_LIST <<< "${SPECS:?set SPECS}"
NGPU=${NGPU:-4}
EPOCHS=${EPOCHS:-3}
echo "[job] ${#SPEC_LIST[@]} runs on $NGPU GPUs"

run_one() {  # spec gpu
    local task variant gmax seed tag
    IFS=: read -r task variant gmax seed <<< "$1"
    tag=$([ "$variant" = vanilla ] && echo vanilla || echo "${variant}${gmax}")
    local out="results/glue/${task}_${tag}_s${seed}.csv"
    if [ -s "$out" ]; then echo "[skip] $1 (done)"; return 0; fi
    echo "[start] $1 on GPU $2 $(date +%T)"
    if CUDA_VISIBLE_DEVICES=$2 python3 bert_qknorm_glue.py --model bert-base-uncased --task "$task" \
        --data "glue_data/$task" --variant "$variant" --g_max "${gmax}" --g_lr 1e-3 --epochs "$EPOCHS" \
        --seed "$seed" --out "$out" > "logs/glue_${SLURM_JOB_ID}_${task}_${tag}_s${seed}.log" 2>&1; then
        echo "[done] $1 $(date +%T): $(tail -1 "$out" | cut -d, -f8)"
    else
        echo "[FAIL] $1 (see logs/glue_${SLURM_JOB_ID}_${task}_${tag}_s${seed}.log)"
        rm -f "$out"
    fi
}

for ((g = 0; g < NGPU; g++)); do
    (
        for ((i = g; i < ${#SPEC_LIST[@]}; i += NGPU)); do run_one "${SPEC_LIST[$i]}" "$g"; done
    ) &
done
wait
echo "[job] DONE"
