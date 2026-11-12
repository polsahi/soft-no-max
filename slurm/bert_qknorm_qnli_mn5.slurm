#!/bin/bash
#SBATCH --job-name=snm_bertqnli
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# Second GLUE task for scalar QK-norm BERT (g <= 5.5): QNLI (104,743 train / 5,463 validation pairs,
# a much larger validation set than SST-2's 872). Vanilla vs QK-norm with the gains on their own
# learning rate (1e-3), seed 0. The acc_bsccs job 45892923 covers the same task but has not started.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=${HF_CACHE_DIR:-$HOME/huggingface}

OUT=results/bert_qknorm_qnli.csv
python3 bert_qknorm_glue.py --model bert-base-uncased --task qnli --data glue/qnli \
    --variant vanilla --epochs 3 --seed 0 --out $OUT
python3 bert_qknorm_glue.py --model bert-base-uncased --task qnli --data glue/qnli \
    --variant qknorm --g_max 5.5 --g_lr 1e-3 --epochs 3 --seed 0 --out $OUT
cat $OUT
echo "[job] DONE"
