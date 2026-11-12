#!/bin/bash
#SBATCH --job-name=snm_bertg3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_bsccs
#SBATCH --account=bsc28

# Accuracy of the bound that is provably safe for the whole secure softmax at 16 bits, not only its
# numerators. Under scalar QK-norm a row sum lies in [e^-2g, N]; CrypTen's Newton reciprocal squares
# 1/sum at scale 2^32 and its 2-party truncation errs with probability ~ (1/sum)^2 * 2^-31 per
# element, so rows need sum >= ~e^-6: 2g <= 6, g <= 3 (MPC_REFRAME.md section 4g', caveat).
# Three seeds of QK-norm with g_max = 3.0; compare with g <= 5.5 (bert_qknorm_sst2_seeds.csv) and
# g <= 6.9 (bert_qknorm_sst2_gmax69.csv).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=${HF_CACHE_DIR:-$HOME/huggingface}

OUT=results/bert_qknorm_sst2_gmax30.csv
for SEED in 0 1 2; do
  python3 bert_qknorm_glue.py --model bert-base-uncased --task sst2 --data glue/sst2 \
      --variant qknorm --g_max 3.0 --g_lr 1e-3 --epochs 3 --seed $SEED --out $OUT
done
cat $OUT
echo "[job] DONE"
