#!/bin/bash
#SBATCH --job-name=snm_bertsst2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --qos=acc_debug
#SBATCH --account=bsc28

# Can a pretrained BERT-base keep its GLUE accuracy with scalar-gain QK-norm attention whose
# logits are provably in [-2g, 0] with g <= 5.5 (exact comparison-free softmax in 16-bit fixed
# point)? Per task: vanilla fine-tune (also the distillation teacher), QK-norm, QK-norm + KD.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results ckpt

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iv cuda | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=${HF_CACHE_DIR:-$HOME/huggingface}

python3 -c "import torch, transformers; print('cuda_ok=', torch.cuda.is_available(), 'transformers', transformers.__version__)"
OUT=results/bert_qknorm_glue_sst2.csv
for TASK in sst2; do
  python3 bert_qknorm_glue.py --model bert-base-uncased --task $TASK --data glue/$TASK \
      --variant vanilla --epochs 3 --out $OUT --save ckpt/bert_vanilla_$TASK
  # the vanilla run saves a raw state_dict; export a HF directory for the teacher
  python3 - <<EOF
import torch
from bert_qknorm_glue import build
ck = torch.load("ckpt/bert_vanilla_$TASK/model.pt", map_location="cpu")
m = build("bert-base-uncased", "vanilla", 5.5, 2)
m.load_state_dict(ck["state_dict"])
m.save_pretrained("ckpt/bert_vanilla_${TASK}_hf")
EOF
  python3 bert_qknorm_glue.py --model bert-base-uncased --task $TASK --data glue/$TASK \
      --variant qknorm --g_max 5.5 --epochs 3 --out $OUT --save ckpt/bert_qknorm_$TASK
  python3 bert_qknorm_glue.py --model bert-base-uncased --task $TASK --data glue/$TASK \
      --variant qknorm --g_max 5.5 --epochs 3 --teacher ckpt/bert_vanilla_${TASK}_hf \
      --out $OUT --save ckpt/bert_qknorm_kd_$TASK
done
cat $OUT
echo "[job] DONE"
