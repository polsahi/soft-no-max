#!/bin/bash
#SBATCH --job-name=snm_mpc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_debug
#SBATCH --account=bsc28

# Secure (2-party) softmax and ViT private inference on a Xeon 8480+ node, with SHAFT's CrypTen
# fork (torch 2.0.1) plus the "shifted" softmax patch. Both parties run on this node as processes,
# one socket's cores each; rounds and bytes are exact, time is LAN-free compute.
#   1. mpc_softmax_bench.py  — per-op cost and accuracy (same as the laptop run, timed here)
#   2. mpc_vit_private.py    — end-to-end CIFAR-10 ViT, if results/ckpt/vit_cifar_*.pt exist

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/mpc

module purge
# Lmod requires mkl before python/3.11.5-gcc
module load impi intel hdf5 mkl python/3.11.5-gcc
# The module exports PYTHONHOME and a PYTHONPATH full of its own eggs; either leaks into the
# venv and breaks isolation, so drop both before creating or using it.
unset PYTHONPATH PYTHONHOME

VENV=$SLURM_SUBMIT_DIR/venv_shaft
[ -d SHAFT ] || tar xzf shaft_patched.tgz
# SHAFT (CrypTen fork) is pure Python: use it from source. An editable pip install would need the
# `wheel` package, which MN5's offline venv does not have.
export PYTHONPATH="$SLURM_SUBMIT_DIR/SHAFT"
# Test the imports, not the directory: a failed install leaves an empty venv behind.
if ! "$VENV/bin/python" -c "import torch, omegaconf, onnx, crypten" >/dev/null 2>&1; then
    rm -rf "$VENV"
    python3.11 -m venv "$VENV"
    # MN5 is offline: everything, including antlr4's runtime, must come as prebuilt wheels
    "$VENV/bin/pip" install -q --no-index --find-links wheels311 torch numpy omegaconf onnx scipy pandas pyyaml future
fi
"$VENV/bin/python" -c "import torch, crypten; print('torch', torch.__version__, 'crypten', crypten.__file__)"
grep -q '"shifted"' SHAFT/crypten/common/functions/approximations.py || { echo "shifted patch missing"; exit 1; }

export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28
cd SHAFT/examples/unit-test
"$VENV/bin/python" "$SLURM_SUBMIT_DIR/mpc_softmax_bench.py" --g 4.75 --lengths 32,64,128,256,512,1024 \
    --rows 16 --regimes cosine,peaked,flat_low --precision_bits 16,20 \
    --out "$SLURM_SUBMIT_DIR/results/mpc/mpc_softmax_mn5.csv"

# BERT-base encoder cost (random weights; cost is weight-independent): the standard 128 tokens,
# then a length sweep, since softmax cost grows with N^2 and QK-norm's with N.
"$VENV/bin/python" "$SLURM_SUBMIT_DIR/mpc_encoder_cost.py" --layers 12 --hidden 768 --heads 12 \
    --seq_lens 32,64,128,256 --out "$SLURM_SUBMIT_DIR/results/mpc/mpc_encoder_cost_bert_base.csv"

# Prefer the 100-epoch checkpoints (vanilla 81.2%, QK-norm 81.7%) over the 15-epoch ones.
CKPT="$SLURM_SUBMIT_DIR/results/ckpt100"
[ -f "$CKPT/vit_cifar_qknorm.pt" ] || CKPT="$SLURM_SUBMIT_DIR/results/ckpt"
if ls "$CKPT"/vit_cifar_*.pt >/dev/null 2>&1; then
    "$VENV/bin/python" "$SLURM_SUBMIT_DIR/mpc_vit_private.py" --ckpt_dir "$CKPT" \
        --data "$SLURM_SUBMIT_DIR/data/cifar10.npz" --n_images 200 --batch 50 --cost_batch 1 \
        --out "$SLURM_SUBMIT_DIR/results/mpc/mpc_vit_private_mn5.csv"
else
    echo "[skip] no ViT checkpoints yet"
fi
echo "[job] DONE"
