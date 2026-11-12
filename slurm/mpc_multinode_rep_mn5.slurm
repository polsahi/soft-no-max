#!/bin/bash
#SBATCH --job-name=snm_mpc2rep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --qos=gp_debug
#SBATCH --account=bsc28

# Repeats 2 and 3 of the two-machine run (variance for the wall-time column). Original: mpc_multinode_mpc_multinode_bert_base.csv.
# Point 1: a real two-machine run of the BERT-base secure comparison. One party per MN5 GPP node
# (Xeon 8480+), talking over the cluster network with gloo, instead of two processes on one node.
# The harness first measures the link (bandwidth, round trip), so the single-node cost model can be
# checked against measured wall time on a real network. A WAN setting cannot be emulated without
# root (no tc/netem), so WAN stays modelled, now anchored by this measured LAN point.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/mpc

module purge
module load impi intel hdf5 mkl python/3.11.5-gcc
unset PYTHONPATH PYTHONHOME
VENV=$SLURM_SUBMIT_DIR/venv_shaft
export PYTHONPATH="$SLURM_SUBMIT_DIR/SHAFT"
"$VENV/bin/python" -c "import torch, crypten; print('torch', torch.__version__)"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "[nodes] ${NODES[*]}"
export MASTER="${NODES[0]}"
export RENDEZVOUS="tcp://${MASTER}:29517" WORLD_SIZE=2 DISTRIBUTED_BACKEND=gloo SNM_MULTINODE=1
export OMP_NUM_THREADS=56 MKL_NUM_THREADS=56

run() {  # tag, extra harness args...
    local tag=$1; shift
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=112 --kill-on-bad-exit=1 \
        bash -c "export RANK=\$SLURM_PROCID; cd $SLURM_SUBMIT_DIR/SHAFT/examples/unit-test && \
                 $VENV/bin/python $SLURM_SUBMIT_DIR/mpc_encoder_cost.py $* \
                 --out $SLURM_SUBMIT_DIR/results/mpc/mpc_multinode_${tag}.csv"
}

for rep in 2 3; do
run bert_base_rep$rep --arch bert --layers 12 --hidden 768 --heads 12 --seq_lens 128,256 --net_probe
done

echo "[job] DONE"
