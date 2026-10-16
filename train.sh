#!/usr/bin/env bash
set -euo pipefail
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
: "${NPROC_PER_NODE:?Set the number of GPUs per node}"
NNODES=${NNODES:-1}
if [[ "$NNODES" == 1 ]]; then
    python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
        -m liber0.training.train "$@"
else
    : "${NODE_RANK:?Set NODE_RANK}"
    : "${MASTER_ADDR:?Set MASTER_ADDR}"
    : "${MASTER_PORT:?Set MASTER_PORT}"
    python -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        -m liber0.training.train "$@"
fi
