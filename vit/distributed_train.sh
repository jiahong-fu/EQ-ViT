#!/bin/bash
# EQ-ViT: consume the optional port before forwarding training arguments.
set -euo pipefail

if [[ $# -lt 1 || ! $1 =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 NUM_PROC [MASTER_PORT] [training arguments...]" >&2
    exit 2
fi

NUM_PROC=$1
shift
MASTER_PORT=29500
if [[ $# -gt 0 && $1 =~ ^[0-9]+$ ]]; then
    MASTER_PORT=$1
    shift
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python -m torch.distributed.run \
    --nproc_per_node="$NUM_PROC" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/train.py" "$@"
