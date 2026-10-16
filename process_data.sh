#!/usr/bin/env bash
set -euo pipefail
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python -m liber0.training.check_data "$@"
