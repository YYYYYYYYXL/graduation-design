#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
[[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]] || OMP_NUM_THREADS=4
[[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]] || MKL_NUM_THREADS=4
export OMP_NUM_THREADS MKL_NUM_THREADS
exec "${DENOISE_PYTHON:-/root/venvs/denoise/bin/python}" -B "$PROJECT_ROOT/train_autodl.py" "$@"
