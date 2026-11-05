#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASE_PYTHON="${BASE_PYTHON:-python3}"
VENV_PYTHON="${DENOISE_PYTHON:-/root/venvs/denoise/bin/python}"
"$BASE_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "请选择Python 3.10、3.11或3.12"'
if [[ ! -x "$VENV_PYTHON" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$(dirname -- "$(dirname -- "$VENV_PYTHON")")"
fi
if ! "$VENV_PYTHON" -c 'import torch; assert torch.__version__.split("+")[0] == "2.8.0" and torch.version.cuda == "12.8"'; then
    "$VENV_PYTHON" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
fi
"$VENV_PYTHON" -m pip install numpy==1.26.4 scipy==1.15.3 scikit-image==0.25.2 pillow==11.1.0 einops==0.8.1 tqdm==4.67.1
"$VENV_PYTHON" -B "$PROJECT_ROOT/train_autodl.py" restormer all --check-only
