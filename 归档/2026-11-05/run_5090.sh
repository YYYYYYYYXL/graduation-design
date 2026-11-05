#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$PROJECT_ROOT/run_autodl.sh" restormer --profile "$PROJECT_ROOT/options/autodl_5090.json" "$@"
