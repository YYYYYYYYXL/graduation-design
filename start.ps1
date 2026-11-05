$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
conda run --no-capture-output -n denoise310 python -B app.py
