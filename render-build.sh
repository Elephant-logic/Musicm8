#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip setuptools wheel
# Render web services are CPU-based. Use PyTorch's CPU wheel index to avoid pulling CUDA runtime packages.
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
python -m pip install -r requirements-render.txt
python -m compileall -q .
