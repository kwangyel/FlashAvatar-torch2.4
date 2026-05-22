#!/usr/bin/env bash
# RunPod setup for FlashAvatar on Python 3.11 + PyTorch 2.4 (CUDA 12.4).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python -m pip install -U pip setuptools wheel

# Install torch 2.4 CUDA 12.4 build.
# python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# User-requested PyTorch3D installation path.
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Project Python dependencies.
python -m pip install --no-build-isolation -r requirements-runpod.txt

# Build CUDA extensions in the active torch environment.
python -m pip install --no-build-isolation -e submodules/simple-knn
python -m pip install --no-build-isolation -e submodules/diff-gaussian-rasterization

python scripts/verify_cuda_extensions.py
