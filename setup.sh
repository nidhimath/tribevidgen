#!/usr/bin/env bash
# NeuroAdGen environment setup script.
# Run from the repo root: bash setup.sh
set -euo pipefail

echo "=== NeuroAdGen Setup ==="

# ---------------------------------------------------------------------------
# 0. Environment variables
# ---------------------------------------------------------------------------
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=./cache

# ---------------------------------------------------------------------------
# 1. Conda environment
# ---------------------------------------------------------------------------
echo "[1/6] Creating conda environment (python 3.11)..."
conda create -n neuroadgen python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neuroadgen

# ---------------------------------------------------------------------------
# 2. PyTorch — MPS (Metal) on macOS, CUDA on Linux
# ---------------------------------------------------------------------------
echo "[2/6] Installing PyTorch..."
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS: use MPS backend (Apple Silicon or Intel)
    pip install torch torchvision torchaudio
else
    # Linux: CUDA 12.1
    pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# ---------------------------------------------------------------------------
# 3. Pin NumPy to exact version required by TribeV2
# TribeV2 0.1.0 pins numpy==2.2.6 exactly — other versions cause conflicts.
# ---------------------------------------------------------------------------
echo "[3/6] Installing NumPy 2.2.6 (required by TribeV2)..."
pip install numpy==2.2.6

# ---------------------------------------------------------------------------
# 4. Diffusion + training libraries
# ---------------------------------------------------------------------------
echo "[4/6] Installing diffusion + training stack..."
pip install \
    diffusers==0.30.3 \
    transformers==4.44.2 \
    accelerate==0.34.2 \
    peft==0.12.0 \
    deepspeed==0.15.0 \
    tokenizers \
    sentencepiece \
    safetensors

# ---------------------------------------------------------------------------
# 5. Brain + visualisation libraries
# IMPORTANT: install tribev2 AFTER numpy<2.1 is pinned.
# ---------------------------------------------------------------------------
echo "[5/6] Installing TribeV2 + brain + visualisation libraries..."
pip install tribev2
pip install \
    nilearn \
    pyvista \
    vtk \
    matplotlib \
    seaborn

# ---------------------------------------------------------------------------
# 6. Remaining utilities
# ---------------------------------------------------------------------------
echo "[6/6] Installing remaining utilities..."
pip install \
    gradio==4.44.0 \
    wandb \
    huggingface_hub \
    moviepy \
    opencv-python \
    imageio \
    imageio-ffmpeg \
    fal-client \
    pytest \
    pyyaml \
    tqdm

# ---------------------------------------------------------------------------
# 7. Install NeuroAdGen package in editable mode
# ---------------------------------------------------------------------------
echo "Installing NeuroAdGen in editable mode..."
pip install -e .

# ---------------------------------------------------------------------------
# 8. Pre-download models (optional — comment out if storage constrained)
# ---------------------------------------------------------------------------
echo ""
echo "=== Optional: Pre-download gated models ==="
echo "Uncomment the sections below to pre-download models."
echo ""

# CogVideoX-5b (single GPU, 24GB VRAM — recommended for development)
# python -c "
# from huggingface_hub import snapshot_download
# snapshot_download('THUDM/CogVideoX-5b', local_dir='./cache/cogvideox-5b')
# "

# Wan2.1-T2V-14B (primary, requires 40+ GB VRAM)
# python -c "
# from huggingface_hub import snapshot_download
# snapshot_download('Wan-AI/Wan2.1-T2V-14B', local_dir='./cache/wan2.1-14b')
# "

# V-JEPA2 ViT-L (for differentiable reward proxy — ~1.3GB)
# python -c "
# from huggingface_hub import snapshot_download
# snapshot_download('facebook/vjepa2-vitl-fpc64-256', local_dir='./cache/vjepa2')
# "

# ---------------------------------------------------------------------------
# 9. Generate ROI masks (requires nilearn + HCP atlas data)
# ---------------------------------------------------------------------------
echo ""
echo "To generate ROI vertex masks, run:"
echo "  python scripts/generate_roi_masks.py"
echo ""
echo "=== Setup complete. Activate with: conda activate neuroadgen ==="
