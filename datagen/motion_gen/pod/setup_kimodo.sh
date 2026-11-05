#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Kimodo conda env setup, plus the model weights (~16 GB) in the Hugging Face cache:
#   nvidia/Kimodo-G1-RP-v1, McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp(-supervised), meta-llama/Meta-Llama-3-8B-Instruct.
# Llama-3 is gated: accept its license on huggingface.co and export HF_TOKEN. If $HALO_S3_ROOT/kimodo-models/ holds a
# copy of the cache (models--*/ folders), it is synced from there in the background instead; the run itself is offline.
set -e

ENV_NAME="${1:-kimodo}"

source "$(dirname "${BASH_SOURCE[0]}")/../../launch/ensure_storage.sh"   # halo_s3 on PATH; aws CLI for s3:// roots

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
S3_MODELS="${HALO_S3_ROOT}/kimodo-models"
MODELS="nvidia/Kimodo-G1-RP-v1 McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp \
McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised meta-llama/Meta-Llama-3-8B-Instruct"
mkdir -p "$HF_CACHE_DIR"

SYNC_PIDS=()
if halo_s3 ls "${S3_MODELS}/" >/dev/null 2>&1; then
    echo "=== Background sync of the model cache from ${S3_MODELS} ==="
    for repo in $MODELS; do
        model="models--${repo//\//--}"
        halo_s3 sync "${S3_MODELS}/${model}/" "${HF_CACHE_DIR}/${model}/" > "/tmp/sync_${model}.log" 2>&1 &
        SYNC_PIDS+=($!)
    done
fi

source "$(dirname "${BASH_SOURCE[0]}")/../../launch/ensure_conda.sh"   # installs Miniconda if there is no conda
echo "=== Conda environment: ${ENV_NAME} ==="
conda env list | grep -q "^${ENV_NAME} " || conda create -n "${ENV_NAME}" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch (CUDA 12.8) ==="
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== Installing system deps (cmake, EGL, ffmpeg) ==="
apt-get update || true
apt-get install -y cmake build-essential libegl1 libgl1 libglx-mesa0 \
    libegl-dev libgl-dev ffmpeg --fix-missing

echo "=== Installing Kimodo from source (with MotionCorrection C++ ext for SOMA+PP) ==="
pip install "git+https://github.com/nv-tlabs/kimodo.git@58e781898b3d7e328a676a75d3e338c45dce3ad9"

echo "=== Installing additional Python deps ==="
pip install mujoco==3.14.0 trimesh scipy matplotlib usd-core==26.8

pip install --upgrade pip

if [ ${#SYNC_PIDS[@]} -eq 0 ]; then
    echo "=== Downloading the models from Hugging Face (Llama-3 needs HF_TOKEN) ==="
    python - $MODELS <<'PY'
import sys
from huggingface_hub import snapshot_download

for repo in sys.argv[1:]:
    # Llama-3: the transformers weights only (skip the original/*.pth copy)
    pat = ["*.json", "*.safetensors", "tokenizer*"] if repo.startswith("meta-llama/") else None
    print("ok", snapshot_download(repo, allow_patterns=pat), flush=True)
PY
fi

echo "=== Waiting for background HF model syncs to finish ==="
fail=0
for pid in "${SYNC_PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[ERROR] model sync pid=$pid failed"
        fail=1
    fi
done
if [ "$fail" -ne 0 ]; then
    echo "=== One or more model syncs failed; per-sync logs in /tmp/sync_*.log ==="
    for f in /tmp/sync_*.log; do
        echo "----- $f -----"; tail -30 "$f" || true
    done
    exit 1
fi
echo "All models cached in $HF_CACHE_DIR"

echo "=== Done. Activate with: conda activate ${ENV_NAME} ==="
