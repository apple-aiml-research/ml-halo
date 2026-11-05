#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Create the conda env for training and open-loop eval.
#
#   bash setup.sh [env_name]      # default env name: halo_generator
#   conda activate halo_generator
#
# Linux (NVIDIA): CUDA 12.8 PyTorch, flash-linear-attention + causal-conv1d for
# fast Qwen3.5 linear attention, and EGL for headless MuJoCo rendering
# (MUJOCO_GL=egl is stored in the env, so activating it sets the variable).
# apt-get needs root or sudo. macOS: the default PyTorch build, without the
# CUDA-only extras.
set -eo pipefail

ENV_NAME="${1:-halo_generator}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMERS_COMMIT=c14b7b9d5523d1eeda1daeef68085d8114aa56f4
# flash-linear-attention>=0.6 requires transformers>=5.6.2 and would replace the
# pinned build above.
FLA_VERSION=0.5.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

if [ "$(uname)" = Linux ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    pip install torch torchvision
fi
pip install "git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_COMMIT}"
pip install -r "$HERE/requirements.txt"

if [ "$(uname)" = Linux ]; then
    pip install "flash-linear-attention==${FLA_VERSION}"
    # Compiles against the local CUDA toolkit. Optional: without it Qwen3.5 uses
    # a torch fallback for the convolution, so a failed build only warns.
    pip install --no-build-isolation causal-conv1d \
        || echo "WARNING: causal-conv1d failed to build; continuing without it."

    SUDO=""
    [ "$(id -u)" -ne 0 ] && SUDO=sudo
    $SUDO apt-get update && $SUDO apt-get install -y libegl1 libgl1 ffmpeg \
        || echo "WARNING: apt-get failed; install EGL (libegl1) for headless rendering."
    conda env config vars set -n "$ENV_NAME" MUJOCO_GL=egl
fi

echo "Done. Activate the env with: conda activate $ENV_NAME"
