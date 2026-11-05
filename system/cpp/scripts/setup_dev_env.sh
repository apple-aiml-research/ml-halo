#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# setup_dev_env.sh — one-command setup for the halo dev box.
#
# Installs everything needed to build and run the sim:
#   1. Homebrew packages (cmake, ninja, eigen, yaml-cpp, onnxruntime, glfw, zmq, protobuf)
#   2. Python packages (halo_comm, halo_serve [+ mlx on Apple Silicon], halo_orchestrator, generator,
#      halo_system, torch, mujoco, onnxruntime, etc.)
#   3. The Hugging Face models the sim loads offline (skip with SKIP_HF_MODELS=1)
# The MuJoCo CMake package for the C++ tracker is built by build_motiontrack_mujoco_zmq.sh.
#
# Idempotent — safe to re-run. Stops on the first failed step.
# Run from the repo root:
#
#   bash system/cpp/scripts/setup_dev_env.sh
#
# Env overrides:
#   PYTHON=python      Python interpreter to use (default: auto-detect)
#   CONDA_ENV=name     Activate this conda env before pip installs (default: none)
#   SKIP_BREW=1        Skip step 1 (e.g. an offline box that already has the C++ deps)
#   SKIP_HF_MODELS=1   Skip step 3 (copy ~/.cache/huggingface/hub/models--* from another box instead)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # system/cpp
MS_DIR="$(cd "$SRC_DIR/.." && pwd)"            # system
REPO_ROOT="$(cd "$MS_DIR/.." && pwd)"          # halo

PLATFORM="$(uname -s)"

echo "========================================================"
echo " halo dev environment setup"
echo " platform: $PLATFORM"
echo " repo:     $REPO_ROOT"
echo "========================================================"

# ── 1. Homebrew (macOS) / apt (Linux) ────────────────────────────────────────
if [[ -n "${SKIP_BREW:-}" ]]; then
    echo "[setup] SKIP_BREW set: not installing system packages"
elif [[ "$PLATFORM" == "Darwin" ]]; then
    if ! xcode-select -p >/dev/null 2>&1; then
        echo "[ERROR] Xcode Command Line Tools are missing: run 'xcode-select --install', then re-run." >&2
        exit 1
    fi
    if ! command -v brew >/dev/null 2>&1; then
        echo "[setup] Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    BREW_PKGS=(cmake ninja eigen yaml-cpp onnxruntime glfw zeromq protobuf)
    echo "[setup] Checking Homebrew packages: ${BREW_PKGS[*]}"
    for pkg in "${BREW_PKGS[@]}"; do
        if ! brew list "$pkg" >/dev/null 2>&1; then
            echo "[setup]   brew install $pkg"
            brew install "$pkg"
        else
            echo "[setup]   $pkg already installed"
        fi
    done
elif [[ "$PLATFORM" == "Linux" ]]; then
    APT_PKGS=(cmake ninja-build g++ pkg-config libeigen3-dev libyaml-cpp-dev \
                  libzmq3-dev libglfw3-dev libgl1-mesa-dev libegl-dev protobuf-compiler libprotobuf-dev curl)
    echo "[setup] apt install: ${APT_PKGS[*]}"
    SUDO=""
    [[ "$(id -u)" -ne 0 ]] && SUDO=sudo   # root containers have no sudo
    $SUDO apt-get update -y
    $SUDO apt-get install -y "${APT_PKGS[@]}"
    # ONNX Runtime has no apt package: fetch the release build_motiontrack_mujoco_zmq.sh expects
    ORT_DIR="$MS_DIR/build/third_party/onnxruntime-linux-x64-1.20.1"
    if [[ ! -d "$ORT_DIR/lib" ]]; then
        echo "[setup] ONNX Runtime 1.20.1 -> $ORT_DIR"
        mkdir -p "$(dirname "$ORT_DIR")"
        curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz \
            | tar xz -C "$(dirname "$ORT_DIR")"
    fi
else
    echo "[ERROR] unsupported platform: $PLATFORM" >&2; exit 1
fi

# ── 2. Python packages ──────────────────────────────────────────────────────
# Pick the Python interpreter.
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -n "${CONDA_ENV:-}" ]]; then
        set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; set -u
    fi
    PYTHON="python"
fi
echo "[setup] Python: $($PYTHON --version 2>&1)"

# Guard the interpreter version. The pinned stack (notably mujoco==3.2.5)
# ships wheels only for CPython 3.10–3.12. On 3.13+ (e.g. miniconda's base
# env) pip finds no wheel and falls back to building mujoco from C source,
# which dies with "MUJOCO_PATH environment variable is not set" — a cryptic
# 160-line traceback. Fail fast here with the actual fix instead.
PY_MINOR="$($PYTHON -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo 0)"
PY_MAJOR="$($PYTHON -c 'import sys; print(sys.version_info[0])' 2>/dev/null || echo 0)"
if [[ "$PY_MAJOR" -ne 3 || "$PY_MINOR" -lt 10 || "$PY_MINOR" -gt 12 ]]; then
    echo "[ERROR] Python $PY_MAJOR.$PY_MINOR is unsupported — the pinned deps" >&2
    echo "        (mujoco==3.2.5, etc.) only have wheels for CPython 3.10–3.12." >&2
    echo "        You are likely on miniconda's base env. Use a matching env:" >&2
    echo "          conda activate halo           # or any 3.10–3.12 env" >&2
    echo "          # or create one:" >&2
    echo "          conda create -n motion python=3.11 -y && conda activate motion" >&2
    echo "        then re-run this setup (or pass CONDA_ENV=<env> / PYTHON=<path>)." >&2
    exit 1
fi

# Single OpenMP runtime. torch (pip) bundles its own libomp; if the conda
# env also has intel-openmp/libiomp5, importing torch aborts with
# "OMP: Error #15 ... libomp already initialized" (Abort trap: 6). Pin the
# conda llvm-openmp so there is one canonical runtime, avoiding the unsafe
# KMP_DUPLICATE_LIB_OK=TRUE workaround.
if command -v conda >/dev/null 2>&1 && ! conda list -f llvm-openmp | grep -q llvm-openmp; then
    echo "[setup]   installing llvm-openmp (conda)..."
    conda install -y llvm-openmp
fi

# Core deps needed by the generator (torch, mujoco, onnxruntime, etc.)
# Install torch first so pip doesn't pull a CPU-only wheel.
if ! $PYTHON -c "import torch, torchvision" 2>/dev/null; then
    echo "[setup]   installing torch..."
    if [[ "$PLATFORM" == "Darwin" ]]; then
        $PYTHON -m pip install 'torch>=2.1' torchvision
    else
        $PYTHON -m pip install 'torch>=2.1' torchvision \
            --index-url https://download.pytorch.org/whl/cu128
    fi
else
    echo "[setup]   torch already installed"
fi

# mujoco (Python package; used by the sim launcher and the viewer)
if ! $PYTHON -c "import mujoco" 2>/dev/null; then
    echo "[setup]   installing mujoco==3.2.5..."
    $PYTHON -m pip install mujoco==3.2.5
else
    echo "[setup]   mujoco already installed"
fi

# onnxruntime (policy inference)
if ! $PYTHON -c "import onnxruntime" 2>/dev/null; then
    echo "[setup]   installing onnxruntime..."
    $PYTHON -m pip install onnxruntime
else
    echo "[setup]   onnxruntime already installed"
fi

# opencv-python (for the generator's camera/image handling)
if ! $PYTHON -c "import cv2" 2>/dev/null; then
    echo "[setup]   installing opencv-python..."
    $PYTHON -m pip install opencv-python
else
    echo "[setup]   opencv already installed"
fi

# The repo packages (editable install). On Apple Silicon halo_serve also gets MLX: the
# orchestrator's local model and the generator's fast path run on it.
APPLE_SILICON=""
[[ "$PLATFORM" == "Darwin" && "$(uname -m)" == "arm64" ]] && APPLE_SILICON=1
echo "[setup]   installing halo_comm, halo_serve, halo_orchestrator, tracker, generator, halo_system (editable)..."
$PYTHON -m pip install -e "$REPO_ROOT/system/comm/python"
if [[ -n "$APPLE_SILICON" ]]; then
    $PYTHON -m pip install -e "$REPO_ROOT/system/serve/python[mlx]"
else
    echo "[setup]   not Apple Silicon: halo_serve without MLX (no local orchestrator model)"
    $PYTHON -m pip install -e "$REPO_ROOT/system/serve/python"
fi
$PYTHON -m pip install -e "$REPO_ROOT/orchestrator"
$PYTHON -m pip install -e "$REPO_ROOT/tracker"
$PYTHON -m pip install -e "$REPO_ROOT/generator"
$PYTHON -m pip install -e "$REPO_ROOT/system"
# The committed *_pb2.py need protobuf 5.29.x; re-assert it in case another package moved it.
$PYTHON -m pip install 'protobuf>=5.29,<6'

# Verify imports
echo "[setup]   verifying imports..."
$PYTHON -c "import halo_comm, halo_serve, halo_orchestrator, halo_system, generator.models.motion_generator, transformers, pyarrow, torchvision, trimesh, PIL, yaml, zmq, mujoco, torch, cv2, onnxruntime; print('  imports OK (mujoco', mujoco.__version__, ', transformers', transformers.__version__, ')')"
if [[ -n "$APPLE_SILICON" ]]; then
    $PYTHON -c "import mlx.core as mx, mlx_vlm, mlx_vlm.structured, llguidance; assert mx.metal.is_available(), 'Metal not available'; print('  MLX OK (mlx', mx.__version__, ')')"
    MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    if (( MEM_GB < 32 )); then
        echo "[WARN] ${MEM_GB} GB RAM: the local orchestrator model (Qwen3.6-35B-A3B-4bit, ~19 GB) needs >= 32 GB."
    fi
fi

# ── 3. Hugging Face models (the sim runs with HF_HUB_OFFLINE=1) ──────────────
#   Qwen/Qwen3.5-2B                     generator vision-language encoder (~4.3 GB)
#   mlx-community/Qwen3.6-35B-A3B-4bit  local orchestrator model (~19 GB, Apple Silicon only)
HF_MODELS=(Qwen/Qwen3.5-2B)
[[ -n "$APPLE_SILICON" ]] && HF_MODELS+=(mlx-community/Qwen3.6-35B-A3B-4bit)
if [[ -n "${SKIP_HF_MODELS:-}" ]]; then
    echo "[setup] SKIP_HF_MODELS set: not downloading ${HF_MODELS[*]}"
else
    for m in "${HF_MODELS[@]}"; do
        echo "[setup]   hf download $m"
        if ! HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 $PYTHON -c \
                "import sys, huggingface_hub as h; h.snapshot_download(sys.argv[1])" "$m" >/dev/null; then
            echo "[WARN] could not download $m. Copy ~/.cache/huggingface/hub/models--${m//\//--} from a box that has it."
        fi
    done
fi

echo "========================================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Build the C++ tracker:"
echo "      bash system/cpp/scripts/build_motiontrack_mujoco_zmq.sh"
echo "   2. Copy a task from system/halo_system/scripts/configs/ and fill in its <...> paths"
echo "      (generator checkpoint + gen_config, tracker policy_path)."
echo "   3. Run it in the sim:"
echo "      python system/halo_system/scripts/run_sim.py --task_config <task>.yaml --web"
echo ""
echo "========================================================"
