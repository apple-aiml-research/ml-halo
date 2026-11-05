#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Isaac render setup (configs/pipeline/render.env; Isaac Lab 2.3, isaaclab.sh at $ISAACLAB_SH, default: the
# official container path /workspace/isaaclab/isaaclab.sh). Installs the X / GL libs, then the Vulkan userspace-lib patch
# (launch/vkfix.sh: why and how), a no-op on hosts where vulkaninfo already works.

set -e
export PYTHONUNBUFFERED=1
# X / GL runtime libs Kit loads (libneuray needs libGL.so.1); the Isaac Lab container has them, a plain CUDA image not
apt-get update && apt-get install -y --no-install-recommends unzip ffmpeg \
    libgl1 libglu1-mesa libegl1 libxrender1 libxi6 libxext6 libsm6 libxt6 libxrandr2 libxinerama1 libxcursor1 \
    libxkbcommon0 libglib2.0-0
# after the GL libs: vkfix checks vulkaninfo, which needs libGL
bash "$(dirname "${BASH_SOURCE[0]}")/../../launch/vkfix.sh"

echo "=== Render setup ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
"${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}" -p -m pip install smplx==0.1.28 trimesh mujoco==3.14.0 yourdfpy==0.0.60 tyro==1.0.16 tqdm matplotlib usd-core==26.8 pyarrow==25.0.1 pandas
source "$(dirname "${BASH_SOURCE[0]}")/../../launch/ensure_storage.sh"   # halo_s3 on PATH; aws CLI for s3:// roots

# THOR robot tree with absolute mesh paths: Isaac's URDF importer skips relative "meshes/..." paths.
THOR_DST="${RENDER_DATA_DIR:-${OUTPUT_DIR:-$PWD/outputs/render}/inputs}/g1_thor"
mkdir -p "$THOR_DST"
cp -r "$(dirname "${BASH_SOURCE[0]}")/../../../assets/robots/g1/." "$THOR_DST/"
URDF="$THOR_DST/urdf/g1_29dof_v2_thor.urdf"
sed -i.bak "s|filename=\"meshes/|filename=\"${THOR_DST}/meshes/|g" "$URDF"
echo "[setup] rewrote $(grep -c "filename=\"${THOR_DST}/meshes/" "$URDF") mesh paths to absolute in $URDF"
echo "=== Render setup complete ==="
