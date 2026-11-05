#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# MuJoCo CPU render setup: OSMesa/EGL + mujoco + python deps (Debian/Ubuntu, as root)
set -e
export PYTHONUNBUFFERED=1

echo "=== MuJoCo CPU setup ==="
echo "Task ID: ${TASK_ID:-local}"
nproc || true

echo "Installing system deps..."
apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 libosmesa6-dev libgl1 libegl1 \
    libglfw3 ffmpeg unzip curl ca-certificates python3-pip

echo "Installing python deps..."
pip3 install --no-cache-dir --break-system-packages \
    mujoco numpy scipy imageio imageio-ffmpeg pyarrow pandas opencv-python-headless usd-core

source "$(dirname "${BASH_SOURCE[0]}")/../../launch/ensure_storage.sh"   # halo_s3 on PATH; aws CLI for s3:// roots

echo "=== setup complete ==="
