#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Scene-gen setup (STAGE_SETUP of configs/pipeline/scene_gen.env). Installs, once, into fixed prefixes:
#   Blender 5.2.0 LTS (the paper scenes; bake, GT frames, USD export) -> $BLENDER_DIR, from download.blender.org
#   Infinigen 892947be + 2 submodules              -> $INFINIGEN_ROOT, from github.com/princeton-vl/infinigen
#   Python 3.11 env with Infinigen (bpy==4.2.0)    -> $INFINIGEN_ENV (conda; Miniconda is installed if missing)
# Each part is skipped when it is already there, so rerunning is cheap. Needs an NVIDIA driver, git, curl;
# apt packages are installed when apt-get is available (Debian/Ubuntu, run as root).
set -eo pipefail
BLENDER_DIR="${BLENDER_DIR:-/opt/blender52}"
BLENDER_TAR=Blender5.2/blender-5.2.0-linux-x64.tar.xz
# download.blender.org can answer cloud IPs with a Cloudflare challenge (403); then try official mirrors
BLENDER_URLS="${BLENDER_URL:-https://download.blender.org/release/$BLENDER_TAR https://mirrors.ocf.berkeley.edu/blender/release/$BLENDER_TAR https://ftp.nluug.nl/pub/graphics/blender/release/$BLENDER_TAR https://mirror.clarkson.edu/blender/release/$BLENDER_TAR}"
INFINIGEN_ROOT="${INFINIGEN_ROOT:-/opt/infinigen_src}"
INFINIGEN_REF="${INFINIGEN_REF:-892947be71c4ff102c6a558d088ffe149561c11f}"
INFINIGEN_ENV="${INFINIGEN_ENV:-/opt/infinigen_env}"

if command -v apt-get >/dev/null; then
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libgl1 libglu1-mesa libxext6 \
        libegl1 libgles2 libglib2.0-0 xz-utils git curl ca-certificates unzip >/dev/null
fi

# Blender, plus the packages the bake / USD export / scene analysis import from its bundled python
if [ ! -x "$BLENDER_DIR/blender" ]; then
    mkdir -p "$BLENDER_DIR"
    for url in $BLENDER_URLS; do
        echo "=== Blender: $url -> $BLENDER_DIR ==="
        # a stalled mirror (under 1 MB/s for 60 s) falls through to the next
        curl -fsSL --connect-timeout 20 --speed-limit 1000000 --speed-time 60 --retry 2 \
            "$url" -o /tmp/blender.tar.xz && break
    done
    tar xJf /tmp/blender.tar.xz -C "$BLENDER_DIR" --strip-components=1 && rm -f /tmp/blender.tar.xz
fi
BLPY=$(ls "$BLENDER_DIR"/*/python/bin/python3* | head -1)
"$BLPY" -m ensurepip >/dev/null 2>&1 || true
"$BLPY" -m pip install -q numpy opencv-python-headless usd-core scipy matplotlib imageio

# Infinigen source at the pinned commit; only 2 of its submodules are needed to import it
if [ ! -f "$INFINIGEN_ROOT/infinigen_examples/configs_indoor/singleroom.gin" ]; then
    echo "=== Infinigen $INFINIGEN_REF -> $INFINIGEN_ROOT ==="
    git clone -q https://github.com/princeton-vl/infinigen.git "$INFINIGEN_ROOT"
    git -C "$INFINIGEN_ROOT" checkout -q "$INFINIGEN_REF"
    git -C "$INFINIGEN_ROOT" submodule update -q --init --depth 1 infinigen/infinigen_gpl infinigen/OcMesher
fi

# Python 3.11 env for Infinigen. Infinigen's own requirements.txt is not installable as is; this is the
# import closure of infinigen_examples.generate_indoors. scikit-image / scikit-learn sit above Infinigen's caps
# (the capped versions have no python 3.11 wheels), hence --no-deps for Infinigen itself.
if ! "$INFINIGEN_ENV/bin/python" -c "import bpy, infinigen" >/dev/null 2>&1; then
    echo "=== Infinigen env -> $INFINIGEN_ENV ==="
    source "$(dirname "${BASH_SOURCE[0]}")/../../launch/ensure_conda.sh"
    conda create -y -q -p "$INFINIGEN_ENV" python=3.11
    IPY="$INFINIGEN_ENV/bin/python"
    "$IPY" -m pip install -q --upgrade pip
    "$IPY" -m pip install -q "bpy==4.2.0" "numpy==1.26.4" "shapely==2.0.5" "gin-config==0.5.0"
    "$IPY" -m pip install -q "imageio==2.31.6" matplotlib networkx "opencv-python==4.8.1.78" pandas psutil \
        "scikit-image==0.22.0" "scikit-learn==1.3.2" scipy tqdm "trimesh==3.22.5" \
        OpenEXR submitit geomdl python-fcl rtree vnoise frozendict PyYAML Pillow coacd
    INFINIGEN_MINIMAL_INSTALL=True INFINIGEN_INSTALL_TERRAIN=False INFINIGEN_INSTALL_CUSTOMGT=False \
        "$IPY" -m pip install -q --no-build-isolation --no-deps -e "$INFINIGEN_ROOT"
fi
(cd "$INFINIGEN_ROOT" && "$INFINIGEN_ENV/bin/python" -c \
    "import bpy, infinigen_examples.generate_indoors; print('infinigen OK, bpy', bpy.app.version_string)")
"$BLENDER_DIR/blender" -b --version | head -1

source "$(dirname "${BASH_SOURCE[0]}")/../../launch/ensure_storage.sh"   # halo_s3 on PATH; aws CLI for s3:// roots
