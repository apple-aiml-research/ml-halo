#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Source me: puts `conda` on PATH, installing Miniconda into $CONDA_DIR (default /opt/miniconda) if there is none.
if ! command -v conda >/dev/null; then
    CONDA_DIR="${CONDA_DIR:-/opt/miniconda}"
    if [ ! -x "$CONDA_DIR/bin/conda" ]; then
        echo "=== Miniconda -> $CONDA_DIR ==="
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "$CONDA_DIR" && rm -f /tmp/miniconda.sh
    fi
    export PATH="$CONDA_DIR/bin:$PATH"
fi
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
