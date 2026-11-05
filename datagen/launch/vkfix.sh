#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# vkfix.sh -- make Vulkan work for Isaac RTX on GPU hosts whose container mounts the NVIDIA ICD json but
# not the userspace GL/Vulkan libraries (vulkaninfo: ERROR_INCOMPATIBLE_DRIVER). Installs the libraries
# of the host driver version from NVIDIA's .run (else apt libnvidia-gl-<branch>). A no-op where
# vulkaninfo already works. Never fails the caller.
set -uo pipefail
vk_ok() { vulkaninfo --summary >/dev/null 2>&1; }

echo "=== vkfix probe (Isaac RTX Vulkan workaround) ==="
if ! command -v vulkaninfo >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y --no-install-recommends vulkan-tools >/dev/null 2>&1 || true
fi
vk_ok && { echo "[vkfix] vulkaninfo already works, skipping"; exit 0; }
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
[ -n "$DRIVER" ] || { echo "[vkfix] nvidia-smi gave no driver version; bailing (Isaac RTX render will likely fail)"; exit 0; }

echo "[vkfix] vulkaninfo broken, patching for host driver $DRIVER"
RUN_FILE="/tmp/nvidia-${DRIVER}.run"
EXTRACT_DIR="/tmp/nvidia-${DRIVER}"
LIB_DST=/usr/lib/x86_64-linux-gnu
DOWNLOADED=false
for URL in "https://us.download.nvidia.com/tesla/${DRIVER}/NVIDIA-Linux-x86_64-${DRIVER}.run" \
           "https://us.download.nvidia.com/XFree86/Linux-x86_64/${DRIVER}/NVIDIA-Linux-x86_64-${DRIVER}.run"; do
    echo "[vkfix] trying $URL"
    curl -fSL --retry 2 --connect-timeout 20 -o "$RUN_FILE" "$URL" && { DOWNLOADED=true; break; }
done

if $DOWNLOADED; then
    mkdir -p "$EXTRACT_DIR"
    ( cd "$EXTRACT_DIR" && sh "$RUN_FILE" --extract-only --target "$EXTRACT_DIR" >/dev/null 2>&1 ) || \
        ( cd /tmp && sh "$RUN_FILE" --extract-only >/dev/null 2>&1 && \
          mv "NVIDIA-Linux-x86_64-${DRIVER}"/* "$EXTRACT_DIR"/ )
    find "$EXTRACT_DIR" -maxdepth 1 -name "*.so.${DRIVER}" -exec cp -n {} "$LIB_DST/" \;
    (cd "$LIB_DST" && for link in libGLX_nvidia.so.0 libEGL_nvidia.so.0 libGLESv1_CM_nvidia.so.1 \
            libGLESv2_nvidia.so.2 libnvidia-opticalflow.so.1 libnvidia-fbc.so.1 libnvidia-allocator.so.1 \
            libnvidia-api.so.1 libnvidia-ptxjitcompiler.so.1 libnvoptix.so.1 libnvidia-encode.so.1 \
            libnvidia-nvvm.so.4 libnvidia-cfg.so.1 libnvidia-opencl.so.1; do
        target="${link%.so.*}.so.${DRIVER}"
        [ -f "$target" ] && ln -sf "$target" "$link"
    done)
    ldconfig
    rm -rf "$EXTRACT_DIR" "$RUN_FILE"
    if vk_ok; then echo "[vkfix] OK — vulkaninfo works post-patch"; else
        echo "[vkfix] STILL BROKEN post-patch; Isaac RTX render will probably fail"
        vulkaninfo --summary 2>&1 | head -20 || true
    fi
else
    echo "[vkfix] no NVIDIA .run found for $DRIVER; falling back to apt libnvidia-gl-<branch>"
    BRANCH="${DRIVER%%.*}"
    apt-get update -qq
    apt-get install -y --no-install-recommends "libnvidia-gl-${BRANCH}-server" >/dev/null 2>&1 || \
        apt-get install -y --no-install-recommends "libnvidia-gl-${BRANCH}" >/dev/null 2>&1 || \
        echo "[vkfix] apt fallback also failed"
    ldconfig
    if vk_ok; then echo "[vkfix] OK — vulkaninfo works post-apt-fallback"; else
        echo "[vkfix] STILL BROKEN post-apt-fallback; Isaac RTX render will probably fail"
    fi
fi
echo "=== vkfix done ==="
