#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# build_motiontrack_real_zmq.sh — build the real-robot G1 tracker (motiontrack_real_zmq) with
# TensorRT, for the Jetson NX. Output: system/build/g1_nx/motiontrack_real_zmq
#
# Usage:
#   system/cpp/scripts/build_motiontrack_real_zmq.sh [extra -D cmake args...]
#
# Env:
#   CC / CXX        compiler. Default gcc-10/g++-10 (fails if missing) — Jetson's
#                   system gcc (9) lacks C++20 <span>. On JetPack 6 use
#                   CXX=g++-11.
#   CONDA_PREFIX    if a conda env is active, it's added to CMAKE_PREFIX_PATH so
#                   CMake finds conda's matched protoc + libprotobuf (the build is
#                   tested against it, not the Jetson's system protobuf 3.6).
#   TENSORRT_ROOT   custom TensorRT install (only if not under the JetPack
#                   default /usr/lib/aarch64-linux-gnu — FindTensorRT.cmake
#                   reads it from the environment).
#   BUILD_TYPE      CMake build type (default: Release).
#   JOBS            parallel build jobs (default: 4). The Jetson NX runs out of
#                   RAM with more, so keep this small on-device.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # system/cpp
MS_DIR="$(cd "$SRC_DIR/.." && pwd)"            # system
REPO_ROOT="$(cd "$MS_DIR/.." && pwd)"          # repo root
BUILD_DIR="$MS_DIR/build/g1_nx"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-4}"

# Default gcc-10/g++-10; override via CC/CXX.
CXX="${CXX:-g++-10}"
CC="${CC:-gcc-10}"
command -v "$CXX" >/dev/null || { echo "[ERROR] $CXX not found (sudo apt-get install -y gcc-10 g++-10, or set CC/CXX)" >&2; exit 1; }
COMPILER_ARGS=(-DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX")

# CMake can't switch compilers inside an existing cache — wipe if it changed.
cache="$BUILD_DIR/CMakeCache.txt"
if [[ -f "$cache" ]]; then
    cached="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$cache")"
    want="$(command -v "$CXX" 2>/dev/null || echo "$CXX")"
    if [[ -n "$cached" && "$cached" != "$want" ]]; then
        echo "[build_motiontrack_real_zmq] compiler changed ($cached -> $want); wiping $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
fi
# Same for the generator: the build always uses Ninja, so clear a stale non-Ninja cache.
if [[ -f "$cache" ]] && ! grep -q '^CMAKE_GENERATOR:INTERNAL=Ninja$' "$cache"; then
    echo "[build_motiontrack_real_zmq] generator changed -> Ninja; wiping $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# Dependency prefix: Jetson's system protobuf (3.6) is too old for proto3
# 'optional'. An active conda env typically provides a modern protoc+libprotobuf
# (and other deps); add $CONDA_PREFIX to CMAKE_PREFIX_PATH so find_package picks
# that matched pair.
USE_CONDA=; [[ -n "${CONDA_PREFIX:-}" ]] && USE_CONDA=1
PREFIX_ARGS=()
if [[ -n "$USE_CONDA" ]]; then
    PREFIX_ARGS+=(-DCMAKE_PREFIX_PATH="$CONDA_PREFIX")
fi

GEN_ARGS=(-G Ninja)   # needs ninja-build (see system/cpp/README.md NX prerequisites)

# libstdc++ ABI bridge. conda-forge's prebuilt libs (protobuf/abseil/yaml-cpp)
# are compiled against a newer libstdc++ (GLIBCXX_3.4.29+ / GCC 11+) than the
# Jetson's system gcc-10 ships, so the FINAL LINK fails with
#   undefined reference to `...@GLIBCXX_3.4.3x'
# even though every object compiled. conda already provides that newer
# libstdc++.so.6 (it's where those symbols come from); make the gcc-10 link use
# it instead of gcc-10's own. We can't just `-L$CONDA_PREFIX/lib` because the
# driver's implicit `-lstdc++` needs an unversioned `libstdc++.so`, which the
# conda runtime package doesn't ship — so drop a symlink in a shim dir and put
# that first on the link search path. Newer libstdc++ is backward-compatible, so
# system CUDA/TRT/Unitree libs keep resolving; conda's lib is already on rpath
# for runtime (keep the env active, or set LD_LIBRARY_PATH=$CONDA_PREFIX/lib).
LINKER_ARGS=()
if [[ -n "$USE_CONDA" && -e "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    SHIM_DIR="$BUILD_DIR/_stdcxx_shim"
    mkdir -p "$SHIM_DIR"
    ln -sf "$CONDA_PREFIX/lib/libstdc++.so.6" "$SHIM_DIR/libstdc++.so"
    LINKER_ARGS+=(-DCMAKE_EXE_LINKER_FLAGS="-L$SHIM_DIR -Wl,-rpath,$CONDA_PREFIX/lib")
fi

echo "[build_motiontrack_real_zmq] src=$SRC_DIR  build=$BUILD_DIR  type=$BUILD_TYPE  jobs=$JOBS"
echo "[build_motiontrack_real_zmq] CC=$CC  CXX=$CXX"
if [[ -n "$USE_CONDA" ]]; then
    echo "[build_motiontrack_real_zmq] CMAKE_PREFIX_PATH += $CONDA_PREFIX (conda protobuf etc.)"
fi
if [[ ${#LINKER_ARGS[@]} -gt 0 ]]; then
    echo "[build_motiontrack_real_zmq] linking against conda libstdc++ ($CONDA_PREFIX/lib/libstdc++.so.6)"
fi
if [[ -n "${TENSORRT_ROOT:-}" ]]; then
    echo "[build_motiontrack_real_zmq] TENSORRT_ROOT=$TENSORRT_ROOT"
fi

cmake -S "$SRC_DIR" -B "$BUILD_DIR" "${GEN_ARGS[@]}" "${COMPILER_ARGS[@]}" "${PREFIX_ARGS[@]}" "${LINKER_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DHALO_COMM_DIR="$REPO_ROOT/system/comm" \
    -DRT_DEPLOY_WITH_UNITREE=ON \
    -DRT_DEPLOY_WITH_TRT=ON \
    "$@"

cmake --build "$BUILD_DIR" --target motiontrack_real_zmq -j "$JOBS"

echo "[build_motiontrack_real_zmq] done -> $BUILD_DIR/motiontrack_real_zmq"
echo "[build_motiontrack_real_zmq] run it with CAP_SYS_NICE + CAP_IPC_LOCK, e.g.:"
echo "    $BUILD_DIR/motiontrack_real_zmq --config tracker/deploy_cpp/configs/g1_v2.yaml --policy <model.plan> --net eth0"
