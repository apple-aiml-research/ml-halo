#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# build_motiontrack_mujoco_zmq.sh — build the MuJoCo sim tracker (motiontrack_mujoco_zmq)
# for the local dev box (macOS / Linux). Output: system/build/{mac,linux}/motiontrack_mujoco_zmq
#
# Usage:
#   system/cpp/scripts/build_motiontrack_mujoco_zmq.sh [extra -D cmake args...]
#
# Env:
#   BUILD_TYPE   CMake build type (default: Release).
#   JOBS         parallel build jobs (default: nproc / hw.ncpu).
#
# Prerequisites (macOS, via Homebrew):
#   brew install cmake ninja eigen yaml-cpp onnxruntime glfw zeromq protobuf
#   bash system/cpp/scripts/setup_dev_env.sh   # run once first
#   # MuJoCo 3.2.5 is built from source into system/build/third_party/ by this script on the first run.
#
# Prerequisites (Linux):
#   apt install cmake ninja-build g++ libeigen3-dev libyaml-cpp-dev \
#               libglfw3-dev libgl1-mesa-dev protobuf-compiler libprotobuf-dev
#   # ONNX Runtime: unpack the release tarball into system/build/third_party/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # system/cpp
MS_DIR="$(cd "$SRC_DIR/.." && pwd)"            # system
REPO_ROOT="$(cd "$MS_DIR/.." && pwd)"          # halo

# Platform-specific build dir + dependency discovery.
case "$(uname -s)" in
    Darwin)
        PLATFORM="mac"
        JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
        EIGEN_DIR="${EIGEN_DIR:-$(brew --prefix eigen 2>/dev/null)/share/eigen3/cmake}"
        YAML_CPP_DIR="${YAML_CPP_DIR:-$(brew --prefix yaml-cpp 2>/dev/null)/lib/cmake/yaml-cpp}"
        ORT_ROOT="${ORT_ROOT:-$(brew --prefix onnxruntime 2>/dev/null)}"
        GLFW_DIR="${GLFW_DIR:-$(brew --prefix glfw 2>/dev/null)/lib/cmake/glfw3}"
        ;;
    Linux)
        PLATFORM="linux"
        JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
        EIGEN_DIR="${EIGEN_DIR:-/usr/lib/x86_64-linux-gnu/cmake/eigen3}"
        YAML_CPP_DIR="${YAML_CPP_DIR:-/usr/lib/cmake/yaml-cpp}"
        ORT_ROOT="${ORT_ROOT:-$MS_DIR/build/third_party/onnxruntime-linux-x64-1.20.1}"
        GLFW_DIR="${GLFW_DIR:-/usr/lib/cmake/glfw3}"
        ;;
    *)
        echo "[ERROR] unsupported platform: $(uname -s)"
        exit 1
        ;;
esac

BUILD_DIR="$MS_DIR/build/$PLATFORM"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# MuJoCo: tracker's deploy_cpp does find_package(mujoco REQUIRED), so build the
# pinned MuJoCo (= the mujoco==3.2.5 wheel) once from source into system/build/third_party.
MUJOCO_VERSION=3.2.5
TP_DIR="$MS_DIR/build/third_party"
MUJOCO_INSTALL="$TP_DIR/mujoco-${MUJOCO_VERSION}"
MUJOCO_CMAKE_DIR="$MUJOCO_INSTALL/lib/cmake/mujoco"
if [[ ! -d "$MUJOCO_CMAKE_DIR" ]]; then
    echo "[build_motiontrack_mujoco_zmq] building MuJoCo ${MUJOCO_VERSION} from source..."
    MUJOCO_SRC="$TP_DIR/mujoco-src-${MUJOCO_VERSION}"
    rm -rf "$MUJOCO_SRC"
    git clone --depth 1 --branch "${MUJOCO_VERSION}" \
        https://github.com/google-deepmind/mujoco.git "$MUJOCO_SRC"
    cmake -S "$MUJOCO_SRC" -B "$MUJOCO_SRC/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DMUJOCO_BUILD_EXAMPLES=OFF \
        -DMUJOCO_BUILD_SIMULATE=OFF \
        -DMUJOCO_BUILD_TESTS=OFF \
        -DCMAKE_CXX_FLAGS="-Wno-deprecated-literal-operator -Wno-deprecated-declarations" \
        -DCMAKE_INSTALL_PREFIX="$MUJOCO_INSTALL"
    cmake --build "$MUJOCO_SRC/build" --target install -j "$JOBS"
fi

# Assemble CMake args.
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5      # CMake 4 rejects FetchContent deps that declare < 3.5
    -DRT_DEPLOY_WITH_MUJOCO=ON
    -DRT_DEPLOY_WITH_ORT=ON
    -DRT_DEPLOY_BUILD_TESTS=OFF
    -DRT_DEPLOY_ENABLE_WERROR=OFF
    -DHALO_COMM_DIR="$REPO_ROOT/system/comm"
    -Dmujoco_DIR="$MUJOCO_CMAKE_DIR"
    -DEigen3_DIR="$EIGEN_DIR"
    -Dyaml-cpp_DIR="$YAML_CPP_DIR"
    -DONNXRUNTIME_ROOT="$ORT_ROOT"
    -Dglfw3_DIR="$GLFW_DIR"
)

GEN_ARGS=(-G Ninja)   # ninja comes from setup_dev_env.sh (brew install ninja)
cache="$BUILD_DIR/CMakeCache.txt"
if [[ -f "$cache" ]] && ! grep -q '^CMAKE_GENERATOR:INTERNAL=Ninja$' "$cache"; then
    echo "[build_motiontrack_mujoco_zmq] generator changed -> Ninja; wiping $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

echo "[build_motiontrack_mujoco_zmq] platform=$PLATFORM  src=$SRC_DIR  build=$BUILD_DIR  type=$BUILD_TYPE  jobs=$JOBS"
echo "[build_motiontrack_mujoco_zmq] MUJOCO_CMAKE_DIR=$MUJOCO_CMAKE_DIR"
echo "[build_motiontrack_mujoco_zmq] ORT_ROOT=$ORT_ROOT"

cmake -S "$SRC_DIR" -B "$BUILD_DIR" "${GEN_ARGS[@]}" "${CMAKE_ARGS[@]}" "$@"
cmake --build "$BUILD_DIR" --target motiontrack_mujoco_zmq -j "$JOBS"

BIN="$BUILD_DIR/motiontrack_mujoco_zmq"
if [[ -x "$BIN" ]]; then
    echo "[build_motiontrack_mujoco_zmq] done -> $BIN"
else
    echo "[build_motiontrack_mujoco_zmq] ERROR: $BIN not found after build"
    exit 1
fi
