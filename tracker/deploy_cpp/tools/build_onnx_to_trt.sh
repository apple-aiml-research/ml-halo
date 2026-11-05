#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# build_onnx_to_trt.sh — build the onnx_to_trt plan builder (TensorRT) and run
# it to convert ONNX policies into serialized TensorRT .plan files.
#
# Builds from deploy_cpp/CMakeLists.txt into deploy_cpp/build/g1_nx; each .plan
# is written next to its source ONNX (or to an explicit output path).
#
# IMPORTANT: a .plan is GPU/SM + TensorRT-version specific — run this ON the
# robot's Orin (or an identical unit). A plan built elsewhere fails to
# deserialize.
#
# Usage:
#   build_onnx_to_trt.sh                              # BATCH: every *.onnx in
#                                                     #   deploy/models/g1/
#   build_onnx_to_trt.sh [model.onnx] [output.plan] [extra onnx_to_trt args...]
#   build_onnx_to_trt.sh --fp16                       # batch + passthrough args
#
#   model.onnx   omit to convert ALL *.onnx in deploy/models/g1/; or pass
#                one explicit path to convert just that model.
#   output.plan  single-model only. Default: <onnx-dir>/<onnx-basename>.plan
#                (next to the ONNX). In batch mode each plan lands next to its
#                source ONNX; an explicit output.plan is ignored.
#   extra args   passed through to onnx_to_trt, e.g. --fp16 --weakly-typed
#                --workspace <GB> --device <N> (applied to every model in batch)
#
# Env:
#   CC / CXX       compiler. Default gcc-10/g++-10 when present (Jetson's system
#                  gcc 9 lacks C++20 <span>). On JetPack 6 use CXX=g++-11. Hosts
#                  without gcc-10 fall back to CMake's default compiler.
#   CONDA_PREFIX   if a conda env is active, it's added to CMAKE_PREFIX_PATH so
#                  CMake finds the env's packages, and the link is pointed at
#                  the env's libstdc++ (see below). Set NO_CONDA_PREFIX=1 to
#                  opt out.
#   TENSORRT_ROOT  custom TensorRT install (see cmake/FindTensorRT.cmake).
#   BUILD_TYPE     CMake build type (default: Release).
#   JOBS           parallel build jobs (default: nproc).
#   DECOMPOSE_LAYERNORM  default 1 (ON): rewrite ONNX LayerNormalization into
#                  primitives before conversion, so TRT<8.6 (e.g. JetPack 5's
#                  8.5) can parse policies that use it. No-op for LayerNorm-free
#                  models, and skipped gracefully if python3/onnx is unavailable.
#                  Set DECOMPOSE_LAYERNORM=0 to disable (e.g. on TRT>=8.6 to keep
#                  the native fused LayerNorm kernel).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"           # deploy_cpp
TRACKER_DIR="$(cd "$SRC_DIR/.." && pwd)"         # tracker
DEPLOY_DIR="$TRACKER_DIR/deploy"             # models + configs
BUILD_DIR="$SRC_DIR/build/g1_nx"
DECOMPOSE_LAYERNORM="${DECOMPOSE_LAYERNORM:-1}"    # default ON (see header)
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$( (nproc 2>/dev/null) || (sysctl -n hw.ncpu 2>/dev/null) || echo 4 )}"

MODELS_DIR="$DEPLOY_DIR/models/g1"

# Collect the ONNX file(s) to convert + an optional explicit output path.
#   - explicit ONNX path as $1        -> convert just that one
#   - no positional ONNX (or $1 is a --flag) -> BATCH: every *.onnx in
#     deploy/models/g1/. Each .plan lands next to its source ONNX.
ONNX_LIST=()
PLAN_OVERRIDE=""
if [[ $# -ge 1 && "$1" != --* ]]; then
    ONNX="$1"; shift
    if [[ ! -f "$ONNX" ]]; then
        echo "[onnx_to_trt] ERROR: ONNX not found: $ONNX" >&2
        echo "             pass a path explicitly, or none to convert all of $MODELS_DIR" >&2
        exit 1
    fi
    ONNX_LIST=("$ONNX")
    # Optional explicit output path (2nd positional, only if it ends in .plan).
    if [[ $# -ge 1 && "$1" == *.plan ]]; then
        PLAN_OVERRIDE="$1"; shift
    fi
else
    shopt -s nullglob
    ONNX_LIST=()
    for f in "$MODELS_DIR"/*.onnx; do
        # Skip LayerNorm-decomposed intermediates an older version of this
        # script left next to the models.
        [[ "$f" == *.ln.onnx ]] && continue
        ONNX_LIST+=("$f")
    done
    shopt -u nullglob
    if [[ ${#ONNX_LIST[@]} -eq 0 ]]; then
        echo "[onnx_to_trt] ERROR: no .onnx files in $MODELS_DIR" >&2
        exit 1
    fi
    echo "[onnx_to_trt] batch mode: ${#ONNX_LIST[@]} model(s) in $MODELS_DIR"
fi
# Whatever remains in "$@" is passed through to onnx_to_trt (e.g. --fp16).

# Compiler: default gcc-10/g++-10 when present (Jetson's gcc 9 lacks <span>);
# override via CC/CXX (e.g. CXX=g++-11 on JetPack 6).
: "${CXX:=$(command -v g++-10 || true)}"
: "${CC:=$(command -v gcc-10  || true)}"
COMPILER_ARGS=()
if [[ -n "${CC:-}"  ]]; then COMPILER_ARGS+=(-DCMAKE_C_COMPILER="$CC"); fi
if [[ -n "${CXX:-}" ]]; then COMPILER_ARGS+=(-DCMAKE_CXX_COMPILER="$CXX"); fi

# CMake can't switch compilers inside an existing cache — wipe if it changed.
cache="$BUILD_DIR/CMakeCache.txt"
if [[ -n "${CXX:-}" && -f "$cache" ]]; then
    cached="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$cache")"
    want="$(command -v "$CXX" 2>/dev/null || echo "$CXX")"
    if [[ -n "$cached" && "$cached" != "$want" ]]; then
        echo "[onnx_to_trt] compiler changed ($cached -> $want); wiping $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
fi

# Prefer Ninja when available.
GEN_ARGS=()
command -v ninja >/dev/null 2>&1 && GEN_ARGS=(-G Ninja)

# Active conda env -> add to CMAKE_PREFIX_PATH. NO_CONDA_PREFIX=1 to opt out.
PREFIX_ARGS=()
if [[ -z "${NO_CONDA_PREFIX:-}" && -n "${CONDA_PREFIX:-}" ]]; then
    PREFIX_ARGS+=(-DCMAKE_PREFIX_PATH="$CONDA_PREFIX")
fi

# libstdc++ ABI bridge — conda's prebuilt libs (e.g. yaml-cpp) need a newer
# libstdc++ (GLIBCXX_3.4.29+) than system gcc-10 ships; point the link at
# conda's libstdc++.so.6 via a shim dir (the unversioned symlink the driver's
# implicit -lstdc++ needs). For yaml-cpp, prefer
# -DCMAKE_DISABLE_FIND_PACKAGE_yaml-cpp=ON (build it from source, ABI-matched) —
# see README.md (Build on the G1 → JetPack 5.x).
LINKER_ARGS=()
if [[ -z "${NO_CONDA_PREFIX:-}" && -n "${CONDA_PREFIX:-}" && -e "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    SHIM_DIR="$BUILD_DIR/_stdcxx_shim"
    mkdir -p "$SHIM_DIR"
    ln -sf "$CONDA_PREFIX/lib/libstdc++.so.6" "$SHIM_DIR/libstdc++.so"
    LINKER_ARGS+=(-DCMAKE_EXE_LINKER_FLAGS="-L$SHIM_DIR -Wl,-rpath,$CONDA_PREFIX/lib")
fi

echo "[onnx_to_trt] building tool (src=$SRC_DIR, build=$BUILD_DIR, type=$BUILD_TYPE)"
echo "[onnx_to_trt] CC=${CC:-<cmake default>}  CXX=${CXX:-<cmake default>}"
if [[ -z "${NO_CONDA_PREFIX:-}" && -n "${CONDA_PREFIX:-}" ]]; then
    echo "[onnx_to_trt] CMAKE_PREFIX_PATH += $CONDA_PREFIX"
fi
if [[ ${#LINKER_ARGS[@]} -gt 0 ]]; then
    echo "[onnx_to_trt] linking against conda libstdc++ ($CONDA_PREFIX/lib/libstdc++.so.6)"
fi
[[ -n "${TENSORRT_ROOT:-}" ]] && echo "[onnx_to_trt] TENSORRT_ROOT=$TENSORRT_ROOT"

# onnx_to_trt needs only RT_DEPLOY_WITH_TRT=ON (TensorRT + CUDA) — no MuJoCo /
# Unitree.
cmake -S "$SRC_DIR" -B "$BUILD_DIR" "${GEN_ARGS[@]}" "${COMPILER_ARGS[@]}" "${PREFIX_ARGS[@]}" "${LINKER_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DRT_DEPLOY_WITH_TRT=ON
cmake --build "$BUILD_DIR" --target onnx_to_trt -j "$JOBS"

# Built directly from deploy_cpp's CMake, so the tool lands at the build root
# (no tracker_build/ prefix, unlike the system/cpp add_subdirectory build).
TOOL="$BUILD_DIR/onnx_to_trt"
if [[ ! -x "$TOOL" ]]; then
    TOOL="$(find "$BUILD_DIR" -name onnx_to_trt -type f -perm -u+x 2>/dev/null | head -1)"
fi
if [[ -z "$TOOL" || ! -x "$TOOL" ]]; then
    echo "[onnx_to_trt] ERROR: built tool not found under $BUILD_DIR" >&2
    exit 1
fi

echo "[onnx_to_trt] converting ${#ONNX_LIST[@]} model(s):"
PY="${PYTHON:-python3}"
# LayerNorm-decomposed intermediates go to a private temp dir, never next to the
# models (a later batch run would otherwise convert them too).
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/onnx_to_trt.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT
LAST_PLAN=""
for ONNX in "${ONNX_LIST[@]}"; do
    SRC_ONNX="$ONNX"
    # TRT < 8.6 can't parse ONNX LayerNormalization. By default (DECOMPOSE_
    # LAYERNORM=1) rewrite it into primitives (ReduceMean/Sub/Mul/Add/Sqrt/Div)
    # via tools/decompose_layernorm.py first — a no-op for LayerNorm-free
    # models. Gracefully skipped (uses the original ONNX) if python3/onnx is
    # missing or the pass fails, so it can never regress a model that already
    # converted. Set DECOMPOSE_LAYERNORM=0 to disable (e.g. TRT>=8.6 native).
    if [[ "$DECOMPOSE_LAYERNORM" != "0" ]]; then
        if "$PY" -c "import onnx" >/dev/null 2>&1; then
            LN_ONNX="$TMP_DIR/$(basename "${ONNX%.onnx}").ln.onnx"
            echo "[onnx_to_trt] decomposing LayerNormalization: $ONNX -> $LN_ONNX (TRT<8.6 compat)"
            if "$PY" "$SCRIPT_DIR/decompose_layernorm.py" "$ONNX" "$LN_ONNX"; then
                SRC_ONNX="$LN_ONNX"
            else
                echo "[onnx_to_trt] WARN: decompose failed; using original ONNX" >&2
            fi
        else
            echo "[onnx_to_trt] WARN: python 'onnx' unavailable; skipping LayerNorm" \
                 "decompose (install: pip3 install onnx). Using original ONNX." >&2
        fi
    fi
    if [[ -n "$PLAN_OVERRIDE" ]]; then
        PLAN="$PLAN_OVERRIDE"
    else
        # .plan next to its source ONNX, same basename (original name, not .ln).
        PLAN="$(cd "$(dirname "$ONNX")" && pwd)/$(basename "${ONNX%.onnx}").plan"
    fi
    echo "    onnx -> $SRC_ONNX"
    echo "    plan -> $PLAN"
    echo "    args -> $*"
    "$TOOL" --onnx "$SRC_ONNX" --plan "$PLAN" "$@"
    LAST_PLAN="$PLAN"
done

if [[ -n "$PLAN_OVERRIDE" ]]; then
    echo "[onnx_to_trt] done. Plan written to $PLAN_OVERRIDE"
else
    echo "[onnx_to_trt] done. Plans written next to each source ONNX"
fi
echo "[onnx_to_trt] run it, e.g.:"
echo "    deploy_g1_real --config configs/g1_v2.yaml --policy $LAST_PLAN --motion <motion.csv> --net eth0"
echo "    (or the runtime-plan node motiontrack_real_zmq, built by"
echo "     system/cpp/scripts/build_motiontrack_real_zmq.sh)"
