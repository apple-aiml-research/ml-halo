#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# setup_real_deployment.sh — one-time setup for real-robot deployment.
#
# Run this ON the Mac mini. It:
#   1. Installs Python packages locally (setup_dev_env.sh + depthai for the OAK camera) and
#      sets HALO_COMM_REGISTRY=<robot>:5560 in ~/.zshrc
#   2. Copies this repo to the robot (rsync), installs the NX build packages (apt, needs
#      passwordless sudo or does nothing), and builds the C++ real tracker (motiontrack_real_zmq)
#      with TensorRT for aarch64, then builds the TensorRT .plan from the ONNX
#      policy — the .plan is GPU/SM + TRT-version specific, so it MUST be built
#      on the NX. It lands next to the ONNX at
#      tracker/deploy/models/g1/<name>.plan.
#      Skipped if the .plan already exists (delete it to rebuild).
#
# After this, start the tracker on the NX and run run_real.py (see system/README.md#run-real).
#
# Usage:
#   bash system/cpp/scripts/setup_real_deployment.sh \
#       --robot_ssh unitree@192.168.123.164 \
#       --robot_repo /home/unitree/halo
#
# Env overrides:
#   POLICY_ONNX=path ONNX policy to convert on the robot, relative to --robot_repo
#                    (default: tracker/deploy/models/g1/motiontrack_all_data.onnx)
#   TRT_ARGS="..."   Extra onnx_to_trt args, e.g. TRT_ARGS="--fp16 --workspace 4"
#   TRACKER_CMAKE_ARGS="..."  Extra CMake args for motiontrack_real_zmq, e.g.
#                    "-DRT_DEPLOY_WITH_REALSENSE=ON" for the depth student (D435i). With it, the
#                    robot gets libnvinfer-plugin-dev and librealsense2 (apt, else built from source)
#   INSTALL_REALSENSE=0  With the depth build, do not build librealsense from source when apt has no
#                    package (the stock Jetson apt has no arm64 one). Default: build it (once, ~20-40 min,
#                    needs GitHub access and passwordless sudo on the robot).
#   REALSENSE_VERSION=tag  librealsense tag for that source build (default: v2.55.1)
#   PYTHON=python    Python interpreter (default: auto-detect)
#   CONDA_ENV=name   Activate this conda env before pip installs (default: none)
#   SKIP_LOCAL=1     Skip step 1 (Mac mini already set up)
#   SKIP_BREW=1 / SKIP_HF_MODELS=1   Passed to setup_dev_env.sh (the mini is often offline:
#                    copy ~/.cache/huggingface/hub/models--* from the laptop instead)
#   SKIP_SYNC=1      Do not rsync the repo to the robot (it is already at --robot_repo)
#   ROBOT_CONDA_ENV=name  Activate this conda env ON THE ROBOT before the C++
#                    build (default: base). The build uses that env's protoc +
#                    libprotobuf (the Jetson's system protobuf 3.6 is not the
#                    tested path). The robot's base env already has one, so the
#                    default works.
set -euo pipefail

# ── helpers ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

err()    { echo -e "${RED}[ERROR]${NC} $*" >&2; }
ok()     { echo -e "${GREEN}[OK]${NC} $*"; }
section(){ echo ""; echo -e "${GREEN}=== $* ===${NC}"; }

# ── parse args ─────────────────────────────────────────────────────────────────
ROBOT_SSH=""
ROBOT_REPO=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot_ssh)  ROBOT_SSH="$2"; shift 2 ;;
        --robot_repo) ROBOT_REPO="$2"; shift 2 ;;
        *)            err "unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # system/cpp
MS_DIR="$(cd "$SRC_DIR/.." && pwd)"            # system
REPO_ROOT="$(cd "$MS_DIR/.." && pwd)"          # halo

echo "========================================================"
echo " real-robot deployment setup"
echo " repo:       $REPO_ROOT"
echo " robot_ssh:  ${ROBOT_SSH:-<not set>}"
echo " robot_repo: ${ROBOT_REPO:-<not set>}"
echo "========================================================"

if [[ -z "$ROBOT_SSH" ]]; then
    err "--robot_ssh is required (e.g. unitree@192.168.123.164)"
    exit 1
fi
if [[ -z "$ROBOT_REPO" ]]; then
    err "--robot_repo is required (e.g. /home/unitree/halo)"
    exit 1
fi
ROBOT_HOST="${ROBOT_SSH#*@}"

# ── 1. Mac mini: Python packages ──────────────────────────────────────────────
if [[ -n "${SKIP_LOCAL:-}" ]]; then
    section "[1/2] Mac mini: skipped (SKIP_LOCAL set)"
else
    section "[1/2] Mac mini: installing Python packages"
    # Reuse setup_dev_env.sh for the local box (idempotent; stops on failure and verifies imports).
    bash "$SCRIPT_DIR/setup_dev_env.sh"
    if [[ -z "${PYTHON:-}" && -n "${CONDA_ENV:-}" ]]; then
        set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; set -u
    fi
    PY="${PYTHON:-python}"
    # OAK camera (run_real.py --motiongen).
    $PY -m pip install -e "$REPO_ROOT/system[real]"
    $PY -c "import depthai; print('  depthai', depthai.__version__)"
    # The nodes find the tracker through the registry on the robot.
    if ! grep -qs '^export HALO_COMM_REGISTRY=' "$HOME/.zshrc"; then
        echo "export HALO_COMM_REGISTRY=$ROBOT_HOST:5560" >> "$HOME/.zshrc"
        echo "  added 'export HALO_COMM_REGISTRY=$ROBOT_HOST:5560' to ~/.zshrc"
    fi
    ok "Python packages installed on the Mac mini."
fi

# ── 2. Robot (NX): build the C++ real tracker over SSH ────────────────────────
if [[ -z "${SKIP_SYNC:-}" ]]; then
    section "[2/2] Robot (NX): copying the repo to $ROBOT_SSH:$ROBOT_REPO"
    rsync -az --exclude .git --exclude system/build --exclude __pycache__ \
        --exclude eval_closedloop_sim_multi_output --exclude docs \
        "$REPO_ROOT/" "$ROBOT_SSH:$ROBOT_REPO/"
fi

section "[2/2] Robot (NX): building C++ real tracker over SSH"
echo "  ssh $ROBOT_SSH"
echo "  repo: $ROBOT_REPO"
echo "  (enter the robot password once — run 'ssh-copy-id $ROBOT_SSH' beforehand to skip it)"

# Run ALL robot steps in ONE SSH session so you only type the password once.
# No PTY (-tt): the remote body arrives on stdin via this heredoc, so an
# interactive prompt couldn't read from the keyboard anyway.
# The heredoc body is sent to the remote bash via stdin (quoted so the local
# shell doesn't expand $ inside; config is passed via the ssh env).
POLICY_ONNX="${POLICY_ONNX:-tracker/deploy/models/g1/motiontrack_all_data.onnx}"
ssh "$ROBOT_SSH" \
    "export ROBOT_REPO='$ROBOT_REPO' POLICY_ONNX='$POLICY_ONNX' TRT_ARGS='${TRT_ARGS:-}' TRACKER_CMAKE_ARGS='${TRACKER_CMAKE_ARGS:-}' INSTALL_REALSENSE='${INSTALL_REALSENSE:-1}' REALSENSE_VERSION='${REALSENSE_VERSION:-v2.55.1}' ROBOT_CONDA_ENV='${ROBOT_CONDA_ENV:-}' TENSORRT_ROOT='${TENSORRT_ROOT:-}' && bash -s" <<'REMOTE_SCRIPT'
set -e

echo "[robot] SSH connected: $(hostname)"

# Verify the repo exists.
if [ ! -d "$ROBOT_REPO" ]; then
    echo "[ERROR] repo not found on robot: $ROBOT_REPO" >&2
    exit 1
fi
echo "[robot] repo OK: $ROBOT_REPO"

# Build packages (Ubuntu / JetPack). gcc-10: the system gcc 9 lacks C++20 <span>.
PKGS="gcc-10 g++-10 ninja-build pkg-config libzmq3-dev libyaml-cpp-dev git"
if sudo -n true 2>/dev/null; then
    sudo -n apt-get install -y $PKGS
else
    echo "[robot] no passwordless sudo: install these once by hand if the build fails:"
    echo "        sudo apt-get install -y $PKGS"
fi
# Unitree SDK2 and TensorRT come with the robot / JetPack; warn early if they are missing.
[ -d "${UNITREE_SDK2_ROOT:-/opt/unitree_robotics}" ] || \
    echo "[robot] WARNING: Unitree SDK2 not found at ${UNITREE_SDK2_ROOT:-/opt/unitree_robotics} (set UNITREE_SDK2_ROOT)"
ls /usr/lib/aarch64-linux-gnu/libnvinfer.so* >/dev/null 2>&1 || [ -n "${TENSORRT_ROOT:-}" ] || \
    echo "[robot] WARNING: TensorRT (libnvinfer) not found; set TENSORRT_ROOT"
# The first tracker build downloads Eigen 5.0 (CMake FetchContent), so the NX needs internet once.

# Modern protobuf: activate the conda env so build_motiontrack_real_zmq.sh adds
# $CONDA_PREFIX (matched protoc + libprotobuf) to CMAKE_PREFIX_PATH; a
# non-interactive SSH shell does not activate it on its own.
# Defaults to 'base'. Override with ROBOT_CONDA_ENV=<name>.
_conda_env="${ROBOT_CONDA_ENV:-base}"
for _c in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge" /opt/miniconda3 /opt/conda; do
    [ -f "$_c/etc/profile.d/conda.sh" ] && . "$_c/etc/profile.d/conda.sh" && break
done
if command -v conda >/dev/null 2>&1 && conda activate "$_conda_env"; then
    echo "[robot] conda env: $_conda_env ($(python -c 'import google.protobuf as p; print("protobuf", p.__version__)' 2>/dev/null || echo 'protobuf ?'))"
else
    echo "[ERROR] could not activate conda env '$_conda_env' on the robot (needs protobuf>=3.15)" >&2
    exit 1
fi

# The C++ build needs protoc + libprotobuf >= 3.15 (proto3 `optional`) from that conda env.
if ! protoc --version 2>/dev/null | awk '{split($2,v,"."); exit !(v[1]>3 || (v[1]==3 && v[2]>=15))}'; then
    echo "[robot] WARNING: protoc >= 3.15 not found in conda env '$_conda_env'; if the build fails run:"
    echo "        conda install -y -c conda-forge 'libprotobuf>=3.15'"
fi
# Build the real tracker.
echo "[robot] building motiontrack_real_zmq (JOBS=4)..."
cd "$ROBOT_REPO"
export PATH="$HOME/.local/bin:$PATH"

# cmake >= 3.22 (JetPack 5 system cmake is 3.16). Checked after conda + ~/.local
# are on PATH; installed (into ~/.local/bin) only when missing or too old.
if ! cmake --version 2>/dev/null | awk 'NR==1{split($3,v,"."); exit !(v[1]>3 || (v[1]==3 && v[2]>=22))}'; then echo "[robot] installing cmake>=3.22 (pip --user)..."; pip install --user 'cmake>=3.22,<4'; fi
# Depth build: the TensorRT plugins (stereo network) and librealsense2 (D435i).
case "$TRACKER_CMAKE_ARGS" in
*RT_DEPLOY_WITH_REALSENSE=ON*)
    sudo -n apt-get install -y libnvinfer-plugin-dev >/dev/null 2>&1 \
        || echo "[robot] WARNING: could not apt-install libnvinfer-plugin-dev (needs passwordless sudo)" >&2
    _rs_found=""
    _rs_have() {
        for _p in /usr /usr/local; do
            if [ -f "$_p/include/librealsense2/rs.hpp" ] && \
               ls "$_p"/lib/librealsense2.so* "$_p"/lib/aarch64-linux-gnu/librealsense2.so* >/dev/null 2>&1; then
                _rs_found="$_p"; return 0
            fi
        done
        return 1
    }
    _rs_have || sudo -n apt-get install -y librealsense2-dev >/dev/null 2>&1 || true
    if ! _rs_have && [ "$INSTALL_REALSENSE" != 0 ]; then
        # No arm64 package on the stock Jetson apt: build it once. System cmake and a clean env:
        # pip/conda cmake 4.x rejects librealsense's cmake_minimum_required, and conda's OpenSSL/libusb
        # must not be linked into a library installed in /usr/local. RSUSB backend = no kernel patch.
        _rs_src="$HOME/librealsense"
        echo "[robot] building librealsense $REALSENSE_VERSION from source (once, ~20-40 min)..."
        if sudo -n true 2>/dev/null; then
            sudo -n apt-get install -y git libssl-dev libusb-1.0-0-dev libudev-dev pkg-config >/dev/null 2>&1 || true
            [ -d "$_rs_src/.git" ] || git clone --depth 1 -b "$REALSENSE_VERSION" \
                https://github.com/IntelRealSense/librealsense.git "$_rs_src" \
                || echo "[robot] librealsense clone failed (needs GitHub access; or copy a clone to $_rs_src)" >&2
            if [ -d "$_rs_src/.git" ] && [ -x /usr/bin/cmake ]; then
                env -u CONDA_PREFIX -u CMAKE_PREFIX_PATH -u CC -u CXX PATH=/usr/sbin:/usr/bin:/sbin:/bin \
                    bash -c 'set -e
                        cd "$0" && rm -rf build && mkdir build && cd build
                        /usr/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DFORCE_RSUSB_BACKEND=ON \
                            -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_WITH_CUDA=OFF
                        make -j2
                        sudo -n make install
                        sudo -n ldconfig' "$_rs_src" \
                    && ( cd "$_rs_src" && sudo -n ./scripts/setup_udev_rules.sh ) >/dev/null 2>&1 || true
            fi
        else
            echo "[robot] no passwordless sudo: cannot install librealsense" >&2
        fi
    fi
    if ! _rs_have; then
        echo "[ERROR] librealsense2 not found on the robot; the depth build needs it." >&2
        echo "        Install it (Intel's librealsense repo, or build v2.55.1 from source with the RSUSB" >&2
        echo "        backend), or drop -DRT_DEPLOY_WITH_REALSENSE=ON for a proprio-only tracker." >&2
        exit 1
    fi
    echo "[robot] librealsense2 found in $_rs_found"
    ;;
esac

JOBS=4 bash system/cpp/scripts/build_motiontrack_real_zmq.sh $TRACKER_CMAKE_ARGS

echo "[robot] build complete: $ROBOT_REPO/system/build/g1_nx/motiontrack_real_zmq"

# Build the TensorRT .plan from the ONNX policy. A .plan is GPU/SM + TRT-version
# specific, so it MUST be built here on the NX. build_onnx_to_trt.sh writes it
# next to the ONNX (deploy/models/g1/<name>.plan). Skipped when the .plan is
# already present (it's slow to build) — delete it to force a rebuild.
PLAN="${POLICY_ONNX%.onnx}.plan"
if [ -f "$PLAN" ]; then
    echo "[robot] plan already exists, skipping onnx_to_trt: $ROBOT_REPO/$PLAN"
elif [ ! -f "$POLICY_ONNX" ]; then
    echo "[robot] WARNING: ONNX policy not found: $ROBOT_REPO/$POLICY_ONNX -- skipping the .plan." >&2
    echo "        copy your policy .onnx there (or pass POLICY_ONNX=<relative path>) and re-run." >&2
else
    echo "[robot] building TensorRT plan from $POLICY_ONNX (JOBS=4)..."
    JOBS=4 bash tracker/deploy_cpp/tools/build_onnx_to_trt.sh "$POLICY_ONNX" $TRT_ARGS
    echo "[robot] plan complete: $ROBOT_REPO/$PLAN"
fi
REMOTE_SCRIPT

ok "Robot build complete (tracker + TensorRT plan)."

# ── done ───────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
ok "Setup complete!"
echo ""
echo " Next: start the tracker on the NX, then run the task from this box:"
echo ""
echo "   # NX"
echo "   export HALO_COMM_REGISTRY=$ROBOT_HOST:5560"
echo "   sudo -E ./system/build/g1_nx/motiontrack_real_zmq --config tracker/deploy_cpp/configs/g1_v2.yaml \\"
echo "       --policy <policy>.plan --lock_memory --rt_cpu 1"
echo "   # Mac mini (new shell, so ~/.zshrc sets HALO_COMM_REGISTRY)"
echo "   python system/halo_system/scripts/run_real.py --task_config <task>.yaml --motiongen --orchestrator --web"
echo ""
echo " Full steps: system/README.md#run-real"
echo " Gamepad: Start → default pose, A → stand up, B → start generation, Select → e-stop"
echo "========================================================"
