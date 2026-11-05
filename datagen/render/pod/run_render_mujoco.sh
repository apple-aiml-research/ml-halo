#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Shard driver for the MuJoCo render (configs/pipeline/render.env, mujoco): same scene assignment,
# --shard-idx and placement as run_render_isaac.sh (shared steps: render_common.sh), MuJoCo backend.

set -e
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
source "$(dirname "${BASH_SOURCE[0]}")/render_common.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then PY="$VIRTUAL_ENV/bin/python"
else PY=python3; fi
echo "Using python: $PY"

render_dirs
render_shard_vars MuJoCo
pull_resume
rollback_resume 256 fpv.mp4 tpv.mp4
pull_specs
build_scene_map "$PY"
start_uploader
render_scenes mujoco "$PY"
final_sync || exit 1
check_episodes oak.mp4 tpv.mp4 meta.json traj.npz || exit 1
echo "=== multiscene done: $(ls "$OUTPUT_DIR/episodes/" 2>/dev/null | wc -l) episodes, $N_UNIQUE scenes ==="
