#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Shard driver for the Isaac render (configs/pipeline/render.env: prod, eval):
# spreads the shard's scene-free specs over SCENES_PER_SHARD scenes from SCENE_MANIFEST and runs
# render/render_episodes.py once per scene. Shared steps: render_common.sh; env vars: docs/data_pipeline.md.

set -e
export PYTHONUNBUFFERED=1
source "$(dirname "${BASH_SOURCE[0]}")/render_common.sh"
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}"

render_dirs
render_shard_vars Isaac
if [ -z "${ROBOT_THOR_URDF:-}" ] && [ -f "$DATA_DIR/g1_thor/urdf/g1_29dof_v2_thor.urdf" ]; then
    export ROBOT_THOR_URDF="$DATA_DIR/g1_thor/urdf/g1_29dof_v2_thor.urdf"   # staged by setup_render_gpu.sh
fi
pull_resume
rollback_resume 1 oak.mp4 tpv.mp4 meta.json traj.npz
pull_specs
if command -v python3 >/dev/null 2>&1; then MAP_PY=(python3)
elif command -v python >/dev/null 2>&1; then MAP_PY=(python)
else MAP_PY=("$ISAACLAB_SH" -p); fi
build_scene_map "${MAP_PY[@]}"
start_uploader
render_scenes isaac "$ISAACLAB_SH" -p
final_sync || exit 1
check_episodes oak.mp4 tpv.mp4 meta.json traj.npz || exit 1
echo "=== multiscene done: $(ls "$OUTPUT_DIR/episodes/" 2>/dev/null | wc -l) episodes, $N_UNIQUE scenes ==="
