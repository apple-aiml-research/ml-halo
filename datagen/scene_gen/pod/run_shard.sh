#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# run_shard.sh -- one shard of a scene_gen sweep (STAGE_CMD of configs/pipeline/scene_gen.env):
# generate, bake, project the GT frames onto the bake, convert to USD, analyze, upload.
#
# Pipelined: Infinigen's coarse task is single-threaded CPU work, so a background pool (gen_worker.py)
# generates scenes while this loop runs the GPU stages of the previous one. Per scene:
#   prep_and_gt -> anchor.json + views_64.json (every later stage needs $ANCHOR_JSON) + the GT frames
#   bake.py     -> scan.obj/.mtl/tex; scene.blend is deleted right after (up to 2.6 GB each)
#   project_gt  -> the GT frames projected onto the baked textures -> asset/; obj_to_usdc -> asset/scan.usdc
#   analyze_scene -> scene_analysis.json + scene_analysis_topdown.png (the render's placement and TPV choice);
#                  a scene it cannot analyze still ships (warning)
#   upload, then the marker $OUT_S3/_done/<scene>.json LAST: a half-uploaded scene has no marker
#   and is redone by the next run of the shard.
# Blender and Infinigen come from scene_gen/env/setup_scene_gen.sh (the STAGE_SETUP); JSON work uses Blender's python.
set -u
DONE_PREFIX_NAME=_done
SG_T0=$(date +%s)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S="$(cd "$HERE/.." && pwd)/scripts"
DATAGEN="$(cd "$HERE/../.." && pwd)"
W=${WORK_DIR:-${OUTPUT_DIR:-/tmp/scene_gen}/work}   # needs ~300 GB
mkdir -p "$W"
: "${OUT_S3:?}"; : "${MANIFEST:?MANIFEST (scene list json) must be set}"
SHARD_IDX=${SHARD_IDX:-0}
N_VIEWS=$((N_INTERIOR_VIEWS + N_ORBIT_VIEWS))

sg_say() { echo "[$(date +%H:%M:%S) +$(( $(date +%s) - SG_T0 ))s] $*"; }
sg_say "=== scene_gen shard $SHARD_IDX/$TOTAL_SHARDS ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -2

source "$HERE/../../launch/ensure_storage.sh" || exit 2

BLENDER_DIR="${BLENDER_DIR:-/opt/blender52}"
BL="$BLENDER_DIR/blender"
BLPY=$(ls "$BLENDER_DIR"/*/python/bin/python3* 2>/dev/null | head -1)
[ -x "$BL" ] && [ -n "$BLPY" ] || { sg_say "FATAL no Blender under $BLENDER_DIR (run scene_gen/env/setup_scene_gen.sh)"; exit 2; }
export INFINIGEN_ROOT="${INFINIGEN_ROOT:-/opt/infinigen_src}" IPY="${INFINIGEN_ENV:-/opt/infinigen_env}/bin/python"
"$IPY" -c "import bpy, numpy, shapely, infinigen; print('ENV_OK', bpy.app.version_string)" \
    || { sg_say "FATAL the Infinigen env ($IPY) does not import (run scene_gen/env/setup_scene_gen.sh)"; exit 2; }
if ! "$IPY" -c "import infinigen.core.constraints.example_solver.room.predefined" >/dev/null 2>&1; then
    export NO_FLOOR_PLAN=1   # no PredefinedFloorPlanSolver: room size from the per-seed variation
    sg_say "no PredefinedFloorPlanSolver -> NO_FLOOR_PLAN=1"
fi

# this shard's scenes not yet marked done, as idx=name lines
halo_s3 ls --recursive "$OUT_S3/$DONE_PREFIX_NAME/" > "$W/done.txt" 2>/dev/null || : > "$W/done.txt"
"$BLPY" - "$MANIFEST" "$SHARD_IDX" "$W/done.txt" > "$W/todo.txt" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1])); shard = int(sys.argv[2]); done = open(sys.argv[3]).read()
m = doc.get("shard_members")
idxs = m[shard] if m and shard < len(m) else [e["idx"] for e in doc["scenes"] if e.get("shard") == shard]
if not idxs:
    sys.exit(3)
name = {e["idx"]: e["name"] for e in doc["scenes"]}
print("\n".join(f"{i}={name[i]}" for i in idxs if f"{name[i]}.json" not in done))
PY
[ $? -eq 3 ] && { sg_say "FATAL shard $SHARD_IDX has no scenes in $MANIFEST"; exit 2; }
IDX_NAME=(); PENDING=()
while IFS='=' read -r _i _n; do [ -n "${_i:-}" ] && { IDX_NAME[$_i]="$_n"; PENDING+=("$_i"); }; done < "$W/todo.txt"
[ "${#PENDING[@]}" -gt 0 ] || { sg_say "all scenes for shard $SHARD_IDX already complete"; exit 0; }
sg_say "todo: ${PENDING[*]}"

GENROOT="$W/gen"; QUEUE="$W/queue"
mkdir -p "$GENROOT" "$QUEUE"
"$IPY" "$S/gen_worker.py" --manifest "$MANIFEST" --scenes "$(IFS=,; echo "${PENDING[*]}")" \
    --out-root "$GENROOT" --queue "$QUEUE" --workers "$GEN_WORKERS" > "$W/gen_pool.log" 2>&1 &
GEN_POOL_PID=$!

N_OK=0; N_FAIL=0
while [ "${#PENDING[@]}" -gt 0 ]; do
    ELAPSED=$(( $(date +%s) - SG_T0 ))
    if [ "$ELAPSED" -gt "$BUDGET_S" ]; then   # stop STARTING scenes; never die mid-bake
        sg_say "budget guard: ${ELAPSED}s > ${BUDGET_S}s -- not starting more scenes"
        break
    fi
    IDX=""; NAME=""
    for _p in "${PENDING[@]}"; do
        _n="${IDX_NAME[$_p]}"
        if [ -f "$QUEUE/$_n.ready" ] || [ -f "$QUEUE/$_n.failed" ]; then IDX="$_p"; NAME="$_n"; break; fi
    done
    if [ -z "$IDX" ]; then
        if ! kill -0 "$GEN_POOL_PID" 2>/dev/null; then
            sg_say "generation pool exited; never produced: ${PENDING[*]}"
            for _p in "${PENDING[@]}"; do
                halo_s3 cp "$GENROOT/${IDX_NAME[$_p]}/gen.log" "$OUT_S3/_failed/${IDX_NAME[$_p]}/gen.log" >/dev/null 2>&1
                N_FAIL=$((N_FAIL + 1))
            done
            break
        fi
        sleep 15
        continue
    fi
    _new=(); for _p in "${PENDING[@]}"; do [ "$_p" = "$IDX" ] || _new+=("$_p"); done
    PENDING=(${_new[@]+"${_new[@]}"})
    if [ ! -f "$QUEUE/$NAME.ready" ]; then
        sg_say "SKIP $NAME (generation failed)"
        tail -20 "$GENROOT/$NAME/gen.log" 2>/dev/null | sed 's/^/    /'
        halo_s3 cp "$GENROOT/$NAME/gen.log" "$OUT_S3/_failed/$NAME/gen.log" >/dev/null 2>&1
        N_FAIL=$((N_FAIL + 1))
        continue
    fi

    BLEND="$GENROOT/$NAME/scene.blend"
    D="$W/scenes/$NAME"; O="$D/out"
    rm -rf "$D"; mkdir -p "$O"
    sg_say "=== $NAME (idx $IDX): blend $(du -h "$BLEND" 2>/dev/null | cut -f1) ==="
    ( while sleep "$ROLLING_UPLOAD_SECS"; do   # rolling upload of logs/metadata while the scene runs
          halo_s3 cp --recursive "$O/" "$OUT_S3/$NAME/" --exclude "bake/*" --exclude "proj1/*" >/dev/null 2>&1
      done ) &
    UPLOADER_PID=$!
    trap 'kill $UPLOADER_PID 2>/dev/null' EXIT INT TERM
    SCENE_T0=$(date +%s)
    STAGE_OK=1

    sg_say "[1/5] prep_and_gt (views=$N_VIEWS gt=$N_PROJ_VIEWS @ ${GT_SPP}spp)"
    "$BL" -b "$BLEND" -P "$S/prep_and_gt.py" -- "$O" "$NAME" \
        "$N_INTERIOR_VIEWS" "$N_ORBIT_VIEWS" "$GT_SPP" "$RENDER_RES" "$IDX" > "$O/prep.log" 2>&1
    grep -E "^\[prep|PREP_OK|WARN" "$O/prep.log" | tail -12
    if [ ! -s "$O/anchor.json" ] || [ ! -s "$O/views_${N_VIEWS}.json" ]; then
        sg_say "FAIL $NAME: prep produced no anchor/views"; tail -30 "$O/prep.log" | sed 's/^/    /'
        STAGE_OK=0
    fi
    if [ "$STAGE_OK" = "1" ]; then
        export ANCHOR_JSON="$O/anchor.json"
        sg_say "[2/5] bake ${BAKE_SPP}spp ${BAKE_MAXRES}px"
        "$BL" -b "$BLEND" -P "$S/bake.py" -- "$O/bake" "$BAKE_SPP" "$BAKE_MAXRES" > "$O/bake.log" 2>&1
        grep -E "restored world transforms|recalculated consistent outward normals" "$O/bake.log" | tail -2
        if [ ! -s "$O/bake/scan.obj" ]; then
            sg_say "FAIL $NAME: bake produced no scan.obj"; tail -30 "$O/bake.log" | sed 's/^/    /'
            STAGE_OK=0
        fi
    fi
    if [ "$STAGE_OK" = "1" ]; then
        rm -f "$BLEND"
        sg_say "[3/5] released scene.blend"
        sg_say "[4/5] project_gt (ss=$PROJ_SS) + scan.usdc"
        "$BLPY" - "$O/views_${N_VIEWS}.json" "$O/prep_manifest.json" "$O/gt" "$O/multi.json" <<'PY'
import json, os, sys
views = json.load(open(sys.argv[1]))["views"]
idxs = json.load(open(sys.argv[2]))["proj_views"]
gtdir, out = sys.argv[3], sys.argv[4]
spec = [{"gt": os.path.abspath(os.path.join(gtdir, f"gt_v{i:02d}.png")),
         "pos": views[i]["pos"], "look": views[i]["look_at"]}
        for i in idxs if os.path.isfile(os.path.join(gtdir, f"gt_v{i:02d}.png"))]
json.dump(spec, open(out, "w"), indent=1)
PY
        rm -rf "$O/proj1" "$O/asset"
        "$BLPY" "$S/project_gt.py" --obj "$O/bake/scan.obj" --multi "$O/multi.json" \
            --out "$O/proj1" --ss "$PROJ_SS" > "$O/project.log" 2>&1
        mkdir -p "$O/asset"   # no projection: ship the plain bake
        if [ -s "$O/proj1/scan.obj" ]; then cp -r "$O/proj1/." "$O/asset/"; else cp -r "$O/bake/." "$O/asset/"; fi
        "$BLPY" "$S/obj_to_usdc.py" "$O/asset/scan.obj" "$O/asset/scan.usdc" > "$O/usdc.log" 2>&1
        if [ ! -s "$O/asset/scan.obj" ] || [ ! -s "$O/asset/scan.usdc" ]; then
            sg_say "FAIL $NAME: no asset/scan.obj or asset/scan.usdc"; tail -20 "$O/usdc.log" | sed 's/^/    /'
            STAGE_OK=0
        fi
        cp "$GENROOT/$NAME/gen.log" "$GENROOT/$NAME/gen_result.json" "$O/" 2>/dev/null
    fi
    if [ "$STAGE_OK" = "1" ]; then
        sg_say "[5/5] analyze_scene"
        mkdir -p "$W/analyze" && ln -sfn "$O" "$W/analyze/$NAME"   # the analysis is named after its scene dir
        PYTHONPATH="$DATAGEN" "$BLPY" -m render.synthetic_scene.analyze_scene \
            --scene-dir "$W/analyze/$NAME" > "$O/analyze.log" 2>&1
        rm -f "$W/analyze/$NAME"
        if [ ! -s "$O/scene_analysis.json" ]; then
            sg_say "WARN $NAME: no scene_analysis.json"
            tail -5 "$O/analyze.log" | sed 's/^/    /'
        fi
    fi

    kill "$UPLOADER_PID" 2>/dev/null; trap - EXIT INT TERM
    rm -rf "$O/bake/tex_src" "$O/proj1"
    mkdir -p "$O/logs" && mv "$O"/*.log "$O/logs/" 2>/dev/null
    halo_s3 cp --recursive "$O/" "$OUT_S3/$NAME/" --exclude "bake/tex/*" --exclude "proj1/*" 2>&1 | tail -1
    SCENE_S=$(( $(date +%s) - SCENE_T0 ))
    if [ "$STAGE_OK" = "1" ]; then
        printf '{"idx": %d, "name": "%s", "scene_s": %d, "shard": %d, "n_views": %d}' \
            "$IDX" "$NAME" "$SCENE_S" "$SHARD_IDX" "$N_VIEWS" > "$W/marker.json"
        halo_s3 cp "$W/marker.json" "$OUT_S3/$DONE_PREFIX_NAME/$NAME.json" 2>&1 | tail -1
        N_OK=$((N_OK + 1))
        sg_say "=== DONE $NAME in ${SCENE_S}s ==="
    else
        N_FAIL=$((N_FAIL + 1))
        sg_say "=== FAILED $NAME after ${SCENE_S}s (no marker: retried next run) ==="
    fi
    rm -rf "$D" "$GENROOT/$NAME"
done

kill "$GEN_POOL_PID" 2>/dev/null
sg_say "=== shard $SHARD_IDX complete: ok=$N_OK failed=$N_FAIL, $(( ($(date +%s) - SG_T0) / 60 )) min ==="
