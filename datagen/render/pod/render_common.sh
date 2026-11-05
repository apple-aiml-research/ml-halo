#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# render_common.sh -- shared part of the render runners (run_render_isaac.sh, run_render_mujoco.sh).
# Source it; it defines functions only. A runner calls, in order:
#   render_dirs; render_shard_vars <engine>; pull_resume; rollback_resume ...; pull_specs; build_scene_map;
#   start_uploader; render_scenes; final_sync.

# OUTPUT_DIR (set by launch/run_stage.sh) gets episodes/ and dataset/; inputs go under RENDER_DATA_DIR.
render_dirs() {
    OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/render}"
    DATA_DIR="${RENDER_DATA_DIR:-$OUTPUT_DIR/inputs}"
    SPEC_DIR="$DATA_DIR/specs"
    TMP_DIR="$DATA_DIR/tmp"
    export OUTPUT_DIR
    mkdir -p "$OUTPUT_DIR" "$SPEC_DIR" "$TMP_DIR"
}

render_shard_vars() {
    PUSH_TO_S3="${PUSH_TO_S3:-1}"   # 0 = keep the output in OUTPUT_DIR only (no resume, no upload)
    SHARD_IDX="${SHARD_IDX:-${RANK:-0}}"
    SHARD_TAG=$(printf '%03d' "$SHARD_IDX")
    # resume state per backend: Isaac and MuJoCo episodes never mix
    local backend_sfx=""; [ "$1" = MuJoCo ] && backend_sfx="_mujoco"
    SHARED_PREFIX="${SHARED_PREFIX_ROOT:-${HALO_S3_ROOT}/rendered_output/${RENDER_BATCH}${backend_sfx}}/shard_${SHARD_TAG}"
    : "${DIRECT_WRITE_DEST:?DIRECT_WRITE_DEST (dataset prefix) not set}"
    : "${EPS_PER_SHARD:?EPS_PER_SHARD not set}"
    DATASET_DEST="${HALO_S3_ROOT}/${DIRECT_WRITE_DEST}"
    echo "=== $1 render ==="
    echo "  shard:        ${SHARD_IDX}/${TOTAL_SHARDS:-?}"
    echo "  render_batch: ${RENDER_BATCH}"
    echo "  shared:       ${SHARED_PREFIX}"
    echo "  dataset dest: ${DATASET_DEST}"
}

pull_resume() {
    [ "$PUSH_TO_S3" = "1" ] && command -v halo_s3 &>/dev/null || return 0
    halo_s3 cp --recursive "${SHARED_PREFIX}/episodes/" "$OUTPUT_DIR/episodes/" 2>/dev/null || true
    # dataset episode ids are shard*1000 + i (the gen's spec dir names)
    local lo=$((SHARD_IDX * 1000)) hi=$((SHARD_IDX * 1000 + EPS_PER_SHARD - 1)) sub ep p
    local subs=(data episode_specs videos/observation.images.fpv_cam videos/observation.images.tpv_cam)
    for sub in "${subs[@]}" meta; do mkdir -p "$OUTPUT_DIR/dataset/$sub"; done
    # no meta pull: the driver rewrites each scene's meta/shard_<i*SCENES_PER_SHARD+k>/ from the resumed episodes,
    # and meta/shard_<i>/ belongs to another render shard (pulling and re-uploading it clobbered that shard's meta)
    if [ $((lo % 100)) -eq 0 ] && [ $((EPS_PER_SHARD % 100)) -eq 0 ]; then
        local inc=()
        for p in $(seq $((lo / 100)) $((hi / 100))); do inc+=(--include "$(printf 'episode_%04d*' "$p")"); done
        echo "[resume] fetching shard-local dataset files for ep ${lo}..${hi} (${#inc[@]} prefix patterns x 4 dirs)"
        for sub in "${subs[@]}"; do
            halo_s3 cp --recursive --exclude '*' "${inc[@]}" \
                "${DATASET_DEST}/${sub}/" "$OUTPUT_DIR/dataset/${sub}/" 2>/dev/null || true
        done
    else
        echo "[resume] fetching shard-local dataset files for ep ${lo}..${hi} (per episode)"
        for ep in $(seq -f "%06g" "$lo" "$hi"); do
            halo_s3 cp "${DATASET_DEST}/data/episode_${ep}.parquet" "$OUTPUT_DIR/dataset/data/" 2>/dev/null || true
            halo_s3 cp "${DATASET_DEST}/episode_specs/episode_${ep}.json" "$OUTPUT_DIR/dataset/episode_specs/" 2>/dev/null || true
            for sub in videos/observation.images.fpv_cam videos/observation.images.tpv_cam; do
                halo_s3 cp "${DATASET_DEST}/${sub}/episode_${ep}.mp4" "$OUTPUT_DIR/dataset/${sub}/" 2>/dev/null || true
            done
        done
    fi
}

# rollback_resume <min_bytes> <file>...: of the pulled episodes that have every <file> (>= min_bytes),
# delete the last ROLLBACK_EPS (their upload may have been cut off) so they are rendered again.
rollback_resume() {
    [ "$PUSH_TO_S3" = "1" ] && [ -d "$OUTPUT_DIR/episodes" ] || return 0
    local min=$1 d f ok n_done done_eps; shift
    done_eps=$(cd "$OUTPUT_DIR/episodes" && for d in */; do
        d=${d%/}; ok=1
        for f in "$@"; do [ "$(wc -c < "$d/$f" 2>/dev/null || echo 0)" -ge "$min" ] || ok=0; done
        [ "$ok" = 1 ] && echo "$d"
    done | sort)
    n_done=$(printf '%s\n' "$done_eps" | grep -c . || true)
    echo "[resume] ${n_done:-0} complete episodes found"
    if [ "${n_done:-0}" -gt 0 ] && [ "${ROLLBACK_EPS}" -gt 0 ]; then
        printf '%s\n' "$done_eps" | tail -n "$ROLLBACK_EPS" | while read -r d; do
            [ -n "$d" ] && rm -rf "${OUTPUT_DIR:?}/episodes/$d"
        done
        echo "[resume] rolled back $ROLLBACK_EPS tail eps"
    fi
}

pull_specs() {
    : "${SPEC_S3_PATH:?SPEC_S3_PATH not set (the render does not generate specs)}"
    local src="${HALO_S3_ROOT}/${SPEC_S3_PATH}/shard_${SHARD_TAG}" a
    echo "[specs] pulling from $src"
    for a in 1 2 3 4 5 6 7 8; do
        halo_s3 sync "$src/" "$SPEC_DIR/" && break
        echo "[specs] pull attempt $a incomplete; retrying after backoff" >&2
        sleep $((15 * a))
    done
    echo "[specs] $(ls -d "$SPEC_DIR"/episode_* 2>/dev/null | wc -l) episode dirs total"
}

_has() { [ -f "$INF_SCENE_DIR/$1" ] && echo y || echo MISSING; }

pull_scene() {
    INF_SCENE_DIR="$DATA_DIR/infinigen_scene/$1"
    INF_SCENE_KEY="${SCENE_S3_PATH}/$1"
    mkdir -p "$INF_SCENE_DIR"
    halo_s3 cp --recursive --exclude '*' --include 'asset/*' --include 'scene_analysis.json' \
        --include 'views_64.json' --include 'anchor.json' --include 'gen_result.json' \
        "${HALO_S3_ROOT}/${SCENE_S3_PATH}/$1/" "$INF_SCENE_DIR/" 2>&1 | tail -1
    echo "[scene] $1 -> $INF_SCENE_DIR (usdc=$(_has asset/scan.usdc) obj=$(_has asset/scan.obj) analysis=$(_has scene_analysis.json))"
}

build_scene_map() {   # args: python command
    : "${SCENE_MANIFEST:?SCENE_MANIFEST (scene list) not set}"
    MAP_FILE="$TMP_DIR/ms_scene_map.tsv"
    local raw="$TMP_DIR/ms_scene_map.raw"
    "$@" common/multiscene_map.py "$SPEC_DIR" "$raw" \
        --manifest "$SCENE_MANIFEST" --shard "$SHARD_IDX" --sps "$SCENES_PER_SHARD"
    grep -E '^scene_[0-9]+	' "$raw" > "$MAP_FILE" || cp "$raw" "$MAP_FILE"   # well-formed rows only
    UNIQUE_SCENES=$(cut -f1 "$MAP_FILE" | sort -u)
    N_UNIQUE=$(printf '%s\n' "$UNIQUE_SCENES" | grep -c . || true)
    echo "[scene] shard $SHARD_IDX covers $N_UNIQUE unique scene(s): $(printf '%s ' $UNIQUE_SCENES)"
}

scene_npy_dir() {
    TMP_NPY="$TMP_DIR/ms_npy_$1"; rm -rf "$TMP_NPY"; mkdir -p "$TMP_NPY"
    local s ep
    while IFS=$'\t' read -r s ep; do
        [ "$s" = "$1" ] && ln -sfn "$ep" "$TMP_NPY/$(basename "$ep")"
    done < "$MAP_FILE"
    echo "[scene] $1: $(ls -d "$TMP_NPY"/episode_* 2>/dev/null | wc -l) episodes"
}

start_uploader() {
    if [ "$PUSH_TO_S3" = "1" ] && command -v halo_s3 &>/dev/null; then
        (while sleep "$ROLLING_UPLOAD_SECS"; do
            halo_s3 cp --recursive "$OUTPUT_DIR/episodes/" "${SHARED_PREFIX}/episodes/" || true
            halo_s3 cp --recursive "$OUTPUT_DIR/dataset/"   "${DATASET_DEST}/"            || true
        done) >/dev/null 2>&1 &
        UPLOADER_PID=$!
        trap "kill $UPLOADER_PID 2>/dev/null || true" EXIT INT TERM
    fi
}

_retry() {   # 3 tries, 30 s / 60 s backoff
    local n=1 rc=0
    while :; do
        "$@" && return 0
        rc=$?
        [ "$n" -ge 3 ] && return "$rc"
        echo "[push] attempt $n/3 failed (rc=$rc): $*; retrying in $((30 * n)) s" >&2
        sleep $((30 * n))
        n=$((n + 1))
    done
}

final_sync() {
    [ -n "${UPLOADER_PID:-}" ] && { kill "$UPLOADER_PID" 2>/dev/null || true; wait "$UPLOADER_PID" 2>/dev/null || true; }
    if [ "$PUSH_TO_S3" = "1" ] && command -v halo_s3 &>/dev/null; then
        _retry halo_s3 cp --recursive "$OUTPUT_DIR/episodes/" "${SHARED_PREFIX}/episodes/" || {
            echo "[push] FAILED: $OUTPUT_DIR/episodes/ -> ${SHARED_PREFIX}/episodes/" >&2; return 1; }
        _retry halo_s3 cp --recursive "$OUTPUT_DIR/dataset/" "${DATASET_DEST}/" || {
            echo "[push] FAILED: $OUTPUT_DIR/dataset/ -> ${DATASET_DEST}/" >&2; return 1; }
    else
        echo "[push] skipped (PUSH_TO_S3=$PUSH_TO_S3); rendered output is in $OUTPUT_DIR"
    fi
}

# check_episodes <file>...: fail unless the shard has as many complete episodes (every <file> present) as
# render_scenes gave it; the renderers can exit 0 after failing a scene.
check_episodes() {
    local d f ok n=0
    for d in "$OUTPUT_DIR"/episodes/*/; do
        ok=1
        for f in "$@"; do [ -s "$d$f" ] || ok=0; done
        n=$((n + ok))
    done
    echo "[check] $n complete episodes, expected ${EXPECTED_EPS:-0}"
    [ "$n" -ge "${EXPECTED_EPS:-0}" ] && [ "$n" -gt 0 ] || { echo "[check] FAILED: shard is incomplete" >&2; return 1; }
}

# render_scenes <backend> <python cmd...>: render_episodes.py once per scene of the shard map, each scene with
# its own --shard-idx (part of the per-episode seeds).
render_scenes() {
    local backend=$1 sub=0 scene n; shift
    EXPECTED_EPS=0
    for scene in $UNIQUE_SCENES; do
        pull_scene "$scene"
        scene_npy_dir "$scene"
        n=$(ls -d "$TMP_NPY"/episode_* 2>/dev/null | wc -l)
        [ "${MAX_TRAJS:-0}" -gt 0 ] && [ "$n" -gt "$MAX_TRAJS" ] && n=$MAX_TRAJS
        EXPECTED_EPS=$((EXPECTED_EPS + n))
        "$@" render/render_episodes.py --backend "$backend" --npy-dir "$TMP_NPY" \
            --scene-dir "$INF_SCENE_DIR" --scene-key "$INF_SCENE_KEY" \
            --shard-idx $(( SHARD_IDX * SCENES_PER_SHARD + sub )) --output-dir "$OUTPUT_DIR" \
            --max-trajs "${MAX_TRAJS:-0}" --render-batch "$RENDER_BATCH" --target-fps "$TARGET_FPS"
        sub=$((sub + 1))
    done
}
