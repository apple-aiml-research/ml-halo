#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Motion generation stage entrypoint (launch/run_stage.sh gen): Kimodo sit-on-chair
# trajectories (walk -> turn -> sit, best-of-N gated by motion_check) via generate_sit_traj.py.
#
# Scene-free (motion-first): the chair sits at the origin and no scene is pulled; the render
# stage places each motion into a room.
# Output: per-episode dirs episode_NNNN/{episode_spec.json, g1_qpos.npy, meta.json, plan.npy}
# under ${HALO_S3_ROOT}/${SPEC_S3_PATH}/shard_NNN/ for the render stage.
#
# Env vars (set by configs/pipeline/gen.env):
#   SHARD_ID / RANK, TOTAL_SHARDS, EPS_PER_SHARD          — sharding (episodes per shard)
#   SPEC_S3_PATH                                          — output prefix
#   SEED_BASE, TIER, STEPS, BATCH, N_MAX_RETRY            — Kimodo / gating knobs
#   ROBOT_VARIANT, CAM_KIND                               — g1_29dof_thor / fisheye
#   HALO_S3_ROOT                                          — data root (s3://bucket[/prefix] or a local dir)
set -e
set -o pipefail
export PYTHONUNBUFFERED=1

source launch/ensure_conda.sh
eval "$(conda shell.bash hook)"
conda activate kimodo
export PATH="$PWD/launch:$HOME/.local/bin:/usr/local/bin:$PATH"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export LOCAL_CACHE=true TEXT_ENCODER_MODE=local MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NEWTON_CACHE_PATH="$HOME/.cache/newton"

: "${HALO_S3_ROOT:?set HALO_S3_ROOT (s3://bucket[/prefix] or a local dir)}"
RANK="${SHARD_ID:-${RANK:-0}}"
TOTAL="${TOTAL_SHARDS:-50}"
NPS="${EPS_PER_SHARD:-4}"
SHARD_TAG=$(printf '%03d' "$RANK")
HF_CACHE_DIR="$HF_HUB_CACHE"

CKPT_DIR=/tmp/kimodo_ckpt; mkdir -p "$CKPT_DIR"
KSNAP=$(dirname "$(find "$HF_CACHE_DIR/models--nvidia--Kimodo-G1-RP-v1/snapshots" -name config.yaml 2>/dev/null | head -1)")
if [ -n "$KSNAP" ] && [ -f "$KSNAP/config.yaml" ]; then
    ln -sfn "$KSNAP" "$CKPT_DIR/Kimodo-G1-RP-v1"
    export CHECKPOINT_DIR="$CKPT_DIR"
    echo "[run] CHECKPOINT_DIR=$CKPT_DIR -> $KSNAP"
else
    echo "[run] WARN: kimodo snapshot config.yaml not found; falling back to HF resolve"
fi

TED=/tmp/text_encoders; mkdir -p "$TED/meta-llama" "$TED/McGill-NLP"
_snap() { ls -d "$HF_CACHE_DIR/$1/snapshots/"*/ 2>/dev/null | head -1; }
for pair in \
  "models--meta-llama--Meta-Llama-3-8B-Instruct:$TED/meta-llama/Meta-Llama-3-8B-Instruct" \
  "models--McGill-NLP--LLM2Vec-Meta-Llama-3-8B-Instruct-mntp:$TED/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp" \
  "models--McGill-NLP--LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised:$TED/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"; do
    src=$(_snap "${pair%%:*}"); dst="${pair##*:}"
    [ -n "$src" ] && ln -sfn "${src%/}" "$dst" && echo "[run] text-enc link ${dst##*/} -> ${src%/}"
done
export TEXT_ENCODERS_DIR="$TED"

OUT_DIR="${OUTPUT_DIR:-/tmp/out}/shard_${SHARD_TAG}"; mkdir -p "$OUT_DIR"

: "${SPEC_S3_PATH:?SPEC_S3_PATH must be set}"
DEST="${HALO_S3_ROOT}/${SPEC_S3_PATH}/shard_${SHARD_TAG}"

echo "[resume] pulling prior specs from $DEST (if any)"
halo_s3 cp --recursive "$DEST/" "$OUT_DIR/" >/dev/null 2>&1 || true
echo "[resume] $(ls -d "$OUT_DIR"/episode_* 2>/dev/null | wc -l) episodes already present"

UPLOAD_SECS="${ROLLING_UPLOAD_SECS:-300}"
( while true; do sleep "$UPLOAD_SECS";
    halo_s3 cp --recursive "$OUT_DIR/" "$DEST/" || true; done ) >/dev/null 2>&1 &
UPLOADER_PID=$!
trap "kill $UPLOADER_PID 2>/dev/null || true" EXIT INT TERM

echo "[run] shard ${RANK}/${TOTAL}, ${NPS} sit trajs, gated, resume + rolling upload"
GEN_ARGS=()
_opt() { [ -n "${!1:-}" ] && GEN_ARGS+=("$2" "${!1}"); return 0; }
_opt SEED_BASE --seed;            _opt N_MAX_RETRY --n-max-retry;   _opt BATCH --batch
_opt TIER --tier;                 _opt STEPS --steps;               _opt CHECK_WORKERS --check-workers
_opt CAM_KIND --cam-kind;         _opt ROBOT_VARIANT --robot-variant
_opt VDR_LEVEL --vdr-level;       _opt ASSET_COLOR --asset-color;   _opt CHAIR_SET --chair-set
_opt SINGLE_STANDING_KF --single-standing-kf;  _opt TURN_CONSTRAINT --turn-constraint
_opt TURN_NF --turn-nf;           _opt TURN_END_BACK_CM --turn-end-back-cm
_opt STAGE_RETRY --stage-retry;   _opt SIT_NF --sit-nf;             _opt SEAM_K --seam-k
[ "${TURN_PIN_START:-1}" = "0" ] && GEN_ARGS+=(--no-turn-pin-start)
echo "[run] generate_sit_traj.py ${GEN_ARGS[*]}"
PYTHONPATH=. python motion_gen/generate_sit_traj.py \
    --n "${NPS}" --shard-id "${RANK}" --device cuda \
    "${GEN_ARGS[@]}" \
    --skip-existing --out "$OUT_DIR"

kill "$UPLOADER_PID" 2>/dev/null || true
halo_s3 cp --recursive "$OUT_DIR/" "$DEST/" 2>&1 | tail -2
echo "=== sit-syn gen done: $(ls -d "$OUT_DIR"/episode_* 2>/dev/null | wc -l) episodes -> $DEST ==="
