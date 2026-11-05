#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# run_stage.sh -- run one pipeline stage (one shard of it) here: on a laptop, a GPU box, or inside any
# container / scheduler task. Schedulers only start it N times with different --shard values.
#
#   launch/run_stage.sh <stage> [variant] [--shard I] [--num-shards N] [KEY=VALUE ...] [--no-setup] [--dry-run]
#
#   <stage>    configs/pipeline/<stage>.env  (scene_gen, gen, render)
#   [variant]  exported as VARIANT; the env file picks its per-variant values from it (e.g. render eval)
#   KEY=VALUE  overrides, visible to the env file (e.g. SCENES=10) and applied again after it
#
# Environment:
#   HALO_S3_ROOT  required. s3://bucket[/prefix] (AWS_* credentials, AWS_ENDPOINT_URL for MinIO/R2/...)
#                 or a local/NFS dir. Every stage reads/writes under it.
#   OUTPUT_DIR    per-job scratch/artifact dir (default ./outputs/<stage>[.<variant>]/shard_<I>)
#   Sharding: --shard/--num-shards, else SHARD_IDX/TOTAL_SHARDS, else RANK/WORLD_SIZE, else the env file.
#   Both SHARD_IDX (render) and SHARD_ID (motion gen) are exported.
#
# It cds to the repo root, puts launch/ (halo_s3) on PATH, exports the env file(s), then runs STAGE_SETUP
# and STAGE_CMD with bash. STAGE_GPUS / STAGE_NODES say what to run it on.
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() { sed -n '7,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }
case "${1:-}" in ""|-h|--help) usage ;; esac
STAGE="$1"; shift
VARIANT=""
if [ $# -gt 0 ] && [[ "$1" != --* ]] && [[ "$1" != *=* ]]; then VARIANT="$1"; shift; fi
SHARD="${SHARD_IDX:-${RANK:-}}"; NSHARDS="${TOTAL_SHARDS:-${WORLD_SIZE:-}}"
DO_SETUP=1; DRY=0; OVERRIDES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --shard) SHARD="$2"; shift 2 ;;
        --num-shards) NSHARDS="$2"; shift 2 ;;
        --no-setup) DO_SETUP=0; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage ;;
        *=*) OVERRIDES+=("$1"); shift ;;
        *) echo "run_stage: unknown argument $1" >&2; usage ;;
    esac
done

ENV_FILE="$REPO/configs/pipeline/$STAGE.env"
[ -f "$ENV_FILE" ] || { echo "run_stage: no $ENV_FILE (stages: $(cd "$REPO/configs/pipeline" && ls *.env | sed 's/\.env$//' | tr '\n' ' '))" >&2; exit 2; }
: "${HALO_S3_ROOT:?set HALO_S3_ROOT (s3://bucket[/prefix] or a local dir)}"
export HALO_S3_ROOT HALO_REPO="$REPO"

set -a
[ -z "$VARIANT" ] || export VARIANT
for kv in ${OVERRIDES[@]+"${OVERRIDES[@]}"}; do export "${kv?}"; done
# shellcheck disable=SC1090
. "$ENV_FILE"
for kv in ${OVERRIDES[@]+"${OVERRIDES[@]}"}; do export "${kv?}"; done
set +a
if [ -n "$SHARD" ]; then export SHARD_IDX="$SHARD" SHARD_ID="$SHARD"; fi
if [ -n "$NSHARDS" ]; then export TOTAL_SHARDS="$NSHARDS"; fi
TAG="${STAGE_TAG:-$STAGE}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/$TAG/shard_$(printf '%03d' "${SHARD_IDX:-${SHARD_ID:-0}}")}"
export PATH="$REPO/launch:$PATH"
cd "$REPO"

echo "[run_stage] $TAG shard ${SHARD_IDX:-${SHARD_ID:-?}}/${TOTAL_SHARDS:-?}  data=$HALO_S3_ROOT  out=$OUTPUT_DIR" \
     "gpus=${STAGE_GPUS:-?} nodes=${STAGE_NODES:-1}"
if [ "$DRY" = 1 ]; then
    echo "[run_stage] setup: ${STAGE_SETUP:-<none>}"
    echo "[run_stage] cmd:   $STAGE_CMD"
    exit 0
fi
mkdir -p "$OUTPUT_DIR"
if [ "$DO_SETUP" = 1 ] && [ -n "${STAGE_SETUP:-}" ]; then
    echo "[run_stage] setup: $STAGE_SETUP"
    bash -c "$STAGE_SETUP"
fi
echo "[run_stage] cmd: $STAGE_CMD"
exec bash -c "$STAGE_CMD"
