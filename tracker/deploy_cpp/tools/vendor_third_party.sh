#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Vendor the FetchContent dependencies declared in CMakeLists.txt into
# deploy_cpp/third_party/ so the project can be configured on an air-gapped
# host (e.g. an onboard robot computer without general internet access).
#
# Workflow:
#   1) Run this script once on a dev machine that has internet.
#   2) rsync the entire deploy_cpp/ tree (including third_party/) to the
#      target.
#   3) On the target, configure with `cmake --preset release-offline`.
#
# The tags below MUST stay in sync with the FetchContent_Declare(GIT_TAG ...)
# blocks in CMakeLists.txt — if you bump a version there, bump it here too.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/.." && pwd)"
VENDOR_DIR="${ROOT}/third_party"

# name | git URL | tag
DEPS=(
    "eigen|https://gitlab.com/libeigen/eigen.git|5.0.0"
    "yaml-cpp|https://github.com/jbeder/yaml-cpp.git|0.8.0"
)

mkdir -p "${VENDOR_DIR}"

for entry in "${DEPS[@]}"; do
    IFS='|' read -r name url tag <<< "${entry}"
    dest="${VENDOR_DIR}/${name}"

    if [[ -d "${dest}/.git" ]]; then
        current_tag="$(git -C "${dest}" describe --tags --exact-match 2>/dev/null || true)"
        if [[ "${current_tag}" == "${tag}" ]]; then
            echo "[vendor] ${name}@${tag} already present — skip"
            continue
        fi
        echo "[vendor] ${name} is at '${current_tag:-unknown}', want '${tag}' — refetching"
        rm -rf "${dest}"
    elif [[ -e "${dest}" ]]; then
        echo "[vendor] ${dest} exists but is not a git checkout — refusing to overwrite" >&2
        exit 1
    fi

    echo "[vendor] cloning ${name}@${tag} from ${url}"
    git clone --depth 1 --branch "${tag}" "${url}" "${dest}"
done

echo
echo "[vendor] done. Sources are under ${VENDOR_DIR}"
echo "[vendor] Now rsync deploy_cpp/ to the offline target and run:"
echo "  cmake --preset release-offline -DRT_DEPLOY_WITH_TRT=ON -DRT_DEPLOY_WITH_UNITREE=ON"
echo "  cmake --build build/release-offline --target deploy_g1_real -j"
