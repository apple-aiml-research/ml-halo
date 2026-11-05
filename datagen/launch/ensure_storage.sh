#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PATH"
: "${HALO_S3_ROOT:?set HALO_S3_ROOT (s3://bucket[/prefix] or a local dir)}"
case "$HALO_S3_ROOT" in
    s3://*)
        if ! command -v aws >/dev/null 2>&1; then
            echo "[storage] installing AWS CLI v2"
            curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o /tmp/awscliv2.zip \
                && unzip -q -o /tmp/awscliv2.zip -d /tmp && /tmp/aws/install --update \
                && rm -rf /tmp/awscliv2.zip /tmp/aws
        fi
        if timeout 120 halo_s3 ls "$HALO_S3_ROOT/" >/dev/null 2>&1; then
            echo "[storage] $HALO_S3_ROOT reachable"
        else
            echo "[storage] WARNING: cannot list $HALO_S3_ROOT/ (credentials? endpoint?)"
        fi ;;
    *)
        mkdir -p "${HALO_S3_ROOT#file://}" && echo "[storage] local data root $HALO_S3_ROOT" ;;
esac
