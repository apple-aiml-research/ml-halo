#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Assign a render shard's episode dirs to its scenes."""

import argparse
import glob
import os


def by_manifest(spec_dir, manifest, shard, sps):
    names = [l.strip() for l in open(manifest) if l.strip()]
    scenes = [names[(shard * sps + s) % len(names)] for s in range(sps)]
    eps = sorted(glob.glob(os.path.join(spec_dir, "episode_*")))
    return [
        (scenes[k % len(scenes)], ep)
        for k, ep in enumerate(eps)
        if os.path.isfile(os.path.join(ep, "episode_spec.json"))
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec_dir", help="shard spec dir with episode_*/ subdirs")
    ap.add_argument("out", help="output file, one <scene><TAB><episode_dir> line per episode")
    ap.add_argument("--manifest", required=True, help="scene list file (one name per line)")
    ap.add_argument("--shard", type=int, default=0, help="render shard index")
    ap.add_argument("--sps", type=int, default=1, help="scenes per shard")
    a = ap.parse_args()
    rows = by_manifest(a.spec_dir, a.manifest, a.shard, a.sps)
    with open(a.out, "w") as f:
        for scene, ep in rows:
            f.write(f"{scene}\t{ep}\n")


if __name__ == "__main__":
    main()
