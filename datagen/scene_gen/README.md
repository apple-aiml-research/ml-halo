# scene_gen: Infinigen indoor scenes, baked for MuJoCo and Isaac

Generates indoor scenes with [Infinigen](https://github.com/princeton-vl/infinigen) and bakes the lighting of
each into an unlit textured mesh (`asset/scan.obj` + `scan.usdc`) that the pipeline uses as rooms, then analyses
it (`scene_analysis.json`, read by the render's placement and TPV choice); step 1 of
[data_pipeline.md](../docs/data_pipeline.md).

| set | scene list | output | command |
|---|---|---|---|
| single-room | `spec/scenes_single.json` | `scenes/single/` | `scene_gen single` (300 shards × 4 scenes) |
| dual-room | `spec/scenes_dual.json` | `scenes/dual/` | `scene_gen dual` (500 shards × 2, ~2× slower) |

```bash
launch/run_stage.sh scene_gen single --shard 0 --num-shards 300 LIGHT_GAIN=2.0 [OUT_S3=<new prefix>]
launch/run_stage.sh scene_gen dual   --shard 0 --num-shards 500
launch/halo_s3 ls $HALO_S3_ROOT/scenes/single/_done/ | wc -l    # finished scenes
```

Setup: `env/setup_scene_gen.sh` (the step's `STAGE_SETUP`) installs Blender 5.2.0 and Infinigen `892947be` with its
Python 3.11 env. Every paper scene used `LIGHT_GAIN=2.0` (unset: the scene list's per-scene gain, 0.9-2.2; the dual
list has none, so 2.0). With your own `OUT_S3` (under `HALO_S3_ROOT`), give the render the same prefix as
`SCENE_S3_PATH=<OUT_S3 without the HALO_S3_ROOT/ part>` (default `scenes/single`). A scene is finished only when
`$OUT_S3/_done/<scene>.json` exists (written last; failures in `_failed/`); reruns skip marked scenes. Some scenes
fail in Infinigen; step 3 renders the 1,000 listed in `common/manifests/render_scenes_1000.txt`.

Per shard (`pod/run_shard.sh`): `gen_worker.py` → `gen_scene.py` → Infinigen via `scripts/infinigen_style.py` →
`prep_and_gt.py` → `bake.py` (`BAKE_SPP=128`, `BAKE_MAXRES=4096`) → `project_gt.py` → `obj_to_usdc.py` →
`python -m render.synthetic_scene.analyze_scene` → upload → `_done` marker. Per scene: `asset/
{scan.obj,scan.mtl,tex/*.png,scan.usdc}`, `bake/bake_manifest.json`, `anchor.json`, `views_64.json`,
`scene_analysis.json` + `scene_analysis_topdown.png` (a scene that cannot be analysed ships without them, with a
`WARN` in the log), `logs/` (`gen.log`: first line = the Infinigen command; `bake.log`, `analyze.log`, ...). A
scene's Infinigen run is killed after `GEN_TIMEOUT_S` (5 h) and retried on the shard's next run.

## Scene lists

The scene list is the full definition of a set (per-scene seed, room type, size, floor plan, style, shard).
Rebuild the single-room list exactly with:

```bash
cd scene_gen/spec
python make_manifest.py --n 1200 --shards 300 --views 0 --gt-frames 0 --out scenes_single.json
```

`scenes_dual.json` was hand-edited after `make_dualroom_manifest.py` and stays frozen. Do not rebuild a list while
its sweep runs (shard membership would change). Per scene (`scene_spec()` in `spec/make_manifest.py`): room type
(LivingRoom 45 % / Office 30 % / Bedroom 25 %), size, furniture density, sofa, floor family, rugs, windows, plain vs
decorated walls (half flat-colour). Dual-room: 5 two-room layouts joined by a door-less opening.

## Pitfalls

- `bake.py` and `project_gt.py` read the camera and keep-region from `$ANCHOR_JSON` (written by `prep_and_gt.py`); there is no default.
- Un-hiding the ceiling seals the room and bakes it dark; the ceiling is never baked, so views pitch down.
- Window panes jump to the origin after deleting an un-baked parent; `bake.py` restores them.
- `pod/run_shard.sh` uses Blender's bundled python for its JSON work and the scene analysis (the setup installs
  numpy, scipy, matplotlib, imageio, opencv and usd-core into it).
- ASCII `.usda` makes Isaac exit with a black frame; always binary `.usdc`.
- A shard can finish with scenes missing (Infinigen failures, or `BUDGET_S`: no new scene starts after it);
  count `_done` markers and rerun the shard.
- Some Infinigen failures are intermittent (`_failed/<scene>/gen.log`) and succeed on the shard's rerun; a scene can
  also stall in generation (no output for hours, usually ~30 min) until `GEN_TIMEOUT_S`. To render a scene list on
  time, put built spare scenes in place of missing ones in your own `SCENE_MANIFEST` (with a new
  `SHARED_PREFIX_ROOT`) before those render shards start.
- Shards are balanced by predicted cost, so the scenes of `render_scenes_1000.txt` rows `0..k` are spread over
  many shards. To build only the scenes a render subset needs, pass a copy of the list whose `shard_members` holds
  one scene per shard: `MANIFEST=<that copy> --shard i --num-shards <its length>`.
