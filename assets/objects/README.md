# Scene objects

Assets the tracker's scene tasks spawn (`tracker/training/.../g1/motion_tracking_scene_env_cfg.py`). Each clip is
bound to its asset by file name (`tracker/training/.../g1/mdp/commands.py`).

| Folder | Clips | Role |
|---|---|---|
| `pushbox/` | `push_v2_*` | free rigid box (0.46 x 0.50 x 0.56 m, 1.0 kg) the robot pushes; Push expert and shared-scene (`SuperScene`) object |
| `labsofa/` | `labsofa_*` | lab sofa, a fixed terrain asset (mesh visual, box colliders); Sit expert and shared-scene (`SuperScene`) terrain |

The shared scene's flat-ground plates are generated in code. See `tracker/training/README.md` ("Scene assets").

## `chairs/` (data pipeline)

The sit-target chair of the staged data pipeline (`datagen/docs/data_pipeline.md`). The paper data drew one of
two chairs per episode; only the arm chair ships, so both slots of `CHAIR_SET=pair` (the default, which keeps the
paper's per-episode draws) and `single` render it. Motion generation writes its path into `episode_spec.json`; both renderers look the chair up here by file name first, then under
`LAB_SCENE_DIR` (default `~/datasets/lab_scene_data`), then download it from
`$HALO_S3_ROOT/lab_scene_data/` (`datagen/render/assets.py`).

| File | Size | md5 | Used by | Origin | License |
|---|---|---|---|---|---|
| `chairs/arm_chair.usdz` | 8,494,775 B | `22edb5746388b13f7de86e6fafad4738` | gen (spec `asset_path`), Isaac render (prop USD), MuJoCo render (USDZ → textured OBJ, needs `pxr`) | the `lab_scene_data/` set the reference renders used (byte-identical) | redistribution approved by the project owner for this release; upstream license not recorded |

The sampler places the chair by its seat geometry (seat surface 0.45 m), not by its mesh.
Collision meshes (convex parts) for closed-loop physics are not part of the data pipeline and are not
included.
