# Robot assets

## Unitree G1 (29 DoF)

| File | Used by |
|---|---|
| `g1/urdf/g1_29dof_macmini_sysid.urdf` + `_isaaclab.json` + `g1/meshes/collision/{vhacd,decomp_full}` | tracker training robot and motion data conversion (system-identified lab G1 with a rear Mac mini enclosure and head cameras) |
| `g1/mjcf/g1_29dof_macmini_sysid.xml`, `g1/mjcf/scene_g1_29dof_macmini_sysid.xml` | the same lab G1 in MuJoCo (same kinematics, masses, colliders; identified armature, joint friction and torque limits; `oak1` and D435i cameras): tracker MuJoCo deployment (`tracker/deploy*/configs`); a copy with the generator's `ego_camera` is the robot of the system's bedroom scenes (`assets/scenes/bedroom`) |
| `g1/urdf/g1_29dof_rev_1_0.urdf`, `g1/mjcf/g1_29dof_rev_1_0.xml`, `g1/mjcf/scene_g1_29dof.xml` | base G1 model (viewers, generator) |
| `g1/urdf/g1_29dof_v2_thor.urdf`, `g1/mjcf/g1_29dof_v2_thor.xml` | generator rendering |

Mesh paths in the URDFs and MJCFs are relative to `g1/` (`meshes/...`).

## Viewing a model in [MuJoCo](https://github.com/google-deepmind/mujoco)

```bash
pip install mujoco
python -m mujoco.viewer
```

Drag and drop an MJCF (`g1/mjcf/*.xml`) or URDF (`g1/urdf/*.urdf`) file onto the viewer.

## Converting the URDF to MJCF

1. Uncomment the `world` link and `floating_base_joint` in the URDF, open it in the MuJoCo viewer, and save the XML.
2. In the XML, add an `<actuator>` per joint, the joint armature and damping, and the pelvis body `pos`.
3. Comment the `world` link and `floating_base_joint` in the URDF again.

`g1_29dof_macmini_sysid.xml` must stay the same robot as `g1_29dof_macmini_sysid.urdf`: per joint, its `armature`,
`frictionloss` and `actuatorfrcrange` (and the motor `ctrlrange`) are the `armature`, `friction` and `effort` of
`g1_29dof_macmini_sysid_isaaclab.json`. Keep the bedroom scenes' copy (`assets/scenes/bedroom/g1_29dof_macmini_sysid_scene.xml`) in step too.

## Converting the URDF to USD (Isaac Sim)

Training converts the URDF to USD itself when it creates the env; no USD is shipped. To inspect it in Isaac Sim, open Isaac Sim (`isaaclab.sh -s`), use *File → Import* on the URDF, set *Links* to *Moveable Base*, and use position
drives for all joints.
