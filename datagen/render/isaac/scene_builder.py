#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Isaac Sim scene: THOR G1 (from URDF), OAK native-fisheye helmet camera, fixed TPV camera."""

from __future__ import annotations

import os

# G1 two-tone body: these links dark, the rest light (THOR adds helmet / backpack / oak1).
ROBOT_DARK_LINKS = {
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "head_link", "logo_link", "helmet_assembly_link", "thor_backpack_link", "oak1_link",
}  # fmt: skip
ROBOT_DARK_RGB = (0.04, 0.04, 0.04)
ROBOT_LIGHT_RGB = (0.85, 0.85, 0.85)
THOR_URDF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "robots", "g1", "urdf", "g1_29dof_v2_thor.urdf")
)


def _create_prim_compat(sim_utils, prim_path, prim_type, translation=None, orientation=None):
    """``sim_utils.create_prim``, or the equivalent USD calls on Isaac Lab versions without it."""
    try:
        sim_utils.create_prim(prim_path, prim_type, translation=translation, orientation=orientation)
    except AttributeError:
        import omni.usd
        from pxr import Gf, UsdGeom

        prim = omni.usd.get_context().get_stage().DefinePrim(prim_path, prim_type)
        xform = UsdGeom.Xformable(prim)
        if translation is not None:
            t = [float(x) for x in translation]
            try:
                xform.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*t))
            except Exception:
                xform.AddTranslateOp().Set(Gf.Vec3d(*t))
        if orientation is not None:
            q = [float(x) for x in orientation]
            # xformOp:orient is GfQuatf on some USD builds and GfQuatd on others.
            try:
                xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*q))
            except Exception:
                xform.AddOrientOp().Set(Gf.Quatd(*q))


def _material(stage, mat_path, color, roughness):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.02)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def set_robot_colors(robot_prim_path: str = "/World/Origin1/Robot") -> None:
    """Bind the dark/light body materials by owning link name; call after ``sim.reset()``.

    The spawned robot is an instanceable reference whose proxies ignore authored bindings, so each
    ``<link>/visuals`` Xform is un-instanced first (``collisions`` are left alone: changing them breaks the
    PhysX articulation view). Visual meshes sit at ``<parent_link>/visuals/<link>/mesh``, so a mesh takes the
    colour of its nearest ``*_link`` / ``pelvis`` ancestor inside the visuals scope.
    """
    import omni.usd
    from pxr import Usd, UsdShade

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(robot_prim_path)
    dark = _material(stage, "/World/RobotLooks/RobotDark", ROBOT_DARK_RGB, 0.7)
    light = _material(stage, "/World/RobotLooks/RobotLight", ROBOT_LIGHT_RGB, 0.35)
    scopes = []
    for prim in Usd.PrimRange(root, predicate=Usd.PrimAllPrimsPredicate):
        if prim.GetTypeName() == "Xform" and prim.GetName() == "visuals":
            scopes.append(prim)
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

    def _owning_link(mesh_prim, scope):
        p = mesh_prim.GetParent()
        while p.IsValid() and p != scope:
            if p.GetName().endswith("_link") or p.GetName() == "pelvis":
                return p.GetName()
            p = p.GetParent()
        return scope.GetParent().GetName()

    n = 0
    for scope in scopes:
        for prim in Usd.PrimRange(scope, predicate=Usd.PrimAllPrimsPredicate):
            if prim.GetTypeName() == "Mesh":
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    dark if _owning_link(prim, scope) in ROBOT_DARK_LINKS else light
                )
                n += 1
    print(f"[robot_color] bound {n} meshes in {len(scopes)} visuals scopes")


def design_scene() -> dict:
    """Spawn the dome light, THOR G1, OAK + TPV cameras (and the unused height / layout cameras)."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sensors.camera import Camera, CameraCfg
    from isaaclab.sim.spawners.sensors import FisheyeCameraCfg

    from render.cameras import OAK_CAM_ROT, OAK_OFFSET_FROM_TORSO, load_oak_intrinsics

    g1_base = None
    for name in ["G1_29DOF_CFG", "UNITREE_G1_29DOF_CFG", "G1_MINIMAL_CFG", "UNITREE_G1_MINIMAL_CFG"]:
        try:
            g1_base = getattr(__import__("isaaclab_assets", fromlist=[name]), name)
            break
        except (ImportError, AttributeError):
            continue
    if g1_base is None:
        raise ImportError("no G1 config found in isaaclab_assets")
    # ROBOT_THOR_URDF: the copy setup_render_gpu.sh stages with absolute mesh paths (the URDF importer
    # skips relative ones).
    urdf = os.environ.get("ROBOT_THOR_URDF") or THOR_URDF
    spawn = sim_utils.UrdfFileCfg(
        asset_path=urdf,
        fix_base=False,
        force_usd_conversion=True,
        merge_fixed_joints=False,  # keep helmet / backpack / oak1 links so set_robot_colors finds them
        articulation_props=g1_base.spawn.articulation_props,
        rigid_props=g1_base.spawn.rigid_props,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    )
    actuators = {k: v for k, v in g1_base.actuators.items() if k != "hands"}
    g1_cfg = g1_base.replace(spawn=spawn, actuators=actuators)
    print(f"[scene] Robot URDF (THOR variant): {urdf}")

    dome_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.97, 0.97, 1.0))
    dome_cfg.func("/World/DomeLight", dome_cfg)
    _create_prim_compat(sim_utils, "/World/Origin1", "Xform", translation=[0.0, 0.0, 0.0])
    g1 = Articulation(cfg=g1_cfg.replace(prim_path="/World/Origin1/Robot"))

    fr = load_oak_intrinsics()["fisheye_render"]
    oak_cam = Camera(
        cfg=CameraCfg(
            prim_path="/World/Origin1/Robot/base/oak_cam",
            update_period=0,
            height=int(fr["height"]),
            width=int(fr["width"]),
            data_types=["rgb"],
            spawn=FisheyeCameraCfg(
                projection_type="fisheyePolynomial",
                fisheye_nominal_width=float(fr["width"]),
                fisheye_nominal_height=float(fr["height"]),
                fisheye_optical_centre_x=float(fr["cx_px"]),
                fisheye_optical_centre_y=float(fr["cy_px"]),
                fisheye_max_fov=float(fr["max_fov_deg"]),
                fisheye_polynomial_a=float(fr["polyA"]),
                fisheye_polynomial_b=float(fr["polyB"]),
                fisheye_polynomial_c=float(fr["polyC"]),
                fisheye_polynomial_d=float(fr["polyD"]),
                fisheye_polynomial_e=float(fr["polyE"]),
                focal_length=1.6995,  # unused by the fisheye projection; K-derived metadata
                horizontal_aperture=3.84,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(pos=OAK_OFFSET_FROM_TORSO, rot=OAK_CAM_ROT, convention="ros"),
        )
    )

    def _pinhole(prim_path, h, w, data_types, focal, aperture, clip):
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=focal, focus_distance=400.0, horizontal_aperture=aperture, clipping_range=clip
        )
        return Camera(
            cfg=CameraCfg(prim_path=prim_path, update_period=0, height=h, width=w, data_types=data_types, spawn=spawn)
        )

    tpv_cam = _pinhole("/World/TPVCamera", 480, 848, ["rgb", "distance_to_image_plane"], 10.0, 20.955, (0.1, 1.0e5))
    # Never read, but kept (and referenced, so the sensor is not collected): all render products share the RTX
    # denoiser state, so removing one changes every pixel.
    height_cam = _pinhole("/World/HeightScanCamera", 64, 64, ["distance_to_image_plane"], 1.5, 4.0, (0.05, 5.0))
    layout_cam = _pinhole("/World/LayoutCamera", 512, 512, ["rgb", "distance_to_image_plane"], 1.0, 14.0, (0.01, 30.0))
    return {"g1": g1, "oak_cam": oak_cam, "tpv_cam": tpv_cam, "height_cam": height_cam, "layout_cam": layout_cam}
