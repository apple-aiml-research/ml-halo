//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// MujocoBackend<N> — MuJoCo physics backend, parameterised on the joint count N.
//
// Loads a MJCF from `xml_path`, sets up simulation, and serves
// read_state/send_command against mjData. Applied PD control is computed
// inside step() and written to mjData.ctrl. For parity with the Python
// SimulatorBackend ([deploy/utils/backends/mujoco_backend.py]), we assume
// position-controlled actuators and simply write torque into d->ctrl after the
// per-joint PD law.
//
// One instantiation is explicitly provided:
//   - MujocoBackend<29>  → G1 (rt_deploy::g1::kDof)
// Adding another robot is a one-line `extern template` + the matching
// instantiation in the .cpp.
//
// Ghost rendering (semi-transparent overlay of the desired motion) is gated
// behind `if constexpr (N == 29)` because `policy::MotionData::dof_poss` is
// fixed at 29 columns and the only consumer (deploy_g1_mujoco) is G1-only.
// The ghost fields in `MujocoBackendConfig` are inert for N != 29.
//
// No remote controller on this backend; poll_remote returns false. The sim
// app skips the operator-gated state-machine flow by constructing the state
// machine directly in CONTROL — see deploy_*_mujoco.cpp.
//
// Gated behind RT_DEPLOY_WITH_MUJOCO=ON.

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/robot/i_robot_backend.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rt_deploy::policy {
struct MotionData;
template <std::size_t N> struct JointRemap;
}  // namespace rt_deploy::policy

namespace rt_deploy::robot {

template <std::size_t N>
struct MujocoBackendConfig {
    using JointVec = Eigen::Matrix<float, static_cast<Eigen::Index>(N), 1>;

    std::string xml_path;
    int         substeps      = 1;       // physics steps per step() call
    bool        enable_viewer = true;    // GLFW window with free camera
    // When true, the backend spawns the viewer on an internal std::thread
    // during construction. When false, no thread is spawned — the caller is
    // responsible for invoking `backend->run_viewer_blocking()` on the main
    // thread (required on macOS, where Cocoa/AppKit must live on thread 0).
    bool        spawn_viewer_thread = true;
    int         viewer_width  = 1200;
    int         viewer_height = 900;

    // If set, qpos[7..7+N) is initialized from this on backend construction
    // (matches Python set_initial_pose). When unset, the MJCF default qpos
    // is used. Sim apps should always set this — otherwise the policy is
    // asked to track from a non-default pose at full kp on tick 0.
    std::optional<JointVec>      initial_joint_pos{};
    std::optional<Eigen::Vector3f> initial_base_pos{};        // qpos[0..2], if set
    std::optional<Eigen::Vector4f> initial_base_quat_wxyz{};  // qpos[3..6], if set
    // If set, overrides the MJCF's default `<option timestep=…>` so physics
    // advances at exactly this rate per mj_step call. Required when the
    // control loop ticks at `1/simulation_dt` wall-clock Hz and calls step()
    // once per tick — without it, d->time drifts relative to wall-clock and
    // the ghost playback (which indexes by d->time) falls behind the policy
    // (which indexes by tick count). Mirrors Python SimulatorBackend's
    // `m.opt.timestep = simulation_dt`.
    std::optional<float>         physics_timestep{};
    // Per-joint torque safety clamp applied inside step() after PD law,
    // mirroring Python SimulatorBackend.send_command's `np.clip(tau, ±max_torque)`.
    // When unset, no clamp is applied.
    std::optional<JointVec>      max_torque{};
    // Identified drive of the sim plant: the motor realises the commanded kp / kd x these per-joint scales
    // (config sim_kp_scale / sim_kd_scale, mirroring Python SimulatorBackend.send_command). Unset = 1.
    std::optional<JointVec>      kp_scale{};
    std::optional<JointVec>      kd_scale{};

    // Ghost visualization — only honored when N == 29 (G1). When
    // `ghost_motion` + `ghost_remap` are non-null the viewer indexes the motion
    // at `d->time` and draws it as a semi-transparent second robot. When
    // either is null, no ghost is drawn.
    const policy::MotionData*         ghost_motion  = nullptr;
    const policy::JointRemap<29>*     ghost_remap   = nullptr;
    std::array<float, 4>              ghost_rgba    = {0.4F, 0.8F, 1.0F, 0.45F};

    // Optional XML override — when non-empty, the backend loads this path
    // instead of `xml_path`. Used by the system/cpp sim node to swap in a
    // user-composed scene (e.g. robot + room) without editing the YAML config.
    // Empty (default) = use `xml_path`.
    std::string xml_override{};

    // Ego-camera offscreen rendering is set up after construction via
    // IRobotBackend::init_ego_renderer(width, height, camera_name) (on the main
    // thread on macOS, for GLFW's window-creation constraint); render_ego_image
    // then renders the named MJCF camera offscreen as top-down RGB, matching
    // mujoco.Renderer output.

    // Depth-camera offscreen rendering for the depth student (sim). When
    // `depth_render_enabled`, the factory INJECTS a depth camera into the model
    // via MjSpec at the training pose on `depth_cam_parent_body` (the shared
    // scene may lack one — same rationale as the Python node's runtime MjSpec
    // inject), and the app calls `init_depth_renderer(...)` after construction;
    // every obs_decim-th step() then renders it -> metric DepthFrame ->
    // depth_inbox() (drained by the control loop). The pose is in the MuJoCo
    // body frame; the defaults are the depth student's training rig (training world
    // quat already converted). Render size and clip are NOT configured here: the
    // app passes them from the observation spec to init_depth_renderer, so there
    // is only one copy of the clip.
    bool                 depth_render_enabled  = false;
    std::string          depth_camera_name     = "depth_cam";
    std::string          depth_cam_parent_body = "torso_link";
    std::array<float, 3> depth_cam_pos         = {0.054058F, 0.033836F, 0.433928F};
    std::array<float, 4> depth_cam_quat_wxyz   = {0.667544F, 0.234450F, -0.240568F, -0.664488F};
    float                depth_cam_fovy        = 59.26F;
};

// Convert a bottom-up OpenGL depth buffer (nonlinear [0,1], as mjr_readPixels
// returns it) into top-down metric eye-space depth in meters:
// z = n*f / (f - d*(f-n)). d >= 1 (no geometry), a non-finite value or a
// non-positive result maps to `far_m`. `gl` and `out` hold width*height floats;
// row r of `out` is row (height-1-r) of `gl`. Header-only and MuJoCo-free.
inline void gl_depth_to_metric(const float* gl, int width, int height,
                               double near_m, double far_m, float* out) noexcept {
    for (int r = 0; r < height; ++r) {
        const std::size_t src_row = static_cast<std::size_t>(height - 1 - r)
                                  * static_cast<std::size_t>(width);
        const std::size_t dst_row = static_cast<std::size_t>(r) * static_cast<std::size_t>(width);
        for (int c = 0; c < width; ++c) {
            const double d = static_cast<double>(gl[src_row + static_cast<std::size_t>(c)]);
            double z = (d >= 1.0) ? far_m : (near_m * far_m / (far_m - d * (far_m - near_m)));
            if (!(z > 0.0) || !std::isfinite(z)) z = far_m;
            out[dst_row + static_cast<std::size_t>(c)] = static_cast<float>(z);
        }
    }
}

// Factory: constructs a MuJoCo-backed backend for joint-count N. Returns
// error on missing / invalid XML, or if the model has fewer than N actuators.
template <std::size_t N>
[[nodiscard]] core::Result<std::unique_ptr<IRobotBackend<N>>>
make_mujoco_backend(const MujocoBackendConfig<N>& cfg);

// Explicit instantiations live in the .cpp; declare them here so consumers
// don't try to compile the bodies (which need the MuJoCo headers).
extern template core::Result<std::unique_ptr<IRobotBackend<29>>>
make_mujoco_backend(const MujocoBackendConfig<29>& cfg);

// Compiled-model camera lookup, for checking the factory's depth-camera
// injection (start-up diagnostics). Returns false when `backend` is not a
// MuJoCo backend or the model has no camera named `camera_name`. pos/quat are
// in the parent body frame, fovy in degrees.
struct MujocoCameraInfo {
    std::string           parent_body;
    std::array<double, 3> pos{};
    std::array<double, 4> quat_wxyz{};
    double                fovy{0.0};
};
template <std::size_t N>
[[nodiscard]] bool mujoco_camera_info(IRobotBackend<N>& backend, const std::string& camera_name,
                                      MujocoCameraInfo& out);
extern template bool mujoco_camera_info(IRobotBackend<29>&, const std::string&, MujocoCameraInfo&);

// MSAA sample count of the backend's offscreen (ego / depth) render buffer, or -1
// when `backend` is not a MuJoCo backend. 0 whenever depth_render_enabled: a
// multisampled depth buffer is averaged at edges on some Linux/EGL drivers.
template <std::size_t N>
[[nodiscard]] int mujoco_offscreen_samples(IRobotBackend<N>& backend);
extern template int mujoco_offscreen_samples(IRobotBackend<29>&);

}  // namespace rt_deploy::robot
