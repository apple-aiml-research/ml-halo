//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// deploy_g1_mujoco — run the G1 motion tracker against the MuJoCo sim.
//
// Requires cmake options: RT_DEPLOY_WITH_MUJOCO=ON
//                         plus at least one of RT_DEPLOY_WITH_TRT / RT_DEPLOY_WITH_ORT.

#include "apps/g1/common.hpp"

#include "rt_deploy/robot/mujoco_backend.hpp"

#if defined(RT_DEPLOY_WITH_ORT)
#    include "rt_deploy/inference/ort_engine.hpp"
#endif
#if defined(RT_DEPLOY_WITH_TRT)
#    include "rt_deploy/inference/trt_engine.hpp"
#endif

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace rt_deploy;

namespace {

std::unique_ptr<inference::IInferenceEngine>
make_engine(g1::PreferredEngine choice, const std::string& policy_path) {
#if defined(RT_DEPLOY_WITH_TRT)
    if (choice == g1::PreferredEngine::kTrt
        || (choice == g1::PreferredEngine::kAuto && policy_path.ends_with(".plan"))) {
        inference::TrtEngineConfig cfg;
        cfg.plan_path = policy_path;
        auto r        = inference::make_trt_engine(cfg);
        if (!r) {
            std::fprintf(stderr, "[error] TRT engine init failed: %s\n", r.error().c_str());
            return nullptr;
        }
        return std::move(r.value());
    }
#endif
#if defined(RT_DEPLOY_WITH_ORT)
    (void)choice;
    inference::OrtEngineConfig cfg;
    cfg.onnx_path = policy_path;
    auto r         = inference::make_ort_engine(cfg);
    if (!r) {
        std::fprintf(stderr, "[error] ORT engine init failed: %s\n", r.error().c_str());
        return nullptr;
    }
    return std::move(r.value());
#endif
    std::fprintf(stderr,
                 "[error] no inference backend compiled in. Rebuild with -DRT_DEPLOY_WITH_TRT=ON "
                 "or -DRT_DEPLOY_WITH_ORT=ON.\n");
    (void)policy_path;
    (void)choice;
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    g1::CommonArgs                args;
    std::vector<std::string>      rest;
    if (!g1::parse_common_args(argc, argv, "deploy_g1_mujoco",
                                "[--no_viewer] [--no_align_motion] [--depth_render]", args, rest)) {
        return 1;
    }

    bool enable_viewer = true;
    // SE(2)+Z-anchor the loaded --motion onto the robot's spawn pose (default
    // on); --no_align_motion plays the CSV verbatim, in its own world frame
    // (no XY, yaw or Z rebase, here or in the control loop).
    bool align_motion  = true;
    // Depth student: render the injected torso depth camera and feed it to the
    // policy. Without it a config with an image obs group is refused at start
    // (no depth inbox -> kConfigError, exit 4).
    bool depth_render  = false;
    for (std::size_t i = 0; i < rest.size(); ++i) {
        if (rest[i] == "--no_viewer") {
            enable_viewer = false;
        } else if (rest[i] == "--no_align_motion") {
            align_motion      = false;
            args.align_motion = false;
        } else if (rest[i] == "--depth_render") {
            depth_render = true;
        } else {
            std::fprintf(stderr, "error: unknown argument %s\n", rest[i].c_str());
            return 1;
        }
    }

    // No motion data ships (see tracker/data/README.md). Bare names are
    // resolved to <nearest ancestor data/>/<name>.csv.
    if (args.motion_path.empty()) {
        std::fprintf(stderr, "[error] --motion is required (a CSV motion file, see tracker/data/README.md)\n");
        return 1;
    }
    args.motion_path = g1::resolve_motion_path(args.motion_path, args.config_path);

    auto cfg_r = g1::load_config(args.config_path);
    if (!cfg_r) {
        std::fprintf(stderr, "[error] config: %s\n", cfg_r.error().c_str());
        return 1;
    }
    const auto cfg   = std::move(cfg_r.value());
    const auto remap = policy::build_joint_remap<g1::kDof>(cfg.deployment_joint_order,
                                                            cfg.training_joint_order);

    policy::MotionLoaderConfig mcfg;
    mcfg.csv_path          = args.motion_path;
    mcfg.input_fps         = args.motion_fps;
    mcfg.output_fps        = static_cast<int>(std::round(1.0F / cfg.control_dt));
    mcfg.joint_remap       = &remap;
    mcfg.standing_joint_pos = cfg.standing_joint_pos;
    mcfg.standing_base_pos  = cfg.standing_base_pos;
    mcfg.standing_base_quat_wxyz = cfg.standing_base_quat_wxyz;
    mcfg.min_output_frames = g1::motion_pad_frames(args.duration_s, mcfg.output_fps);
    auto motion_r = policy::load_motion(mcfg);
    if (!motion_r) {
        std::fprintf(stderr, "[error] motion: %s\n", motion_r.error().c_str());
        return 1;
    }
    auto motion = std::move(motion_r.value());

    // Build the standing-pose motion once at startup and wrap both motions
    // in a MotionSwitcher, the same wiring as the real-robot app. The sim app
    // starts directly in CONTROL and the MuJoCo backend has no remote, so
    // neither the CONTROL-entry alignment nor the B/A swaps ever run here; the
    // motion is anchored once, below, before the loop starts.
    auto standing_r = g1::make_standing_motion(cfg, remap, mcfg.min_output_frames);
    if (!standing_r) {
        std::fprintf(stderr, "[error] standing motion: %s\n", standing_r.error().c_str());
        return 1;
    }
    g1::MotionSwitcher switcher{&motion, std::move(standing_r.value())};

    robot::MujocoBackendConfig<g1::kDof> bk_cfg;
    bk_cfg.xml_path     = cfg.xml_path;
    // Sub-step physics `control_decimation` times per control tick: the loop
    // ticks (and infers) once per control step while MuJoCo keeps advancing at
    // the fine simulation_dt. Mirrors Python deploy_mujoco's per-step physics
    // loop and keeps physics fidelity independent of the control/inference rate.
    bk_cfg.substeps     = std::max(1, cfg.decimation);
    bk_cfg.enable_viewer = enable_viewer;
    // Match Python deploy_mujoco.py — the robot spawns at cfg.default_angles
    // (deployment order), not at the motion's first frame.
    bk_cfg.initial_joint_pos = cfg.default_angles;
    bk_cfg.max_torque        = cfg.max_torque;
    bk_cfg.kp_scale          = cfg.sim_kp_scale;  // the sim plant's identified drive (sim only)
    bk_cfg.kd_scale          = cfg.sim_kd_scale;
    // Force the physics timestep to simulation_dt so each mj_step advances by
    // that amount. With substeps = control_decimation, one control tick advances
    // d->time by control_dt, so d->time stays in lockstep with the control rate
    // and the ghost playback (indexed by d->time) tracks the policy exactly.
    bk_cfg.physics_timestep  = cfg.simulation_dt;
    // Ghost overlay: replay the desired motion as a semi-transparent light-blue
    // copy of the robot in the same viewer. Kinematics-only, no physics or
    // collisions — helpful for visually comparing the policy's tracking error.
    bk_cfg.ghost_motion = &motion;
    bk_cfg.ghost_remap  = &remap;
    // Depth student: inject the torso depth camera at load (MjSpec) so the
    // depth renderer can find it after construction.
    bk_cfg.depth_render_enabled = depth_render;
    // macOS: AppKit / NSApplication must initialize on the main thread, so the
    // viewer can't run on an internally-spawned std::thread. Force the backend
    // to defer viewer setup; we'll run the control loop on a worker and the
    // viewer on the main thread below.
#ifdef __APPLE__
    bk_cfg.spawn_viewer_thread = false;
#endif
    auto bk_r       = robot::make_mujoco_backend<g1::kDof>(bk_cfg);
    if (!bk_r) {
        std::fprintf(stderr, "[error] MuJoCo backend: %s\n", bk_r.error().c_str());
        return 1;
    }
    auto backend = std::move(bk_r.value());

    // Depth render (main thread — macOS GL). The depth render reuses
    // the offscreen context that init_ego_renderer creates; deploy_mujoco has no
    // ego RGB use, so bootstrap that context with the injected depth camera
    // itself (the ego RGB output is never read here). The backend then renders
    // every obs_decim-th step() → depth_inbox → the control loop feeds it to the
    // policy.
    bool depth_active = false;
    if (depth_render) {
        // Geometry AND clip come from the observation spec, never from literals
        // here: the sim depth source must clamp to exactly what the obs group
        // encodes, and a second hardcoded copy of that number can silently
        // drift from it. See GroupCfg::clip_min.
        auto spec_r = g1::load_observation_spec(args.config_path);
        if (!spec_r) {
            std::fprintf(stderr, "[error] observation spec: %s\n", spec_r.error().c_str());
            return 1;
        }
        const g1::GroupCfg* img = nullptr;
        for (const auto& g : spec_r.value().groups) {
            if (g.is_image) {
                img = &g;
                break;
            }
        }
        if (img == nullptr) {
            std::fprintf(stderr,
                         "[error] --depth_render given but %s declares no image observation group, "
                         "so nothing would consume the rendered frames.\n",
                         args.config_path.c_str());
            return 1;
        }
        // Bootstrap the offscreen context at 848x480 (the ego-render default of
        // the system/cpp sim node) — on macOS the render targets the GLFW window framebuffer, and
        // a tiny (84x48) window vs the scene's large offscreen size overruns it and
        // segfaults. The depth render draws an 84x48 viewport inside this buffer.
        // Use the name the factory injected (bk_cfg.depth_camera_name), not a
        // second copy of the literal that could drift from it.
        (void)backend->init_ego_renderer(848, 480, bk_cfg.depth_camera_name);
        depth_active = backend->init_depth_renderer(bk_cfg.depth_camera_name, img->img_cols,
                                                    img->img_rows, img->clip_min, img->clip_max,
                                                    img->obs_decim);
        if (!depth_active) {
            std::fprintf(stderr, "[deploy_g1_mujoco] --depth_render init failed; no depth inbox, "
                                 "so the control loop will refuse this image-group config.\n");
        } else {
            // Print the RESOLVED depth rate. The hold changes no shape and no
            // width, so a wrong depth rate is invisible everywhere else — and an
            // omitted obs_decim silently doubles it relative to training.
            const double control_hz = 1.0 / static_cast<double>(cfg.control_dt);
            std::fprintf(stderr,
                         "[deploy_g1_mujoco] depth render %dx%d  clip=[%.2f,%.2f] m  "
                         "obs_decim=%d -> depth fed at %.1f Hz (control %.1f Hz)\n",
                         img->img_cols, img->img_rows, static_cast<double>(img->clip_min),
                         static_cast<double>(img->clip_max), img->obs_decim,
                         control_hz / static_cast<double>(img->obs_decim), control_hz);
            if (img->obs_decim == 1) {
                std::fprintf(stderr,
                             "[deploy_g1_mujoco] NOTE: obs_decim=1 feeds a fresh frame every "
                             "control step. The depth student trains with obs_decim=2 "
                             "(25 Hz), so this is a train/deploy mismatch on "
                             "the only exteroceptive input. Set obs_decim: 2 in the image group.\n");
            }
        }
    }
    core::TripleBuffer<g1::DepthFrame>* const depth_inbox
        = depth_active ? backend->depth_inbox() : nullptr;

    // SE(2)+Z-anchor the loaded motion onto the robot's spawn pose — the same
    // rebase the control loop applies when it installs a cold-start plan
    // (g1::send_ref_motion → policy::align_motion_to_anchor, robot-anchor
    // branch). Without it a CSV authored in a non-canonical world frame (e.g. a
    // closed-loop recording that starts at world (-5.71, 3.94) facing 152°) is
    // tracked verbatim while the robot spawns at the config pose, so heading +
    // XY + Z are all off. Anchoring frame 0 to where the robot actually spawned
    // makes the reference (and the ghost) start exactly on the robot — zero
    // initial tracking error. Canonical-frame training clips already start at
    // the spawn pose, so for those this is a no-op. --no_align_motion opts out.
    if (align_motion) {
        math::RobotState<g1::kDof> spawn{};
        (void)backend->read_state(spawn);
        const Eigen::Vector3f anchor_pos = spawn.odom_valid
            ? spawn.odom_position : cfg.standing_base_pos;
        const Eigen::Vector4f anchor_quat = spawn.odom_valid
            ? spawn.odom_quaternion_wxyz : cfg.standing_base_quat_wxyz;
        policy::align_motion_to_anchor(motion, anchor_pos, anchor_quat);
        std::fprintf(stderr,
            "[deploy_g1_mujoco] SE(2)+Z anchored motion frame0 to spawn pose: "
            "xy=(%+.3f, %+.3f) z=%.3f\n",
            static_cast<double>(anchor_pos[0]),
            static_cast<double>(anchor_pos[1]),
            static_cast<double>(anchor_pos[2]));
    }

    auto engine = make_engine(g1::parse_engine_choice(args.engine), args.policy_path);
    if (!engine) {
        return 1;
    }

    // Sim path: the loop ticks (and infers) at the control rate (1/control_dt);
    // the backend sub-steps physics cfg.decimation times per tick (set above)
    // so MuJoCo still advances at simulation_dt. Override any user --control_hz
    // since the sim rate is fixed by the trained policy / config.
    args.control_hz = std::max(1, static_cast<int>(std::round(1.0F / cfg.control_dt)));

#ifdef __APPLE__
    // macOS path: control loop on a worker, viewer on main (Cocoa constraint).
    // The main thread retains ownership of both the backend and the engine so
    // the viewer can stay alive after the control loop exits cleanly. Without a
    // viewer the loop simply runs on the main thread (below).
    if (enable_viewer) {
    int         exit_code = 0;
    std::thread control_thread{[&]() {
        exit_code = g1::run_control_loop(args, cfg, remap, &motion,
                                          *backend, *engine,
                                          math::ControlState::kControl,
                                          &switcher,
                                          /*state_outbox=*/nullptr, /*motion_inbox=*/nullptr,
                                          /*motion_scratch_capacity=*/0,
                                          /*inference_tick_out=*/nullptr,
                                          /*active_plan_snapshot_out=*/nullptr,
                                          /*trim_install_pending=*/nullptr,
                                          /*trim_install_tick=*/nullptr,
                                          /*tracker_ready_out=*/nullptr,
                                          /*control_loop_out=*/nullptr,
                                          /*on_loop_exit=*/{},
                                          depth_inbox);
        // When the control loop exits (signal, max_ticks, fault), tell the
        // viewer to close so the main thread unblocks and we exit cleanly.
        backend->request_viewer_stop();
    }};
    backend->run_viewer_blocking();  // returns when window closed
    // User-closed viewer → tell the control loop to wind down.
    g1::signal_requested().store(true, std::memory_order_release);
    control_thread.join();
    return exit_code;
    }
#endif
    return g1::run_control_loop(args, cfg, remap, &motion,
                                 *backend, *engine,
                                 math::ControlState::kControl,
                                 &switcher,
                                 /*state_outbox=*/nullptr, /*motion_inbox=*/nullptr,
                                 /*motion_scratch_capacity=*/0,
                                 /*inference_tick_out=*/nullptr,
                                 /*active_plan_snapshot_out=*/nullptr,
                                 /*trim_install_pending=*/nullptr,
                                 /*trim_install_tick=*/nullptr,
                                 /*tracker_ready_out=*/nullptr,
                                 /*control_loop_out=*/nullptr,
                                 /*on_loop_exit=*/{},
                                 depth_inbox);
}
