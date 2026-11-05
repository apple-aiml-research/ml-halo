//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// motiontrack_mujoco_zmq — halo_system tracker NODE (MuJoCo sim).
//
// tracker's G1 control loop against MuJoCo + a halo_comm TrackerServer that
// receives motion plans from an external generator and answers state / trim
// over protobuf (services g1/get_state, g1/put_plan, g1/get_trim + topic
// g1/state). The transport lives in halo_comm (halo_comm::Node); the tracker
// is pure control, linked via add_subdirectory.
//
// Requires CMake: RT_DEPLOY_WITH_MUJOCO=ON + RT_DEPLOY_WITH_ORT=ON.

#include "apps/g1/common.hpp"   // tracker app glue (CLI, run_control_loop)
#include "transport/tracker_server.hpp"

#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/g1/observation_builder.hpp"
#include "rt_deploy/robot/mujoco_backend.hpp"

#include "rt_deploy/inference/ort_engine.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace rt_deploy;

namespace {

struct ZmqExtraArgs {
    std::string xml_override;
    halo_system::TrackerServerConfig srv;
    // Initial robot pose overrides (--init_pos/--init_yaw): spawn the robot at a
    // given pose instead of the YAML default pose.
    bool        have_init_pos = false;
    float       init_pos[3]   = {0.0F, 0.0F, 0.0F};
    bool        have_init_yaw = false;
    float       init_yaw_deg  = 0.0F;
    // Ego-camera offscreen render → get_state Image trailer (image-conditioned
    // policies).
    bool        ego_render        = false;
    int         ego_render_width  = 848;
    int         ego_render_height = 480;
    std::string ego_camera_name   = "ego_camera";
    // Depth student: inject the torso depth camera, render it every obs_decim-th
    // step and feed it to the policy (config must declare an image group).
    bool        depth_render      = false;
};

constexpr int kDepthPubPort = 5553;   // g1/depth for the web viewer

bool parse_extras(const std::vector<std::string>& rest, ZmqExtraArgs& out) {
    auto take = [&](std::size_t& i, const char* flag) -> const std::string* {
        if (i + 1 >= rest.size()) {
            std::fprintf(stderr, "error: missing value for %s\n", flag);
            return nullptr;
        }
        return &rest[++i];
    };
    for (std::size_t i = 0; i < rest.size(); ++i) {
        const std::string& a = rest[i];
        if (a == "--scene_xml") {
            const auto* v = take(i, "--scene_xml"); if (v == nullptr) return false;
            out.xml_override = *v;
        } else if (a == "--bind_ip") {
            const auto* v = take(i, "--bind_ip"); if (v == nullptr) return false;
            out.srv.bind_ip = *v;
        } else if (a == "--state_port") {
            const auto* v = take(i, "--state_port"); if (v == nullptr) return false;
            out.srv.state_port = std::atoi(v->c_str());
        } else if (a == "--plan_port") {
            const auto* v = take(i, "--plan_port"); if (v == nullptr) return false;
            out.srv.plan_port = std::atoi(v->c_str());
        } else if (a == "--trim_port") {
            const auto* v = take(i, "--trim_port"); if (v == nullptr) return false;
            out.srv.trim_port = std::atoi(v->c_str());
        } else if (a == "--state_pub_port") {
            const auto* v = take(i, "--state_pub_port"); if (v == nullptr) return false;
            out.srv.state_pub_port = std::atoi(v->c_str());
        } else if (a == "--init_pos") {
            if (i + 3 >= rest.size()) {
                std::fprintf(stderr, "error: --init_pos needs 3 values: x y z\n");
                return false;
            }
            out.init_pos[0]   = static_cast<float>(std::atof(rest[++i].c_str()));
            out.init_pos[1]   = static_cast<float>(std::atof(rest[++i].c_str()));
            out.init_pos[2]   = static_cast<float>(std::atof(rest[++i].c_str()));
            out.have_init_pos = true;
        } else if (a == "--init_yaw") {
            const auto* v = take(i, "--init_yaw"); if (v == nullptr) return false;
            out.init_yaw_deg  = static_cast<float>(std::atof(v->c_str()));
            out.have_init_yaw = true;
        } else if (a == "--ego_render") {
            out.ego_render = true;
        } else if (a == "--depth_render") {
            out.depth_render = true;
        } else if (a == "--ego_render_width") {
            const auto* v = take(i, "--ego_render_width"); if (v == nullptr) return false;
            out.ego_render_width = std::atoi(v->c_str());
        } else if (a == "--ego_render_height") {
            const auto* v = take(i, "--ego_render_height"); if (v == nullptr) return false;
            out.ego_render_height = std::atoi(v->c_str());
        } else if (a == "--ego_camera_name") {
            const auto* v = take(i, "--ego_camera_name"); if (v == nullptr) return false;
            out.ego_camera_name = *v;
        } else {
            std::fprintf(stderr, "error: unknown argument %s\n", a.c_str());
            return false;
        }
    }
    return true;
}

std::unique_ptr<inference::IInferenceEngine> make_engine(const std::string& policy_path) {
    inference::OrtEngineConfig cfg;
    cfg.onnx_path = policy_path;
    auto r        = inference::make_ort_engine(cfg);
    if (!r) { std::fprintf(stderr, "[error] ORT init: %s\n", r.error().c_str()); return nullptr; }
    return std::move(r.value());
}

}  // namespace

int main(int argc, char** argv) {
    g1::CommonArgs           args;
    std::vector<std::string> rest;
    if (!g1::parse_common_args(argc, argv, "motiontrack_mujoco_zmq",
                                "[--scene_xml PATH] [--bind_ip IP] "
                                "[--state_port N] [--plan_port N] [--trim_port N] "
                                "[--state_pub_port N] "
                                "[--init_pos X Y Z] [--init_yaw DEG] [--ego_render] "
                                "[--ego_render_width N] [--ego_render_height N] "
                                "[--ego_camera_name NAME] [--depth_render]",
                                args, rest)) {
        return 1;
    }
    ZmqExtraArgs extras;
    if (!parse_extras(rest, extras)) return 1;

    // Nothing else sets g1::signal_requested() mid-run, so the control loop
    // would otherwise run its full --duration (standing long after the
    // generator's plans end). Install SIGINT/SIGTERM handlers so the launcher's
    // teardown SIGINT stops the loop promptly on its next tick.
    g1::install_signal_handlers();

    auto cfg_r = g1::load_config(args.config_path);
    if (!cfg_r) { std::fprintf(stderr, "[error] config: %s\n", cfg_r.error().c_str()); return 1; }
    const auto cfg   = std::move(cfg_r.value());
    const auto remap = policy::build_joint_remap<g1::kDof>(cfg.deployment_joint_order,
                                                           cfg.training_joint_order);

    const int output_fps = static_cast<int>(std::round(1.0F / cfg.control_dt));
    // The standing pose only bridges boot → first generator plan, and the
    // control loop holds the last frame past frame_count, so size its buffer by
    // a bounded horizon — NOT the full --duration (now 60000 s by default,
    // which would materialize ~3M identical held-pose frames, hundreds of MB).
    const float        standing_horizon_s = std::min(args.duration_s, 600.0F);
    const std::int32_t min_output_frames = static_cast<std::int32_t>(
        standing_horizon_s * static_cast<float>(output_fps)
        + 5.0F * static_cast<float>(output_fps));

    auto standing_r = g1::make_standing_motion(cfg, remap, min_output_frames);
    if (!standing_r) { std::fprintf(stderr, "[error] standing: %s\n", standing_r.error().c_str()); return 1; }
    // standing-only switcher (no CSV original): A-replay is then a no-op.
    g1::MotionSwitcher switcher{/*original=*/nullptr, std::move(standing_r.value())};

    // One-shot yaw-align of the standing motion to --init_yaw BEFORE the loop.
    // Without it the standing reference has
    // yaw=0 while the robot spawns at --init_yaw, so the policy would command
    // joints to rotate the robot back to identity and the visible yaw collapses.
    if (extras.have_init_yaw) {
        const float yaw_rad = extras.init_yaw_deg * static_cast<float>(M_PI) / 180.0F;
        const Eigen::Vector4f init_quat_wxyz{
            std::cos(yaw_rad * 0.5F), 0.0F, 0.0F, std::sin(yaw_rad * 0.5F)};
        (void)switcher.switch_to_standing(init_quat_wxyz);
        std::fprintf(stderr,
            "[motiontrack_mujoco_zmq] standing motion yaw-aligned to init_yaw=%.2f deg\n",
            static_cast<double>(extras.init_yaw_deg));
    }

    robot::MujocoBackendConfig<g1::kDof> bk_cfg;
    bk_cfg.xml_path         = cfg.xml_path;
    bk_cfg.xml_override     = extras.xml_override;   // empty → use xml_path
    // Substep model: mirrors the standalone deploy_g1_mujoco.
    // The loop ticks at the inference rate (50 Hz, 1/control_dt) and the
    // backend does cfg.decimation (4) physics substeps internally per mj_step
    // call, so MuJoCo's solver sees an uninterrupted 4-substep chain rather
    // than 4 separate mj_steps that the loop reads mjData between. Avoids the
    // ZMQ-side stale-state-outbox interaction we saw with a 200 Hz loop.
    bk_cfg.substeps         = std::max(1, cfg.decimation);
    bk_cfg.enable_viewer    = false;   // no desktop window; the web viewer shows the run
    bk_cfg.initial_joint_pos = cfg.default_angles;
    bk_cfg.max_torque       = cfg.max_torque;
    bk_cfg.kp_scale         = cfg.sim_kp_scale;  // the sim plant's identified drive (sim only)
    bk_cfg.kd_scale         = cfg.sim_kd_scale;
    // Force physics to advance by simulation_dt per substep, so substeps × dt
    // = control_dt (i.e. one mj_step call covers exactly one inference period).
    bk_cfg.physics_timestep = cfg.simulation_dt;
    // Initial base pose overrides (see ZmqExtraArgs): spawn the robot at
    // --init_pos/--init_yaw instead of the YAML default.
    if (extras.have_init_pos) {
        bk_cfg.initial_base_pos = Eigen::Vector3f{
            extras.init_pos[0], extras.init_pos[1], extras.init_pos[2]};
    }
    if (extras.have_init_yaw) {
        const float yaw_rad = extras.init_yaw_deg * static_cast<float>(M_PI) / 180.0F;
        bk_cfg.initial_base_quat_wxyz = Eigen::Vector4f{
            std::cos(yaw_rad * 0.5F), 0.0F, 0.0F, std::sin(yaw_rad * 0.5F)};  // wxyz
    }
    bk_cfg.depth_render_enabled = extras.depth_render;
    auto bk_r = robot::make_mujoco_backend<g1::kDof>(bk_cfg);
    if (!bk_r) { std::fprintf(stderr, "[error] MuJoCo backend: %s\n", bk_r.error().c_str()); return 1; }
    auto backend = std::move(bk_r.value());

    // Ego-camera offscreen renderer — initialized on the MAIN thread (macOS GL
    // window-creation constraint) BEFORE the server thread starts. The state
    // RPC's Image trailer then renders the named MJCF camera on demand.
    bool ego_render_active = false;
    if (extras.ego_render) {
        ego_render_active = backend->init_ego_renderer(
            extras.ego_render_width, extras.ego_render_height, extras.ego_camera_name);
        if (!ego_render_active) {
            std::fprintf(stderr,
                "[motiontrack_mujoco_zmq] --ego_render requested but init failed; "
                "get_state will omit the image.\n");
        }
    }

    // Depth render (same steps as tracker/deploy_cpp/apps/g1/deploy_mujoco.cpp). It reuses the
    // ego offscreen context, so it needs --ego_render (run_sim always passes it). Size, clip and
    // hold come from the config's image group.
    bool depth_active = false;
    if (extras.depth_render) {
        auto spec_r = g1::load_observation_spec(args.config_path);
        if (!spec_r) {
            std::fprintf(stderr, "[error] observation spec: %s\n", spec_r.error().c_str());
            return 1;
        }
        const g1::GroupCfg* img = nullptr;
        for (const auto& g : spec_r.value().groups) {
            if (g.is_image) { img = &g; break; }
        }
        if (img == nullptr) {
            std::fprintf(stderr, "[error] --depth_render given but %s declares no image "
                                 "observation group.\n", args.config_path.c_str());
            return 1;
        }
        if (!ego_render_active) {
            std::fprintf(stderr, "[error] --depth_render needs an active --ego_render context.\n");
            return 1;
        }
        depth_active = backend->init_depth_renderer(bk_cfg.depth_camera_name, img->img_cols,
                                                    img->img_rows, img->clip_min, img->clip_max,
                                                    img->obs_decim);
        if (depth_active) {
            std::fprintf(stderr, "[motiontrack_mujoco_zmq] depth render %dx%d clip=[%.2f,%.2f] m "
                                 "obs_decim=%d\n", img->img_cols, img->img_rows,
                         static_cast<double>(img->clip_min), static_cast<double>(img->clip_max),
                         img->obs_decim);
        } else {
            std::fprintf(stderr, "[motiontrack_mujoco_zmq] --depth_render init failed; the control "
                                 "loop will refuse this image-group config.\n");
        }
    }
    // The backend's depth buffer has ONE consumer, so a relay thread takes each rendered frame,
    // forwards it to the control loop's inbox and keeps a copy for the g1/depth topic. Frames are
    // rendered at most once per 20 ms control tick; the relay polls every 1 ms, so the policy
    // still gets each frame on the next tick.
    core::TripleBuffer<g1::DepthFrame>  depth_relay_tb;
    std::mutex                          depth_copy_mu;
    g1::DepthFrame                      depth_copy;
    bool                                have_depth_copy = false;
    std::atomic<bool>                   depth_relay_done{false};
    std::thread                         depth_relay;
    core::TripleBuffer<g1::DepthFrame>* const depth_inbox
        = depth_active ? &depth_relay_tb : nullptr;

    auto engine = make_engine(args.policy_path);
    if (!engine) return 1;

    // Loop ticks at the inference rate (1/control_dt = 50 Hz). One mj_step per
    // tick, with cfg.decimation substeps inside that mj_step (set above) →
    // 20 ms of physics per tick, matching deploy_g1_mujoco.
    args.control_hz = std::max(1, output_fps);
    std::fprintf(stderr,
        "[motiontrack_mujoco_zmq] substeps=%d, loop=%d Hz, infer=%d Hz (matches deploy_g1_mujoco)\n",
        bk_cfg.substeps, args.control_hz, output_fps);

    // ── Transport seam (owned by main) ───────────────────────────────────────
    core::TripleBuffer<math::RobotState<g1::kDof>> state_tb;
    core::TripleBuffer<policy::RawMotionPlan>      motion_tb;
    core::TripleBuffer<g1::ActivePlanSnapshot>     snapshot_tb;
    std::atomic<std::int32_t>                      inference_tick_mirror{0};
    std::atomic<bool>                              trim_install_pending{false};
    std::atomic<std::int32_t>                      trim_install_tick{0};
    core::TripleBuffer<g1::ExecutedRefEntry>       ref_tb;   // per-tick reference -> server
    std::atomic<bool>                              tracker_ready{true};  // sim: ready immediately

    const std::int32_t scratch_capacity =
        policy::kMaxTransitionSteps
        + static_cast<std::int32_t>(static_cast<float>(policy::RawMotionPlan::kMaxRows)
                                    * static_cast<float>(output_fps) / 30.0F);
    // The active-plan snapshot must hold a whole installed plan, or a trim late in
    // a long plan would have no rows to answer from.
    if (scratch_capacity > g1::kMaxSnapshotRows) {
        std::fprintf(stderr, "[error] control rate %d Hz needs a %d-row plan scratch; the "
                             "active-plan snapshot holds %d\n",
                     output_fps, scratch_capacity, g1::kMaxSnapshotRows);
        return 1;
    }

    halo_system::TrackerServerConfig srv_cfg = extras.srv;
    // The topic PUB runs at the control rate (50 Hz, one per tick).
    srv_cfg.state_pub_hz = output_fps;
    srv_cfg.fallback_base_height_m = cfg.standing_base_pos[2];
    srv_cfg.tracker_ready          = &tracker_ready;  // sim: always true
    // Bridge the backend's ego renderer into the server's transport-neutral
    // image callback (valid for the life of the server; backend outlives it).
    if (ego_render_active) {
        auto* backend_raw = backend.get();
        srv_cfg.render_image = [backend_raw](std::vector<std::uint8_t>& rgb,
                                             int& w, int& h) -> bool {
            return backend_raw->render_ego_image(rgb, w, h);
        };
    }

    // Started here (after every early return) so a failed start never leaves it joinable.
    if (depth_active) {
        auto* src = backend->depth_inbox();
        depth_relay = std::thread([&, src]() {
            while (!depth_relay_done.load(std::memory_order_acquire)) {
                if (const auto* f = src->try_acquire_latest(); f != nullptr) {
                    depth_relay_tb.producer_slot() = *f;
                    depth_relay_tb.publish();
                    const std::scoped_lock lk{depth_copy_mu};
                    depth_copy      = *f;
                    have_depth_copy = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    if (depth_active) {
        srv_cfg.depth_pub_port = kDepthPubPort;
        srv_cfg.depth_provider = [&](g1::DepthFrame& out) -> bool {
            const std::scoped_lock lk{depth_copy_mu};
            if (!have_depth_copy) return false;
            out = depth_copy;
            return true;
        };
        std::fprintf(stderr, "[motiontrack_mujoco_zmq] g1/depth @ %d (%d Hz)\n",
                     srv_cfg.depth_pub_port, srv_cfg.depth_pub_hz);
    }

    g1::ControlLoop* control_loop_ptr = nullptr;   // put_plan acks after install, as on the robot
    srv_cfg.control_loop = &control_loop_ptr;
    halo_system::TrackerServer server{
        state_tb, motion_tb, &snapshot_tb,
        &inference_tick_mirror, &trim_install_pending, &trim_install_tick,
        remap, &ref_tb, srv_cfg};
    server.start();

    auto run_loop = [&]() {
        return g1::run_control_loop(args, cfg, remap, switcher.standing(),
                                    *backend, *engine,
                                    math::ControlState::kControl, &switcher,
                                    &state_tb, &motion_tb, scratch_capacity,
                                    &inference_tick_mirror, &snapshot_tb,
                                    &trim_install_pending, &trim_install_tick,
                                    /*tracker_ready_out=*/nullptr,
                                    /*control_loop_out=*/&control_loop_ptr,
                                    /*on_loop_exit=*/[&]() { server.stop(); },
                                    depth_inbox, /*remote_buttons_out=*/nullptr,
                                    /*ref_outbox=*/&ref_tb);
    };
    auto report = [&](int code) {
        server.stop();
        depth_relay_done.store(true, std::memory_order_release);
        if (depth_relay.joinable()) depth_relay.join();
        std::fprintf(stderr,
            "[motiontrack_mujoco_zmq] state_queries=%llu plans=%llu trims=%llu\n",
            static_cast<unsigned long long>(server.state_queries()),
            static_cast<unsigned long long>(server.plans_received()),
            static_cast<unsigned long long>(server.trim_requests()));
        return code;
    };

    return report(run_loop());
}
