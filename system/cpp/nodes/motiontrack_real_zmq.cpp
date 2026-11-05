//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// motiontrack_real_zmq — halo_system tracker NODE (real Unitree G1 hardware).
//
// tracker's G1 control loop on the Unitree DDS backend + a halo_comm
// TrackerServer that receives motion plans from an external generator and
// answers state / trim over protobuf. The transport lives in halo_comm.
//
// Requires CMake: RT_DEPLOY_WITH_UNITREE=ON + (RT_DEPLOY_WITH_ORT or _TRT); the depth student
// (--depth_config, D435i) also needs RT_DEPLOY_WITH_REALSENSE=ON.
// Run with CAP_SYS_NICE + CAP_IPC_LOCK (see deploy_real.cpp).

#include "apps/g1/common.hpp"
#include "apps/g1/depth_setup.hpp"
#include "transport/tracker_server.hpp"

#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/g1/unitree_backend.hpp"

#if defined(RT_DEPLOY_WITH_ORT)
#    include "rt_deploy/inference/ort_engine.hpp"
#endif
#if defined(RT_DEPLOY_WITH_TRT)
#    include "rt_deploy/inference/trt_engine.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace rt_deploy;

namespace {

struct RealZmqExtraArgs {
    std::string network_interface = "eth0";
    halo_system::TrackerServerConfig srv;
    g1::DepthArgs depth;  // --depth_config / --depth_source / --depth_plan (depth student)
};

bool parse_extras(const std::vector<std::string>& rest, RealZmqExtraArgs& out) {
    auto take = [&](std::size_t& i, const char* flag) -> const std::string* {
        if (i + 1 >= rest.size()) { std::fprintf(stderr, "error: missing value for %s\n", flag); return nullptr; }
        return &rest[++i];
    };
    for (std::size_t i = 0; i < rest.size(); ++i) {
        const int depth_rc = g1::parse_depth_flag(rest, i, out.depth);
        if (depth_rc < 0) return false;
        if (depth_rc > 0) continue;
        const std::string& a = rest[i];
        if (a == "--net") {
            const auto* v = take(i, "--net"); if (v == nullptr) return false;
            out.network_interface = *v;
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
        } else {
            std::fprintf(stderr, "error: unknown argument %s\n", a.c_str());
            return false;
        }
    }
    return g1::validate_depth_args(out.depth);
}

std::unique_ptr<inference::IInferenceEngine>
make_engine(g1::PreferredEngine choice, const std::string& policy_path) {
#if defined(RT_DEPLOY_WITH_TRT)
    if (choice == g1::PreferredEngine::kTrt
        || (choice == g1::PreferredEngine::kAuto && policy_path.ends_with(".plan"))) {
        inference::TrtEngineConfig cfg;
        cfg.plan_path = policy_path;
        auto r        = inference::make_trt_engine(cfg);
        if (!r) { std::fprintf(stderr, "[error] TRT init: %s\n", r.error().c_str()); return nullptr; }
        return std::move(r.value());
    }
#endif
#if defined(RT_DEPLOY_WITH_ORT)
    (void)choice;
    inference::OrtEngineConfig cfg;
    cfg.onnx_path = policy_path;
    cfg.provider  = inference::OrtProvider::kCuda;
    auto r        = inference::make_ort_engine(cfg);
    if (!r) {  // a CPU-only ONNX Runtime has no CUDA provider
        std::fprintf(stderr, "[warn] ORT CUDA provider unavailable (%s); using the CPU provider\n", r.error().c_str());
        cfg.provider = inference::OrtProvider::kCpu;
        r            = inference::make_ort_engine(cfg);
    }
    if (!r) { std::fprintf(stderr, "[error] ORT init: %s\n", r.error().c_str()); return nullptr; }
    return std::move(r.value());
#endif
    std::fprintf(stderr, "[error] no inference backend compiled in.\n");
    (void)policy_path; (void)choice;
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    g1::CommonArgs           args;
    std::vector<std::string> rest;
    if (!g1::parse_common_args(argc, argv, "motiontrack_real_zmq",
                                "[--net IFACE] [--bind_ip IP] [--state_port N] "
                                "[--plan_port N] [--trim_port N] [--state_pub_port N] "
                                "[--depth_config YAML] [--depth_source cleaned|raw] [--depth_plan PLAN]",
                                args, rest)) {
        return 1;
    }
    RealZmqExtraArgs extras;
    if (!parse_extras(rest, extras)) return 1;
    // Sim-eval aid only: it freezes the whole control loop after a plan install,
    // which on hardware means no fresh command for that long.
    if (args.sim_pause_after_plan_s > 0.0F) {
        std::fprintf(stderr, "error: --sim_pause_after_plan_s is a sim-only option\n");
        return 1;
    }

    std::fprintf(stderr, "[SAFETY] Robot in open area; e-stop in reach; policy tested in sim.\n");

    auto cfg_r = g1::load_config(args.config_path);
    if (!cfg_r) { std::fprintf(stderr, "[error] config: %s\n", cfg_r.error().c_str()); return 1; }
    const auto cfg   = std::move(cfg_r.value());
    const auto remap = policy::build_joint_remap<g1::kDof>(cfg.deployment_joint_order,
                                                           cfg.training_joint_order);

    const int output_fps = static_cast<int>(std::round(1.0F / cfg.control_dt));
    // Standing pose only bridges boot → first generator plan, and the control
    // loop holds the last frame past frame_count, so bound its buffer by a
    // fixed horizon rather than the full --duration (now 60000 s by default,
    // which would allocate ~3M identical held-pose frames).
    const float        standing_horizon_s = std::min(args.duration_s, 600.0F);
    const std::int32_t min_output_frames = static_cast<std::int32_t>(
        standing_horizon_s * static_cast<float>(output_fps) + 5.0F * static_cast<float>(output_fps));

    auto standing_r = g1::make_standing_motion(cfg, remap, min_output_frames);
    if (!standing_r) { std::fprintf(stderr, "[error] standing: %s\n", standing_r.error().c_str()); return 1; }
    g1::MotionSwitcher switcher{/*original=*/nullptr, std::move(standing_r.value())};

    g1::UnitreeBackendConfig bk_cfg;
    bk_cfg.network_interface = extras.network_interface;
    auto bk_r                = g1::make_unitree_backend(bk_cfg);
    if (!bk_r) { std::fprintf(stderr, "[error] Unitree backend: %s\n", bk_r.error().c_str()); return 1; }
    auto backend = std::move(bk_r.value());

    auto engine = make_engine(g1::parse_engine_choice(args.engine), args.policy_path);
    if (!engine) return 1;

    args.control_hz = std::max(1, output_fps);

    core::TripleBuffer<math::RobotState<g1::kDof>> state_tb;
    core::TripleBuffer<policy::RawMotionPlan>      motion_tb;
    core::TripleBuffer<g1::ActivePlanSnapshot>     snapshot_tb;
    std::atomic<std::int32_t>                      inference_tick_mirror{0};
    std::atomic<bool>                              trim_install_pending{false};
    std::atomic<std::int32_t>                      trim_install_tick{0};
    core::TripleBuffer<g1::ExecutedRefEntry>       ref_tb;   // per-tick reference -> server
    std::atomic<bool>                              tracker_ready{false};
    std::atomic<std::uint32_t>                     remote_buttons{0};   // gamepad bits -> g1/state
    g1::ControlLoop*                               control_loop_ptr = nullptr;

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

    // Depth producer for the depth student: refuses a bad config, a clip mismatch, a policy config without an
    // image group or depth threads at/above the loop's priority, and waits for the first frame (depth_setup.hpp).
    // Started before the server so no plan is accepted while the camera warms up. Inert without --depth_config.
    g1::DepthRuntime depth;
    if (g1::setup_depth(args, extras.depth, depth) != 0) return 1;
    std::function<void()> depth_stats = depth.stats_reporter();
    // The depth inbox has ONE consumer, so (as in motiontrack_mujoco_zmq) a relay thread takes each frame,
    // forwards it to the control loop's inbox and keeps a copy for the g1/depth topic (the viewer). The
    // relay polls every 1 ms; frames keep their own stamp, so the loop's staleness check is unchanged.
    core::TripleBuffer<g1::DepthFrame>  depth_relay_tb;
    std::mutex                          depth_copy_mu;
    g1::DepthFrame                      depth_copy;
    bool                                have_depth_copy = false;
    std::atomic<bool>                   depth_relay_done{false};
    std::thread                         depth_relay;
    core::TripleBuffer<g1::DepthFrame>* const depth_inbox
        = depth.inbox_ptr != nullptr ? &depth_relay_tb : nullptr;

    halo_system::TrackerServerConfig srv_cfg = extras.srv;
    srv_cfg.state_pub_hz           = output_fps;
    srv_cfg.fallback_base_height_m = cfg.standing_base_pos[2];
    srv_cfg.tracker_ready         = &tracker_ready;
    srv_cfg.remote_buttons        = &remote_buttons;
    srv_cfg.control_loop          = &control_loop_ptr;
    if (depth_inbox != nullptr) {
        srv_cfg.depth_pub_port = 5553;   // g1/depth, same port as the MuJoCo node
        srv_cfg.depth_provider = [&](g1::DepthFrame& out) -> bool {
            const std::scoped_lock lk{depth_copy_mu};
            if (!have_depth_copy) return false;
            out = depth_copy;
            return true;
        };
        std::fprintf(stderr, "[motiontrack_real_zmq] g1/depth @ %d (%d Hz)\n",
                     srv_cfg.depth_pub_port, srv_cfg.depth_pub_hz);
    }
    // The robot computer hosts the discovery registry (both machines export HALO_COMM_REGISTRY=<robot>:5560,
    // system/README.md#run-real); the server's channels register with it.
    halo_comm::init();

    halo_system::TrackerServer server{
        state_tb, motion_tb, &snapshot_tb,
        &inference_tick_mirror, &trim_install_pending, &trim_install_tick,
        remap, &ref_tb, srv_cfg};
    server.start();

    std::fprintf(stderr, "[INFO] Waiting for robot state on %s...\n",
                 extras.network_interface.c_str());

    // Started here (no early return before the join below).
    if (depth_inbox != nullptr) {
        auto* src = depth.inbox_ptr;
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

    const int code = g1::run_control_loop(args, cfg, remap, switcher.standing(),
                                          *backend, *engine,
                                          math::ControlState::kInit, &switcher,
                                          &state_tb, &motion_tb, scratch_capacity,
                                          &inference_tick_mirror, &snapshot_tb,
                                          &trim_install_pending, &trim_install_tick,
                                          /*tracker_ready_out=*/&tracker_ready,
                                          /*control_loop_out=*/&control_loop_ptr,
                                          // Stop the server before the loop is destroyed.
                                          /*on_loop_exit=*/[&]() {
                                              server.stop();
                                              if (depth_stats) depth_stats();
                                          },
                                          /*depth_inbox=*/depth_inbox,
                                          /*remote_buttons_out=*/&remote_buttons,
                                          /*ref_outbox=*/&ref_tb);
    server.stop();
    depth_relay_done.store(true, std::memory_order_release);   // after the exit damping
    if (depth_relay.joinable()) depth_relay.join();
    std::fprintf(stderr, "[motiontrack_real_zmq] state_queries=%llu plans=%llu trims=%llu\n",
                 static_cast<unsigned long long>(server.state_queries()),
                 static_cast<unsigned long long>(server.plans_received()),
                 static_cast<unsigned long long>(server.trim_requests()));
    return code;
}
