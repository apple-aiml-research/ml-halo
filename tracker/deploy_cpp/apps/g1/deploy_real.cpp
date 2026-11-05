//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// deploy_g1_real — run the G1 motion tracker on the real robot via Unitree
// SDK2.
//
// Requires cmake options: RT_DEPLOY_WITH_UNITREE=ON
//                         plus at least one of RT_DEPLOY_WITH_TRT / RT_DEPLOY_WITH_ORT.
//
// Optional depth (the depth student's `student_depth` obs group) requires
// RT_DEPLOY_WITH_REALSENSE=ON and is opt-in at runtime:
//
//   --depth_config configs/depth_d435i.yaml     # start a depth producer
//   --depth_source cleaned|raw                  # override the YAML's choice
//   --depth_plan   stereo.plan                  # override the stereo .plan,
//                                               # e.g. to A/B two stereo models
//                                               # (path taken literally; the
//                                               #  {ROOT_DIR} placeholder is a
//                                               #  YAML-only convenience)
//
// Without --depth_config no camera is opened, no depth thread runs, and the
// inbox stays null (proprioceptive-only policies). With it, a non-RT producer
// publishes 48x84 metric depth into a wait-free TripleBuffer that the control
// loop drains with one non-blocking read per tick — it never blocks on, waits
// for, or is slowed by the depth thread.
//
// Intended to be run with CAP_SYS_NICE + CAP_IPC_LOCK so SCHED_FIFO + mlockall
// succeed:
//   sudo setcap 'cap_sys_nice=eip cap_ipc_lock=eip' ./deploy_g1_real
// Run as a non-root user afterwards.
// NOTE: capabilities live on the inode and must be re-applied after EVERY
// rebuild; a cap-bearing binary also ignores LD_LIBRARY_PATH, so CUDA / TensorRT
// / librealsense lib dirs must be reachable via /etc/ld.so.conf.d + ldconfig or
// a baked RUNPATH.

#include "apps/g1/common.hpp"
#include "apps/g1/depth_setup.hpp"

#include "rt_deploy/g1/unitree_backend.hpp"

#if defined(RT_DEPLOY_WITH_ORT)
#    include "rt_deploy/inference/ort_engine.hpp"
#endif
#if defined(RT_DEPLOY_WITH_TRT)
#    include "rt_deploy/inference/trt_engine.hpp"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace rt_deploy;

namespace {

struct RealExtraArgs {
    std::string network_interface = "eth0";
    // Depth for the depth student. Empty config => no depth producer is
    // started and the depth inbox stays null (proprioceptive-only policies).
    // The flags and every refusal around them live in depth_setup.hpp.
    g1::DepthArgs depth;
    // DRY RUN: run the whole pipeline but never publish a command. See
    // UnitreeBackendConfig::send_enabled.
    bool        no_send = false;
    // Enter CONTROL at tick 0 instead of waiting for the operator's Start+A.
    // Inference only runs in CONTROL, so a timing experiment needs this — but on
    // a live robot it would apply policy output to an un-ramped pose, so it is
    // interlocked to --no_send below.
    bool        start_in_control = false;
};

bool parse_real_extras(const std::vector<std::string>& rest, RealExtraArgs& out) {
    for (std::size_t i = 0; i < rest.size(); ++i) {
        const int        depth_rc = g1::parse_depth_flag(rest, i, out.depth);
        if (depth_rc < 0) return false;
        if (depth_rc > 0) continue;
        std::string_view a{rest[i]};
        if (a == "--net") {
            const std::string* v = core::take_value(rest, i);
            if (v == nullptr) return false;
            out.network_interface = *v;
        } else if (a == "--no_send") {
            out.no_send = true;
        } else if (a == "--start_in_control") {
            out.start_in_control = true;
        } else {
            std::fprintf(stderr, "error: unknown argument %s\n", std::string{a}.c_str());
            return false;
        }
    }
    if (!g1::validate_depth_args(out.depth)) {
        return false;
    }
    // SAFETY INTERLOCK. --start_in_control skips the ramp to the default pose and
    // hands the policy's very first output straight to the joints from whatever
    // pose the robot happens to be in. That is only ever acceptable when nothing
    // reaches the wire, so refuse the combination outright rather than trusting
    // the operator to remember. Refusing is cheap; the failure mode is not.
    if (out.start_in_control && !out.no_send) {
        std::fprintf(stderr,
                     "error: --start_in_control requires --no_send.\n"
                     "       Entering CONTROL at tick 0 skips the ramp to the default pose, so the\n"
                     "       policy's first action would be applied to an un-ramped pose. That is\n"
                     "       only safe as a dry run. For a real run, press Start then A.\n");
        return false;
    }
    return true;
}

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
    cfg.provider  = inference::OrtProvider::kCuda;
    auto r        = inference::make_ort_engine(cfg);
    if (!r) {
        // A CPU-only ONNX Runtime has no CUDA provider: run the policy on the CPU instead.
        std::fprintf(stderr, "[warn] ORT CUDA provider unavailable (%s); using the CPU provider\n", r.error().c_str());
        cfg.provider = inference::OrtProvider::kCpu;
        r            = inference::make_ort_engine(cfg);
    }
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
    g1::CommonArgs            args;
    std::vector<std::string>  rest;
    const std::string extra_usage = std::string{"[--net IFACE] "} + g1::kDepthUsage
                                     + "\n                                 "
                                       "[--no_send] [--start_in_control]";
    if (!g1::parse_common_args(argc, argv, "deploy_g1_real", extra_usage.c_str(),
                                args, rest)) {
        return 1;
    }
    RealExtraArgs extras;
    if (!parse_real_extras(rest, extras)) {
        return 1;
    }
    // Sim-eval aid only: it freezes the whole control loop after a plan install,
    // which on hardware means no fresh command for that long.
    if (args.sim_pause_after_plan_s > 0.0F) {
        std::fprintf(stderr, "error: --sim_pause_after_plan_s is a sim-only option\n");
        return 1;
    }

    // No motion data ships (see tracker/data/README.md). Bare names resolve to
    // <nearest ancestor data/>/<name>.csv, as in the Python deployment.
    if (args.motion_path.empty()) {
        std::fprintf(stderr, "[error] --motion is required (a CSV motion file, see tracker/data/README.md)\n");
        return 1;
    }
    args.motion_path = g1::resolve_motion_path(args.motion_path, args.config_path);

    if (extras.no_send) {
        std::fprintf(stderr,
            "\n"
            "  ====================  DRY RUN — NO COMMANDS WILL BE SENT  ====================\n"
            "  Every LowCmd_ is built and CRC'd exactly as normal and then DISCARDED, so the\n"
            "  CPU and GPU load, the 50 Hz timing and the jitter are the real thing while the\n"
            "  joints receive nothing: a timing run under the real load.\n"
            "\n"
            "  This is NOT a safety mechanism on its own. The robot is never commanded at all,\n"
            "  INCLUDING the damping command normally sent on exit — what the joints do when\n"
            "  lowcmd stops arriving is firmware behaviour this program does not control.\n"
            "  Support the robot on a stand and disable its motors or engage the E-stop.\n"
            "  =============================================================================\n\n");
    }
    std::fprintf(stderr, "[SAFETY] Make sure:\n"
                          "  1. Robot is in a safe, open area\n"
                          "  2. Emergency stop is within reach\n"
                          "  3. Policy has been tested in MuJoCo first\n");

    auto cfg_r = g1::load_config(args.config_path);
    if (!cfg_r) {
        std::fprintf(stderr, "[error] config: %s\n", cfg_r.error().c_str());
        return 1;
    }
    const auto cfg   = std::move(cfg_r.value());
    const auto remap = policy::build_joint_remap<g1::kDof>(cfg.deployment_joint_order,
                                                            cfg.training_joint_order);

    policy::MotionLoaderConfig mcfg;
    mcfg.csv_path                = args.motion_path;
    mcfg.input_fps               = args.motion_fps;
    mcfg.output_fps              = static_cast<int>(std::round(1.0F / cfg.control_dt));
    mcfg.joint_remap             = &remap;
    mcfg.standing_joint_pos      = cfg.standing_joint_pos;
    mcfg.standing_base_pos       = cfg.standing_base_pos;
    mcfg.standing_base_quat_wxyz = cfg.standing_base_quat_wxyz;
    mcfg.min_output_frames       = g1::motion_pad_frames(args.duration_s, mcfg.output_fps);
    auto motion_r = policy::load_motion(mcfg);
    if (!motion_r) {
        std::fprintf(stderr, "[error] motion: %s\n", motion_r.error().c_str());
        return 1;
    }
    auto motion = std::move(motion_r.value());

    // Build the standing-pose motion once at startup and wrap both motions in
    // a MotionSwitcher so the operator can swap between them at runtime
    // (B → standup, A → replay original). Mirrors Python deploy_real.py:
    // do_standup / replay_original_motion.
    auto standing_r = g1::make_standing_motion(cfg, remap, mcfg.min_output_frames);
    if (!standing_r) {
        std::fprintf(stderr, "[error] standing motion: %s\n", standing_r.error().c_str());
        return 1;
    }
    g1::MotionSwitcher switcher{&motion, std::move(standing_r.value())};

    g1::UnitreeBackendConfig bk_cfg;
    bk_cfg.network_interface = extras.network_interface;
    bk_cfg.send_enabled      = !extras.no_send;
    // Tell the control loop that nothing actuates (see ControlLoopConfig::dry_run).
    args.dry_run             = extras.no_send;
    auto bk_r                = g1::make_unitree_backend(bk_cfg);
    if (!bk_r) {
        std::fprintf(stderr, "[error] Unitree backend: %s\n", bk_r.error().c_str());
        return 1;
    }
    auto backend = std::move(bk_r.value());

    auto engine = make_engine(g1::parse_engine_choice(args.engine), args.policy_path);
    if (!engine) {
        return 1;
    }

    // Mirror Python deploy_real.py / run_real_step semantics:
    //   - Loop ticks at control_dt (50 Hz with cfg.control_dt = 0.02) and runs
    //     get_observation() + infer + apply_action() every iteration — control
    //     and inference share one rate (the sim's physics sub-stepping is a
    //     MuJoCo-only concern, handled in the backend, not on the robot).
    //   - State machine starts in kInit; operator presses Start to begin the
    //     2 s cosine ramp to the default pose, then A to enter CONTROL.
    // Override any user-supplied --control_hz: getting it wrong on real
    // hardware is unsafe.
    args.control_hz = std::max(1, static_cast<int>(std::round(1.0F / cfg.control_dt)));

    std::fprintf(stderr, "[INFO] Waiting for robot state on interface %s...\n",
                 extras.network_interface.c_str());

    // ── Optional depth producer (depth student `student_depth`) ──────────────
    // Owns the inbox and the producer for the rest of main, and refuses at
    // startup on a bad config, a clip disagreement, a policy config with no image
    // group, or depth threads that would outrank the control loop. See
    // depth_setup.hpp.
    g1::DepthRuntime depth;
    if (g1::setup_depth(args, extras.depth, depth) != 0) {
        return 1;
    }

    // Depth summary only — deliberately NOT depth.source->stop(). on_loop_exit
    // fires between loop.run() returning and backend.cleanup() publishing the
    // damping command, so a slow teardown here delays the robot reaching a safe
    // state; the producer is torn down by ~DepthRuntime after run_control_loop
    // has damped instead. Empty when depth is off.
    std::function<void()> on_loop_exit = depth.stats_reporter();

    return g1::run_control_loop(args, cfg, remap, &motion,
                                 *backend, *engine,
                                 extras.start_in_control ? math::ControlState::kControl
                                                         : math::ControlState::kInit,
                                 &switcher,
                                 /*state_outbox=*/nullptr, /*motion_inbox=*/nullptr,
                                 /*motion_scratch_capacity=*/0,
                                 /*inference_tick_out=*/nullptr,
                                 /*active_plan_snapshot_out=*/nullptr,
                                 /*trim_install_pending=*/nullptr,
                                 /*trim_install_tick=*/nullptr,
                                 /*tracker_ready_out=*/nullptr,
                                 /*control_loop_out=*/nullptr,
                                 /*on_loop_exit=*/std::move(on_loop_exit),
                                 /*depth_inbox=*/depth.inbox_ptr);
}
