//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// App helpers: CLI parser, signal handler, and the glue that wires every
// component together once a backend + inference engine exist. Kept short so
// each deploy_*.cpp binary is a thin entry point.

#include "rt_deploy/core/cli_args.hpp"
#include "rt_deploy/core/clock.hpp"
#include "rt_deploy/core/jitter_histogram.hpp"
#include "rt_deploy/core/loop_trace.hpp"
#include "rt_deploy/core/parse_int.hpp"
#include "rt_deploy/core/rate_probe.hpp"
#include "rt_deploy/core/rt_thread.hpp"
#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/g1/action_decoder.hpp"
#include "rt_deploy/g1/config.hpp"
#include "rt_deploy/g1/control_loop.hpp"
#include "rt_deploy/g1/depth_clip_registry.hpp"
#include "rt_deploy/g1/motion_switcher.hpp"
#include "rt_deploy/g1/observation_builder.hpp"
#include "rt_deploy/g1/state_machine.hpp"
#include "rt_deploy/inference/i_inference_engine.hpp"
#include "rt_deploy/policy/joint_remap.hpp"
#include "rt_deploy/policy/motion_loader.hpp"
#include "rt_deploy/policy/raw_motion_plan.hpp"
#include "rt_deploy/robot/i_robot_backend.hpp"
#include "rt_deploy/robot/safety_monitor.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace rt_deploy::g1 {

// CLI arguments shared by deploy_g1_mujoco, deploy_g1_real and the system/cpp
// tracker nodes.
struct CommonArgs {
    std::string   config_path;
    std::string   policy_path;    // .plan (TRT) or .onnx (ORT)
    std::string   motion_path;    // CSV (required by deploy_g1_mujoco / deploy_g1_real;
                                  // the system/cpp nodes ignore it)
    std::string   engine   = "auto";  // "trt" / "ort" / "auto"
    int           motion_fps = 30;
    // Default 60000 s (≈ 16 h), matching the Python tracker nodes. The
    // system/cpp tracker nodes size their standing-pose buffer from a separate
    // 600 s horizon, so this only sets when the loop self-exits.
    float         duration_s = 60000.0F;
    int           control_hz = 50;   // == inference rate; deploy apps override
                                      // from cfg.control_dt (G1: 1/0.02 = 50 Hz)
    int           rt_fifo_priority = 0;      // 0 = no SCHED_FIFO
    int           rt_cpu_affinity  = -1;
    bool          lock_memory      = false;
    bool          profile          = false;  // --profile: per-thread Hz + compute stats
    // --trace_csv PATH: append one CSV row per control iteration (period, total
    // work time, obs/infer/act split, wall-clock timestamp). Empty = off.
    // Complements --profile: that one prints aggregate percentiles, this one
    // keeps every sample so the timing can be overlaid on an external video.
    std::string   trace_csv;
    // --depth_stats_every N: every N INFERENCE ticks, print the actual
    // `student_depth` group tensor statistics (post-encode, as the engine
    // received them), the raw metric DepthFrame they came from, and the action
    // vector the engine returned. 0 (default) = off, and off costs one integer
    // compare per inference tick.
    //
    // Diagnostic only — it never changes what the policy is fed. It writes to
    // stderr FROM THE RT CONTROL THREAD, so keep N at a second or more (50 at
    // 50 Hz) on a powered run, or redirect stderr to a file: an unbuffered write
    // to a pipe with a slow reader blocks the control loop.
    int           depth_stats_every = 0;
    // --depth_stale_far_ticks N: abandon a frozen depth frame for the far plane
    // after N stale control ticks. -1 keeps the ControlLoopConfig default
    // (25 ticks = 500 ms at 50 Hz); 0 disables the fallback and holds the frozen
    // frame forever, which is only appropriate for a bench experiment that needs
    // to observe that behaviour. See ControlLoopConfig::depth_stale_far_ticks.
    int           depth_stale_far_ticks = -1;
    // Set by deploy_g1_real's --no_send. Tells the control loop that nothing
    // actuates, which relaxes the image-group-without-depth refusal so a no-depth
    // timing baseline can be measured. See ControlLoopConfig::dry_run.
    bool          dry_run = false;
    // Which depth estimator actually runs, recorded by setup_depth()
    // (apps/g1/depth_setup.hpp) AFTER any --depth_source / --depth_plan
    // override, so logs report what ran rather than what the YAML asked for.
    // Empty => no depth producer was started.
    std::string   depth_source_name;   // "cleaned" / "raw"
    std::string   depth_plan_path;     // stereo .plan; empty for the raw path
    // Proceed even though the config's depth encode clip differs from the
    // training clip that g1/depth_clip_registry.cpp expects (0.2/2.5). Only for
    // a policy genuinely trained with a different clip — take that clip from
    // the run's own logged training config and update the registry rather than
    // keep passing this.
    bool          allow_depth_clip_mismatch = false;
    // Sim-eval aid (NOT for the real robot): hold the playback counter for this
    // many seconds after a generated plan installs, so a slow external orchestrator
    // in a separate process isn't outrun. 0 (default) = off. Set only by the
    // multi-process sim eval; see ControlLoopConfig::sim_pause_after_plan.
    float         sim_pause_after_plan_s = 0.0F;
    // Anchor the motion onto the robot at startup and at CONTROL entry (see
    // ControlLoopConfig::align_motion_to_robot). Cleared by deploy_g1_mujoco's
    // --no_align_motion.
    bool          align_motion = true;
};

// Strict float parse for CLI values: the whole string must be a finite number
// (strtof would read "0.5s" as 0.5 and "abc" as 0). On failure `out` is left
// untouched.
[[nodiscard]] inline bool parse_float_strict(const char* s, float& out) noexcept {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    char* end = nullptr;
    errno     = 0;
    const float v = std::strtof(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !std::isfinite(v)) {
        return false;
    }
    out = v;
    return true;
}

using core::parse_int_strict;

// Minimal getopt-free arg parser. Returns true on success, prints usage on
// failure. Unknown flags yield an error.
inline bool parse_common_args(int argc, char** argv, const char* prog,
                                const char* extra_usage, CommonArgs& out,
                                std::vector<std::string>& rest) {
    auto usage = [&]() {
        std::fprintf(
            stderr,
            "usage: %s [common] %s\n"
            "  --config PATH          YAML config (e.g. configs/g1_v2.yaml)\n"
            "  --policy PATH          .plan (TensorRT) or .onnx (ONNX Runtime) policy\n"
            "  --motion PATH          CSV motion file, or a bare name resolved to <data>/<name>.csv (required by deploy_g1_mujoco/deploy_g1_real; ignored by the ZMQ nodes)\n"
            "  --engine auto|trt|ort  Inference backend (default auto)\n"
            "  --motion_fps N         (default 30)\n"
            "  --duration S           Max runtime in seconds (default 60000)\n"
            "  --control_hz N         Control rate; deploy apps derive it from config control_dt (G1: 50 Hz), so this is normally ignored\n"
            "  --rt_prio N            SCHED_FIFO priority 1..99 (default 0 = off)\n"
            "  --rt_cpu N             CPU to pin control thread to (default -1 = none)\n"
            "  --lock_memory          Call mlockall(MCL_CURRENT|MCL_FUTURE)\n"
            "  --profile              Print per-thread Hz + compute-time stats every 2 s\n"
            "  --trace_csv PATH       Record per-iteration loop timing to PATH (CSV, non-blocking)\n"
            "  --depth_stats_every N  Every N inference ticks, log the actual student_depth tensor + action stats (default 0 = off)\n"
            "  --allow_depth_clip_mismatch  Proceed when the config's depth clip differs from the one this checkpoint trained with\n"
            "  --depth_stale_far_ticks N  Feed the far plane after N stale depth ticks (-1 = keep the default 25 = 500ms; 0 = hold forever)\n"
            "  --sim_pause_after_plan_s S  Sim eval only: hold the playback counter S s after a plan installs (default 0 = off)\n",
            prog, extra_usage);
    };

    // Refuses a value that is itself a flag (core::take_value), so a forgotten
    // value is an error rather than swallowing the next option.
    auto next = [&](int& i) -> const char* { return core::take_value(argc, argv, i); };

    for (int i = 1; i < argc; ++i) {
        std::string_view a{argv[i]};
        if (a == "-h" || a == "--help") {
            usage();
            return false;
        } else if (a == "--config") {
            const char* v = next(i); if (v == nullptr) return false; out.config_path = v;
        } else if (a == "--policy") {
            const char* v = next(i); if (v == nullptr) return false; out.policy_path = v;
        } else if (a == "--motion") {
            const char* v = next(i); if (v == nullptr) return false; out.motion_path = v;
        } else if (a == "--engine") {
            const char* v = next(i); if (v == nullptr) return false; out.engine = v;
        } else if (a == "--motion_fps") {
            const char* v = next(i); if (v == nullptr) return false;
            if (!parse_int_strict(v, out.motion_fps) || out.motion_fps < 1
                || out.motion_fps > 1000) {
                std::fprintf(stderr, "error: --motion_fps expects an integer in 1..1000, got '%s'\n",
                             v);
                return false;
            }
        } else if (a == "--duration") {
            const char* v = next(i); if (v == nullptr) return false;
            if (!parse_float_strict(v, out.duration_s) || !(out.duration_s > 0.0F)) {
                std::fprintf(stderr, "error: --duration expects seconds > 0, got '%s'\n", v);
                return false;
            }
        } else if (a == "--control_hz") {
            const char* v = next(i); if (v == nullptr) return false;
            if (!parse_int_strict(v, out.control_hz) || out.control_hz < 1
                || out.control_hz > 1000) {
                std::fprintf(stderr, "error: --control_hz expects an integer in 1..1000, got '%s'\n",
                             v);
                return false;
            }
        } else if (a == "--rt_prio") {
            const char* v = next(i); if (v == nullptr) return false;
            if (!parse_int_strict(v, out.rt_fifo_priority) || out.rt_fifo_priority < 0
                || out.rt_fifo_priority > 99) {
                std::fprintf(stderr, "error: --rt_prio expects an integer in 0..99, got '%s'\n", v);
                return false;
            }
        } else if (a == "--rt_cpu") {
            const char* v = next(i); if (v == nullptr) return false;
            if (!parse_int_strict(v, out.rt_cpu_affinity) || out.rt_cpu_affinity < -1
                || out.rt_cpu_affinity > 1023) {
                std::fprintf(stderr,
                             "error: --rt_cpu expects a CPU index (or -1 for none), got '%s'\n", v);
                return false;
            }
        } else if (a == "--lock_memory") {
            out.lock_memory = true;
        } else if (a == "--profile") {
            out.profile = true;
        } else if (a == "--trace_csv") {
            const char* v = next(i); if (v == nullptr) return false; out.trace_csv = v;
        } else if (a == "--depth_stats_every") {
            const char* v = next(i); if (v == nullptr) return false;
            if (!parse_int_strict(v, out.depth_stats_every)) {
                std::fprintf(stderr, "error: --depth_stats_every expects an integer, got '%s'\n", v);
                return false;
            }
        } else if (a == "--allow_depth_clip_mismatch") {
            out.allow_depth_clip_mismatch = true;
        } else if (a == "--depth_stale_far_ticks") {
            const char* v = next(i); if (v == nullptr) return false;
            if (!parse_int_strict(v, out.depth_stale_far_ticks)) {
                std::fprintf(stderr, "error: --depth_stale_far_ticks expects an integer, got '%s'\n",
                             v);
                return false;
            }
        } else if (a == "--sim_pause_after_plan_s") {
            const char* v = next(i); if (v == nullptr) return false;
            if (!parse_float_strict(v, out.sim_pause_after_plan_s)
                || out.sim_pause_after_plan_s < 0.0F) {
                std::fprintf(stderr,
                             "error: --sim_pause_after_plan_s expects seconds >= 0, got '%s'\n", v);
                return false;
            }
        } else {
            // Pass through for the binary-specific parser.
            rest.emplace_back(argv[i]);
        }
    }

    if (out.config_path.empty()) {
        out.config_path = "configs/g1_v2.yaml";  // default; override with --config
    }
    if (out.policy_path.empty()) {
        std::fprintf(stderr, "error: --policy is required\n");
        usage();
        return false;
    }
    // parse_engine_choice() maps anything unknown to auto, so a typo would run a
    // backend the operator did not ask for.
    if (!core::is_engine_choice(out.engine)) {
        std::fprintf(stderr, "error: --engine must be auto, trt or ort, got '%s'\n",
                     out.engine.c_str());
        return false;
    }
    // A negative cadence would silently disable the diagnostic (the loop tests
    // `> 0`), which is the opposite of what someone typing the flag wants.
    if (out.depth_stats_every < 0) {
        std::fprintf(stderr, "error: --depth_stats_every must be >= 0 (0 = off)\n");
        return false;
    }
    // -1 is the "keep the built-in default" sentinel; any other negative value
    // has no meaning and would otherwise be silently treated as that default.
    if (out.depth_stale_far_ticks < -1) {
        std::fprintf(stderr,
                     "error: --depth_stale_far_ticks must be >= 0 (0 = hold forever), "
                     "or -1 for the default\n");
        return false;
    }
    return true;
}

// Installs SIGINT / SIGTERM handlers that set a shared flag. The caller
// should poll this and call ControlLoop::request_stop() when set.
inline std::atomic<bool>& signal_requested() noexcept {
    static std::atomic<bool> flag{false};
    return flag;
}
inline void handle_signal(int /*signum*/) noexcept {
    signal_requested().store(true, std::memory_order_release);
}
inline void install_signal_handlers() noexcept {
    std::signal(SIGINT, &handle_signal);
    std::signal(SIGTERM, &handle_signal);
}

// Resolve the "init joint position" the state machine ramps toward during
// MOVING_TO_DEFAULT. Matches Python G1Controller.__init__:
//   - if motion is loaded: motion frame 0 (training order) remapped to
//     deployment order via train_to_deploy;
//   - else fall back to cfg.default_angles (deployment order).
inline Eigen::Matrix<float, static_cast<Eigen::Index>(kDof), 1>
resolve_init_joint_pos(const Config&                            cfg,
                        const policy::JointRemap<kDof>&          remap,
                        const policy::MotionData*                motion) noexcept {
    if (motion != nullptr && motion->num_frames() > 0) {
        Eigen::Matrix<float, static_cast<Eigen::Index>(kDof), 1> deploy;
        for (std::size_t i = 0; i < kDof; ++i) {
            deploy[static_cast<Eigen::Index>(i)]
                = motion->dof_poss(0, static_cast<Eigen::Index>(remap.train_to_deploy[i]));
        }
        return deploy;
    }
    return cfg.default_angles;
}

// Minimum length (output frames) to pad a loaded/standing motion to: the run
// duration plus 5 s of slack, but at most a 600 s horizon. Every reader clamps
// its index to the last frame, and the padding rows are all the same held
// pose, so a longer pad changes nothing except memory: the 60000 s default
// --duration would otherwise materialise ~3M frames (~850 MB) per motion.
[[nodiscard]] inline std::int32_t motion_pad_frames(float duration_s, int output_fps) noexcept {
    const float horizon_s = std::min(duration_s, 600.0F) + 5.0F;
    return static_cast<std::int32_t>(horizon_s * static_cast<float>(output_fps));
}

// Which inference backend to prefer given the CLI choice (validated by
// parse_common_args). Defined in core/cli_args.hpp, next to the CLI value parsers.
using core::PreferredEngine;
using core::parse_engine_choice;

// Mirror Python g1_controller.create_motion_loader path resolution:
//   - empty string  → empty (loader builds standing pose)
//   - ends with .csv → returned verbatim
//   - otherwise     → walk up from `config_path` looking for a `data/` dir,
//                     then return `<data_dir>/<name>.csv`. Matches Python's
//                     `Path(__file__).parent.parent.parent / 'data' / f'{name}.csv'`.
inline std::string resolve_motion_path(const std::string& motion,
                                         const std::string& config_path) {
    if (motion.empty()) return motion;
    if (motion.size() >= 4 && motion.compare(motion.size() - 4, 4, ".csv") == 0) {
        return motion;
    }
    namespace fs = std::filesystem;
    fs::path probe;
    if (!config_path.empty() && fs::exists(config_path)) {
        probe = fs::weakly_canonical(fs::path{config_path}).parent_path();
    } else {
        probe = fs::current_path();
    }
    while (!probe.empty() && probe != probe.root_path()) {
        if (fs::exists(probe / "data")) {
            return (probe / "data" / (motion + ".csv")).string();
        }
        probe = probe.parent_path();
    }
    // Couldn't anchor — return as-is and let the loader's open() fail loudly.
    return motion;
}

// Wire deps, run the control loop, print summary. Returns the process exit code.
// `initial_state` selects the state machine's starting state — defaults to
// kInit (operator-gated, real-robot flow). Sim apps that have no remote should
// pass kControl so the policy runs immediately.
//
// `motion` is taken by non-const pointer so the control loop can yaw-align it
// on the kDefaultPos → kControl transition. Pass nullptr to disable alignment.
//
// `motion_switcher` is the operator-driven swap controller (B → standup,
// A → replay original). Pass nullptr to disable swaps; the loop then aligns
// `motion` once at CONTROL entry. When non-null the switcher's `original()`
// should equal `motion` so CONTROL-entry alignment lands on the right buffer.
//
// `state_outbox` / `motion_inbox` are transport-neutral hooks. Any transport
// (network node, in-process driver) that wants to read state or
// push a new motion attaches here. When `motion_inbox` is non-null,
// `motion_scratch_capacity` must be > kMaxTransitionSteps — the control loop
// allocates its internal MotionData scratch from that capacity in its ctor.
// Pass nullptrs (and 0 capacity) for a transport-free run.
//
// Backend and engine are passed by reference — the caller retains ownership
// for the duration of the call. This lets platforms that need to keep the
// backend alive after the control loop returns (macOS, where the viewer
// lives on the main thread and must out-live the worker control loop) do so
// without fighting the signature.
inline int run_control_loop(const CommonArgs&                      args,
                              const Config&                          cfg,
                              const policy::JointRemap<kDof>&        remap,
                              policy::MotionData*                    motion,
                              robot::IRobotBackend<kDof>&            backend,
                              inference::IInferenceEngine&           engine,
                              math::ControlState                     initial_state
                                  = math::ControlState::kInit,
                              MotionSwitcher*                        motion_switcher
                                  = nullptr,
                              core::TripleBuffer<math::RobotState<kDof>>*
                                  state_outbox = nullptr,
                              core::TripleBuffer<policy::RawMotionPlan>*
                                  motion_inbox = nullptr,
                              std::int32_t
                                  motion_scratch_capacity = 0,
                              // ── Streaming transport seam ─────────────────
                              // All optional + default-null; deploy_mujoco.cpp
                              // and deploy_real.cpp leave them unset. The
                              // system/cpp tracker nodes wire them to drive
                              // trim + active-plan-snapshot publishing through
                              // the TripleBuffer/atomic seam.
                              std::atomic<std::int32_t>*
                                  inference_tick_out = nullptr,
                              core::TripleBuffer<ActivePlanSnapshot>*
                                  active_plan_snapshot_out = nullptr,
                              std::atomic<bool>*
                                  trim_install_pending = nullptr,
                              std::atomic<std::int32_t>*
                                  trim_install_tick = nullptr,
                              // Optional latched operator-ready output. When
                              // non-null, the control loop sets it true when
                              // the operator presses B in CONTROL with no
                              // original motion (multi-process mode). Read by
                              // the transport.
                              std::atomic<bool>*
                                  tracker_ready_out = nullptr,
                              // Out-param: set to the constructed ControlLoop (for
                              // request_stop() and status reads). The active motion
                              // is owned and rewritten by the control thread: read
                              // the executed reference through ref_outbox instead.
                              // Stays the caller's initial value (nullptr) until
                              // the loop attaches here.
                              ControlLoop**
                                  control_loop_out = nullptr,
                              // Fired AFTER loop.run() returns but BEFORE the
                              // loop's destructor — i.e. while the loop is
                              // alive but quiescent. Transports that
                              // dereference `control_loop_out` from another
                              // thread MUST stop here (e.g. server.stop()) so
                              // they can't read the pointer once the loop's
                              // stack frame goes away.
                              std::function<void()>
                                  on_loop_exit = {},
                              // Optional latest-depth inbox for the depth
                              // student's `student_depth` image obs group.
                              // Filled by the depth producer (sim render / real
                              // RealSense); the loop drains it wait-free.
                              // nullptr => no depth: a config with an image
                              // group is refused at start (unless dry run).
                              core::TripleBuffer<DepthFrame>*
                                  depth_inbox = nullptr,
                              // Optional gamepad mirror (ControlLoopDeps::remote_buttons_out).
                              std::atomic<std::uint32_t>*
                                  remote_buttons_out = nullptr,
                              // Optional per-tick reference outbox
                              // (ControlLoopDeps::ref_outbox).
                              core::TripleBuffer<ExecutedRefEntry>*
                                  ref_outbox = nullptr) {
    if (args.lock_memory) {
        if (const int rc = core::lock_process_memory(); rc != 0) {
            std::fprintf(stderr, "[warn] mlockall failed (errno=%d); continuing\n", rc);
        }
    }

    // Profiling — must be enabled BEFORE any RateProbe is constructed in
    // child threads, so they pick up the global flag at probe creation time.
    core::set_profiling_enabled(args.profile);
    core::RateMonitor rate_monitor;  // RAII; no-op when --profile not set

    // Observation layout is fully declared in the config's `observation:` block
    // (Python parity). One config-driven builder handles the proprio and depth
    // students: group names map to the ONNX input names.
    auto obs_spec_r = g1::load_observation_spec(args.config_path);
    if (!obs_spec_r) {
        std::fprintf(stderr, "[error] observation spec: %s\n", obs_spec_r.error().c_str());
        return 1;
    }
    // The depth encode clip is a training contract, and getting it wrong is
    // invisible downstream: shape, num_obs and the ONNX signature all still match
    // while every pixel is mis-scaled. So the config's declared clip is checked
    // against the training clip, and a mismatch REFUSES rather than warns.
    // --allow_depth_clip_mismatch is the escape hatch for a checkpoint trained
    // with a different clip.
    if (const std::string problem
            = g1::check_depth_clip_matches_checkpoint(obs_spec_r.value(), args.policy_path);
        !problem.empty()) {
        if (args.allow_depth_clip_mismatch) {
            std::fprintf(stderr, "[rt_deploy] WARNING: %s\n"
                                  "               Proceeding under --allow_depth_clip_mismatch.\n",
                         problem.c_str());
        } else {
            std::fprintf(stderr, "[error] depth clip: %s\n", problem.c_str());
            return 1;
        }
    }
    ObservationBuilder obs_builder{cfg, obs_spec_r.value(), remap, motion};
    ActionDecoder      decoder{cfg, remap};

    StateMachineConfig sm_cfg;
    sm_cfg.initial_state          = initial_state;
    const auto                init_pos = resolve_init_joint_pos(cfg, remap, motion);
    StateMachine              state_machine{cfg, sm_cfg, init_pos};

    robot::SafetyConfig<kDof>   safety_cfg;
    // Stale window = 3 × control period.
    const auto control_period_ns
        = static_cast<std::int64_t>(1'000'000'000 / args.control_hz);
    safety_cfg.stale_ns         = control_period_ns * 3;
    robot::SafetyMonitor<kDof>  safety{safety_cfg};

    core::JitterHistogram       jitter;

    ControlLoopConfig clcfg;
    clcfg.control_period         = std::chrono::nanoseconds{control_period_ns};
    clcfg.state_ready_timeout_ns = 10'000'000'000;  // 10 s
    clcfg.rt_thread_config.fifo_priority = args.rt_fifo_priority;
    clcfg.rt_thread_config.cpu_affinity  = args.rt_cpu_affinity;
    clcfg.rt_thread_config.name          = "g1-control";
    clcfg.clip_observation       = cfg.clip_observation;
    clcfg.clip_action            = cfg.clip_action;
    // Rounded, at least one tick: truncating a sub-second --duration to 0 would
    // mean "unbounded" (max_ticks == 0).
    clcfg.max_ticks = std::max<std::int64_t>(
        1, std::llround(static_cast<double>(args.duration_s)
                        * static_cast<double>(args.control_hz)));
    clcfg.motion_scratch_capacity = motion_scratch_capacity;
    clcfg.align_motion_to_robot   = args.align_motion;
    clcfg.sim_pause_after_plan    = std::chrono::nanoseconds{
        static_cast<std::int64_t>(args.sim_pause_after_plan_s * 1.0e9F)};
    clcfg.depth_stats_every       = static_cast<std::int32_t>(args.depth_stats_every);
    if (args.depth_stale_far_ticks >= 0) {  // -1 = keep the built-in default
        clcfg.depth_stale_far_ticks = static_cast<std::int32_t>(args.depth_stale_far_ticks);
    }
    clcfg.dry_run                 = args.dry_run;

    ControlLoopDeps deps{backend, engine, obs_builder, decoder,
                          state_machine, safety, jitter,
                          /*mutable_motion=*/motion,
                          /*motion_switcher=*/motion_switcher,
                          /*state_outbox=*/state_outbox,
                          /*motion_inbox=*/motion_inbox,
                          /*inference_tick_out=*/inference_tick_out,
                          /*active_plan_snapshot_out=*/active_plan_snapshot_out,
                          /*trim_install_pending=*/trim_install_pending,
                          /*trim_install_tick=*/trim_install_tick};
    deps.tracker_ready_out  = tracker_ready_out;
    deps.depth_inbox        = depth_inbox;
    deps.remote_buttons_out = remote_buttons_out;
    deps.ref_outbox         = ref_outbox;

    // Per-iteration loop-timing trace (--trace_csv). Created HERE, on the
    // caller's (non-RT) thread, so its writer thread inherits SCHED_OTHER and
    // can never preempt the control thread — which calls
    // configure_current_thread() only once loop.run() starts.
    std::unique_ptr<core::LoopTrace> loop_trace;
    if (!args.trace_csv.empty()) {
        const core::LoopTrace::MetaList meta{
            {"impl",                 "cpp"},
            {"robot",                cfg.robot_name},
            {"engine",               std::string{engine.backend_name()}},
            {"config",               args.config_path},
            {"policy",               args.policy_path},
            {"motion",               args.motion_path},
            {"control_hz",           std::to_string(args.control_hz)},
            {"rt_fifo_priority",     std::to_string(args.rt_fifo_priority)},
            {"rt_cpu_affinity",      std::to_string(args.rt_cpu_affinity)},
            // Whether a depth source was attached for THIS run, recorded in the
            // trace itself: comparing a depth-on and a depth-off trace is only
            // meaningful if the two runs really differ in this one variable, and
            // a header line answers it without parsing every row.
            {"depth_inbox",          depth_inbox != nullptr ? "1" : "0"},
            // WHICH depth estimator, not merely whether one ran: two runs with
            // different stereo plans are otherwise identical in every header
            // field. Empty when no depth ran, so two depth-off runs compare equal.
            {"depth_source",         args.depth_source_name},
            {"depth_plan",           args.depth_plan_path},
            // A trace recorded with the depth_stats dump running is NOT a clean
            // jitter baseline — the dump fprintf()s from the RT thread.
            {"depth_stats_every",    std::to_string(args.depth_stats_every)},
        };
        loop_trace = core::LoopTrace::create(args.trace_csv, meta);
        if (!loop_trace) {
            std::fprintf(stderr, "[error] --trace_csv: cannot open %s for writing\n",
                         args.trace_csv.c_str());
            return 1;
        }
        deps.loop_trace = loop_trace.get();
        std::fprintf(stderr, "[rt_deploy] loop trace → %s\n", args.trace_csv.c_str());
    }

    ControlLoop     loop{clcfg, deps};
    // Publish the loop pointer so a transport (running on its own thread,
    // started before this call) can call request_stop() / current_tick(). The
    // executed reference is read through ref_outbox: active_motion() is owned and
    // rewritten by the control thread. The loop is fully constructed here.
    if (control_loop_out != nullptr) {
        *control_loop_out = &loop;
    }

    std::fprintf(stderr, "[rt_deploy] engine=%s  control_hz=%d (== inference)  ticks=%lld\n",
                 std::string{engine.backend_name()}.c_str(),
                 args.control_hz,
                 static_cast<long long>(clcfg.max_ticks));
    for (std::size_t k = 0; k < engine.input_count(); ++k) {
        const auto in_shape = engine.input_shape_at(k);
        const auto in_name  = engine.input_name_at(k);
        std::fprintf(stderr, "[rt_deploy] policy input %zu '%.*s': [", k,
                     static_cast<int>(in_name.size()), in_name.data());
        for (std::size_t i = 0; i < in_shape.dims.size(); ++i) {
            std::fprintf(stderr, "%s%d", i == 0 ? "" : ",",
                        static_cast<int>(in_shape.dims[i]));
        }
        std::fprintf(stderr, "]  elements: %zu\n", in_shape.elements());
    }
    if (initial_state == math::ControlState::kInit) {
        std::fprintf(stderr, "[rt_deploy] press Start to move to default, then A to begin.\n");
    } else {
        std::fprintf(stderr, "[rt_deploy] starting in %s state.\n",
                     initial_state == math::ControlState::kControl ? "CONTROL" : "non-INIT");
    }

    install_signal_handlers();

    // Poll the signal flag from a watcher thread that forwards to the loop.
    std::atomic<bool> watcher_done{false};
    std::thread       watcher([&]() {
        static core::RateProbe probe{"watcher"};
        while (!watcher_done.load(std::memory_order_acquire)) {
            probe.tick();
            {
                auto scope = probe.scope();
                if (signal_requested().load(std::memory_order_acquire)) {
                    loop.request_stop();
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
        }
    });

    const ControlLoopExit exit_code = loop.run();
    watcher_done.store(true, std::memory_order_release);
    if (watcher.joinable()) watcher.join();

    // Loop is now quiescent but still alive. Stop transports that read
    // `control_loop_out` BEFORE the loop's destructor runs and its stack frame
    // is reused; otherwise a transport thread still holding the loop pointer
    // races the destruction and dereferences a stale stack address.
    if (on_loop_exit) on_loop_exit();
    if (control_loop_out != nullptr) *control_loop_out = nullptr;

    // Damping on exit — backend-specific cleanup publishes the final command.
    backend.cleanup();

    const auto summary = jitter.summarize();
    std::fprintf(stderr,
                 "[rt_deploy] exit=%u  ticks=%lld  jitter: min=%lldns max=%lldns "
                 "p50=%lldus p99=%lldus p999=%lldus\n",
                 static_cast<unsigned>(exit_code),
                 static_cast<long long>(loop.tick_count()),
                 static_cast<long long>(summary.min_ns),
                 static_cast<long long>(summary.max_ns),
                 static_cast<long long>(summary.p50_us),
                 static_cast<long long>(summary.p99_us),
                 static_cast<long long>(summary.p999_us));
    if (safety.is_latched()) {
        std::fprintf(stderr, "[rt_deploy] safety fault: %s\n",
                     std::string{robot::to_string(safety.fault())}.c_str());
    }

    // Drain + close the loop trace. The control loop is quiescent by now, so
    // every recorded sample makes it to disk. Non-zero `dropped` would mean the
    // writer thread was starved — the trace is still valid, just gappy.
    if (loop_trace) {
        loop_trace->close();
        std::fprintf(stderr,
                     "[rt_deploy] loop trace: %llu rows → %s (dropped=%llu)\n",
                     static_cast<unsigned long long>(loop_trace->written()),
                     loop_trace->path().c_str(),
                     static_cast<unsigned long long>(loop_trace->dropped()));
    }

    // Final per-thread profile summary (no-op if --profile wasn't set).
    core::RateMonitor::print_final_summary();

    if (exit_code == ControlLoopExit::kConfigError) {
        // The `exit=` line above prints the ControlLoopExit value (6); the
        // PROCESS exit code is 4. Not to be confused with `exit=4`
        // (kRequested), which is a clean stop and returns 0.
        std::fprintf(stderr,
                     "[rt_deploy] config error: obs spec does not match the policy's inputs; "
                     "the control loop never ticked (process exit code 4)\n");
    }

    switch (exit_code) {
        case ControlLoopExit::kOdomTimeout:   return 1;
        case ControlLoopExit::kSafetyLatched: return 2;
        case ControlLoopExit::kStateTimeout:  return 3;
        // Misconfiguration, not a run-time fault — must not look like success to
        // a launch script.
        case ControlLoopExit::kConfigError:   return 4;
        case ControlLoopExit::kEngineFailure: return 5;
        default:                               return 0;
    }
}

}  // namespace rt_deploy::g1
