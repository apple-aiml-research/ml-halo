//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// depth_setup — the APP-side half of the depth-student `student_depth` path.
//
// Everything the control loop does with depth lives in the library
// (src/g1/control_loop.cpp). What an application has to do is small, but every
// step of it is a refusal that stops a robot from running blind or on
// mis-scaled depth:
//
//   * parse --depth_config / --depth_source / --depth_plan, and refuse the
//     combinations that would silently do nothing,
//   * refuse a depth config against a policy config that has no image group
//     (nothing would consume the frames),
//   * cross-check the depth YAML's clip against the observation group's clip —
//     the same number lives in two files, and a mismatch is invisible
//     downstream,
//   * refuse depth threads at or above the control loop's RT priority,
//   * wait for the first frame, so a depth policy never starts blind,
//   * own the inbox so it outlives the producer.
//
// Used by apps/g1/deploy_real.cpp; kept in its own header so any other
// real-robot binary that drives a depth policy can share the same checks.
//
// Without --depth_config every function here is inert: no camera is opened, no
// thread starts, the inbox pointer stays null, and a proprioceptive-only policy
// runs exactly as it would in a build without depth support.

#include "apps/g1/common.hpp"

#include "rt_deploy/core/cli_args.hpp"
#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/g1/depth_frame.hpp"
#include "rt_deploy/g1/observation_builder.hpp"

#if defined(RT_DEPLOY_WITH_REALSENSE)
#    include "rt_deploy/vision/config.hpp"
#    include "rt_deploy/vision/depth_source.hpp"
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace rt_deploy::g1 {

// The three depth flags, in one place so every app spells them the same way.
struct DepthArgs {
    // --depth_config: the depth producer's YAML. Empty => depth is off, which is
    // the default and the only state a proprioceptive-only policy ever sees.
    std::string config;
    // --depth_source: override the YAML's `depth_source` (raw|cleaned), so a
    // raw-vs-cleaned A/B needs neither a rebuild nor an edited config.
    std::string source;
    // --depth_plan: override the YAML's stereo `plan_path` (cleaned only), so an
    // A/B between two stereo plans is likewise a flag. Taken literally — the
    // {ROOT_DIR} placeholder is a YAML-only convenience.
    std::string plan;

    [[nodiscard]] bool enabled() const noexcept { return !config.empty(); }
};

// One line of --help, shared so the two binaries cannot drift. A plain
// `const char*` because parse_common_args takes one; callers that want to
// concatenate should build a named std::string and pass its c_str().
inline constexpr const char* kDepthUsage =
    "[--depth_config YAML] [--depth_source cleaned|raw] [--depth_plan PLAN]";

// Parse one depth flag out of an app's leftover-argument vector.
//
// Returns 1 when `rest[i]` was a depth flag (and advances `i` past its value),
// 0 when it belongs to the caller, -1 on a malformed depth flag. Written as a
// hook rather than a whole parser because each app owns its own extra flags
// (--net, --bind_ip, ports, …) and only wants depth folded into that loop.
[[nodiscard]] inline int parse_depth_flag(const std::vector<std::string>& rest,
                                          std::size_t& i, DepthArgs& out) {
    const std::string_view a{rest[i]};
    // core::take_value refuses a value that is itself a flag, so
    // `--depth_config --no_send` is an error instead of a config named "--no_send".
    const auto             take = [&]() -> const std::string* {
        return core::take_value(rest, i);
    };
    if (a == "--depth_config") {
        const auto* v = take();
        if (v == nullptr) return -1;
        out.config = *v;
        return 1;
    }
    if (a == "--depth_plan") {
        const auto* v = take();
        if (v == nullptr) return -1;
        out.plan = *v;
        return 1;
    }
    if (a == "--depth_source") {
        const auto* v = take();
        if (v == nullptr) return -1;
        out.source = *v;
        if (out.source != "raw" && out.source != "cleaned") {
            std::fprintf(stderr, "error: --depth_source must be 'raw' or 'cleaned', got '%s'\n",
                         out.source.c_str());
            return -1;
        }
        return 1;
    }
    return 0;
}

// Cross-flag checks, run once after the whole argument vector is parsed.
//
// Both refusals exist because the alternative is an operator believing a run was
// configured a way it was not: an override with no producer is inert, and a
// stereo plan means nothing to the camera's own on-chip block matcher.
[[nodiscard]] inline bool validate_depth_args(const DepthArgs& a) {
    if (!a.enabled() && (!a.plan.empty() || !a.source.empty())) {
        std::fprintf(stderr,
                     "error: --depth_plan/--depth_source require --depth_config; without it no "
                     "depth producer is started and the override would do nothing.\n");
        return false;
    }
    if (!a.plan.empty() && a.source == "raw") {
        std::fprintf(stderr,
                     "error: --depth_plan is meaningless with --depth_source raw (the raw path is "
                     "the camera's own on-chip block matcher and loads no TensorRT plan).\n");
        return false;
    }
    return true;
}

// Owns everything the depth path needs for the lifetime of a run.
//
// MEMBER ORDER IS LOAD-BEARING. `inbox` is declared before `source`, so it is
// destroyed AFTER it — which is what makes it safe to let the producer be torn
// down by this struct's destructor (after run_control_loop has damped) instead
// of stopping it on the way out. Declare one of these in main's frame.
struct DepthRuntime {
    core::TripleBuffer<DepthFrame>  inbox;
    // Handed to run_control_loop. Stays null unless a producer actually started,
    // which is what tells the control loop "no depth" versus "depth".
    core::TripleBuffer<DepthFrame>* inbox_ptr = nullptr;
#if defined(RT_DEPLOY_WITH_REALSENSE)
    std::unique_ptr<vision::IDepthSource> source;
#endif

    // A summary callback for run_control_loop's `on_loop_exit`, or an empty
    // std::function when depth is off.
    //
    // Deliberately reports and does NOT stop the producer. on_loop_exit fires
    // between loop.run() returning and backend.cleanup() publishing the damping
    // command, so anything slow in it delays the robot reaching a safe state —
    // and stopping a depth source joins two threads that can each be parked in a
    // 1 s librealsense wait_for_frames timeout. A publish into an inbox nobody
    // reads is a no-op, so leaving the producer running until the destructor is
    // safe as well as faster.
    [[nodiscard]] std::function<void()> stats_reporter() {
#if defined(RT_DEPLOY_WITH_REALSENSE)
        if (source) {
            return [this]() {
                const auto s = source->stats();
                std::fprintf(stderr,
                             "[rt_deploy] depth: captured=%llu published=%llu dropped=%llu "
                             "overruns=%llu errors=%llu last=%.1fms rate=%.1fHz\n",
                             static_cast<unsigned long long>(s.captured),
                             static_cast<unsigned long long>(s.published),
                             static_cast<unsigned long long>(s.dropped),
                             static_cast<unsigned long long>(s.overruns),
                             static_cast<unsigned long long>(s.errors),
                             static_cast<double>(s.last_ms), static_cast<double>(s.ema_hz));
            };
        }
#endif
        return {};
    }
};

// Start the depth producer described by `da`, or do nothing when depth is off.
//
// Returns 0 on success (with rt.inbox_ptr set iff a producer is running) and 1
// when the app should exit. A depth failure is FATAL at startup on purpose: a
// depth policy running on the blank far-plane fallback is driving blind, and
// refusing before the robot moves beats discovering it mid-motion. After start()
// the same failures only degrade freshness — see control_loop.cpp's staleness
// handling.
//
// `args` is written, not just read: `depth_source_name` / `depth_plan_path` are
// recorded post-override so the trace header says what actually ran rather than
// what the YAML asked for.
[[nodiscard]] inline int setup_depth(CommonArgs& args, const DepthArgs& da, DepthRuntime& rt) {
    if (!da.enabled()) {
        return 0;
    }
#if defined(RT_DEPLOY_WITH_REALSENSE)
    auto vcfg_r = vision::load_vision_config(da.config);
    if (!vcfg_r) {
        std::fprintf(stderr, "[error] depth config: %s\n", vcfg_r.error().c_str());
        return 1;
    }
    auto vcfg = std::move(vcfg_r.value());
    if (!da.source.empty()) {
        vcfg.source = (da.source == "raw") ? vision::DepthSource::kRaw
                                            : vision::DepthSource::kCleaned;
        // The IR projector goes with the source (see emitter: in
        // depth_d435i.yaml): the learned matcher needs it OFF, the on-chip block
        // matcher wants its texture ON. An override that kept the YAML's value
        // would run the A/B with the wrong projector setting on one side.
        vcfg.emitter = (vcfg.source == vision::DepthSource::kRaw);
        std::fprintf(stderr, "[rt_deploy] --depth_source %s: IR emitter %s\n",
                     da.source.c_str(), vcfg.emitter ? "ON" : "OFF");
    }
    if (!da.plan.empty()) {
        if (vcfg.source != vision::DepthSource::kCleaned) {
            std::fprintf(stderr,
                         "[error] --depth_plan given but the effective depth_source is 'raw' "
                         "(from %s); the raw path loads no plan\n",
                         da.config.c_str());
            return 1;
        }
        vcfg.plan_path = da.plan;
    }
    // Record what will actually run, post-override, for the trace header.
    args.depth_source_name = std::string{vision::to_string(vcfg.source)};
    args.depth_plan_path   = (vcfg.source == vision::DepthSource::kCleaned) ? vcfg.plan_path
                                                                            : std::string{};

    // ── The depth clip has ONE source of truth: the observation spec ──────
    // The source clamps metric depth to [clip_min_m, clip_max_m] and the obs
    // image group then re-clips to its own [clip_min, clip_max]. The same number
    // lives in two files, and a mismatch fails silently.
    //
    // It is not merely redundant, it is lossy in one direction: a source near
    // clamp ABOVE the group's destroys real geometry the policy expects to see.
    // Clamping to 0.3 when the group encodes from 0.2 flattens everything in
    // [0.2, 0.3) m onto 0.3 — the closest obstacles, where being wrong matters
    // most — and no later stage can recover it.
    //
    // So the group's clip WINS, and a depth YAML that disagrees is a hard error
    // rather than a silent override: if the two files disagree, one of them is
    // stale and the operator should know which.
    auto spec_r = load_observation_spec(args.config_path);
    if (!spec_r) {
        std::fprintf(stderr, "[error] observation spec: %s\n", spec_r.error().c_str());
        return 1;
    }
    const GroupCfg* img = nullptr;
    for (const auto& g : spec_r.value().groups) {
        if (g.is_image) {
            img = &g;
            break;
        }
    }
    if (img == nullptr) {
        std::fprintf(stderr,
                     "[error] --depth_config given but %s declares no image observation "
                     "group, so nothing would consume the depth frames. Use a depth-student "
                     "config, or drop --depth_config.\n",
                     args.config_path.c_str());
        return 1;
    }
    const int  depth_obs_decim = img->obs_decim;
    const auto differs = [](float a, float b) { return std::fabs(a - b) >= 1.0e-4F; };
    if (differs(vcfg.clip_min_m, img->clip_min) || differs(vcfg.clip_max_m, img->clip_max)) {
        std::fprintf(stderr,
                     "[error] depth clip disagreement between two config files:\n"
                     "          %s  observation.groups.%s.clip = [%.4g, %.4g] m\n"
                     "          %s  clip_min_m/clip_max_m      = [%.4g, %.4g] m\n"
                     "        These must be the same number. The observation group is the "
                     "policy's contract,\n"
                     "        so fix the depth config to match it (a source near-clip above "
                     "the group's\n"
                     "        silently discards the closest obstacles).\n",
                     args.config_path.c_str(), img->name.c_str(),
                     static_cast<double>(img->clip_min), static_cast<double>(img->clip_max),
                     da.config.c_str(),
                     static_cast<double>(vcfg.clip_min_m),
                     static_cast<double>(vcfg.clip_max_m));
        return 1;
    }

    // The depth threads exist to feed an observation; they must never be able to
    // preempt the loop that keeps the robot standing. VisionConfig::validate()
    // can only range-check its own priorities — it cannot see the control loop's.
    // This is the one place both numbers are known, so it is the one place the
    // ordering can actually be enforced.
    const auto rt_below_loop = [&args](const vision::ThreadRtConfig& trt, const char* which) {
        if (trt.fifo_priority <= 0) {
            return true;  // SCHED_OTHER — below every SCHED_FIFO thread
        }
        if (args.rt_fifo_priority <= 0 || trt.fifo_priority >= args.rt_fifo_priority) {
            std::fprintf(stderr,
                         "[error] depth config: %s.fifo_priority=%d would run at or above the "
                         "control loop (--rt_prio %d); depth threads must sit "
                         "strictly below it\n",
                         which, trt.fifo_priority, args.rt_fifo_priority);
            return false;
        }
        return true;
    };
    if (!rt_below_loop(vcfg.capture_rt, "capture_rt")
        || !rt_below_loop(vcfg.inference_rt, "inference_rt")) {
        return 1;
    }
    // Same for the core: a depth thread pinned to the control loop's CPU
    // competes with it for every cycle (vision/config.hpp: the depth threads must
    // stay off that core). Only the capture thread exists on the raw path.
    const auto off_loop_cpu = [&args](const vision::ThreadRtConfig& trt, const char* which) {
        if (args.rt_cpu_affinity >= 0 && trt.cpu_affinity == args.rt_cpu_affinity) {
            std::fprintf(stderr,
                         "[error] depth config: %s.cpu_affinity=%d is the control loop's core "
                         "(--rt_cpu %d); pin the depth threads to another core\n",
                         which, trt.cpu_affinity, args.rt_cpu_affinity);
            return false;
        }
        return true;
    };
    if (!off_loop_cpu(vcfg.capture_rt, "capture_rt")
        || (vcfg.source == vision::DepthSource::kCleaned
            && !off_loop_cpu(vcfg.inference_rt, "inference_rt"))) {
        return 1;
    }

    auto src_r = vision::make_depth_source(vcfg, rt.inbox);
    if (!src_r) {
        std::fprintf(stderr, "[error] depth source: %s\n", src_r.error().c_str());
        return 1;
    }
    rt.source = std::move(src_r.value());
    // start() returns once the producer threads run, not once frames flow. A depth
    // policy must not reach CONTROL on the far-plane fallback, so wait for the first
    // published frame (camera warm-up is a few seconds) and refuse if none arrives.
    constexpr auto kFirstFrameTimeout = std::chrono::seconds(15);
    const auto     deadline           = std::chrono::steady_clock::now() + kFirstFrameTimeout;
    while (rt.source->stats().published == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (rt.source->stats().published == 0) {
        std::fprintf(stderr,
                     "[error] depth source %s published no frame within %lld s; refusing to run "
                     "a depth policy blind (check the camera / USB link)\n",
                     std::string{rt.source->name()}.c_str(),
                     static_cast<long long>(kFirstFrameTimeout.count()));
        rt.source.reset();
        return 1;
    }
    rt.inbox_ptr = &rt.inbox;
    std::fprintf(stderr,
                 "[rt_deploy] depth source=%s  %dx%d@%dHz -> %dx%d  clip=[%.2f,%.2f]m\n",
                 std::string{rt.source->name()}.c_str(), vcfg.width, vcfg.height,
                 static_cast<int>(vcfg.rate_hz), vcfg.out_width, vcfg.out_height,
                 static_cast<double>(vcfg.clip_min_m), static_cast<double>(vcfg.clip_max_m));
    // The image group's obs_decim is a SIM-ONLY knob and is deliberately NOT
    // applied here. The camera is asynchronous and already slower than the 50 Hz
    // control loop, so the sensor's own rate is the hold; the correct frame is
    // the newest one, and holding on top of that would only add latency. Said out
    // loud because the config declares obs_decim: 2 and an operator reading it
    // could reasonably expect 25 Hz on the robot too.
    if (depth_obs_decim > 1) {
        std::fprintf(stderr,
                     "[rt_deploy] note: observation obs_decim=%d is a SIM-only hold and is not "
                     "applied on hardware — the camera runs at %d Hz asynchronously and the "
                     "loop always consumes the newest frame.\n",
                     depth_obs_decim, static_cast<int>(vcfg.rate_hz));
    }
    return 0;
#else
    (void)rt;
    (void)args;
    std::fprintf(stderr, "[error] --depth_config needs RT_DEPLOY_WITH_REALSENSE=ON "
                          "(rebuild with -DRT_DEPLOY_WITH_REALSENSE=ON)\n");
    return 1;
#endif
}

}  // namespace rt_deploy::g1
