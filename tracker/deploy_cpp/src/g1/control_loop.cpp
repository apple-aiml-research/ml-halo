//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/g1/control_loop.hpp"

#include "rt_deploy/core/rate_probe.hpp"
#include "rt_deploy/g1/send_ref_motion.hpp"
#include "rt_deploy/policy/joint_remap.hpp"
#include "rt_deploy/policy/motion_loader.hpp"
#include "rt_deploy/policy/motion_processing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <thread>

namespace rt_deploy::g1 {

namespace {

// In-place global clip. Works for both Eigen column vectors and matrices via
// cwiseMax/cwiseMin (operator[] isn't defined on matrices).
template <typename Derived>
void clip_inplace(Eigen::MatrixBase<Derived>& v, float bound) noexcept {
    v = v.cwiseMax(-bound).cwiseMin(bound);
}

// Human-readable state name for the diagnostic status line.
const char* state_name(math::ControlState s) noexcept {
    switch (s) {
        case math::ControlState::kInit:            return "INIT";
        case math::ControlState::kMovingToDefault: return "MOVING";
        case math::ControlState::kDefaultPos:      return "DEFAULT";
        case math::ControlState::kControl:         return "CONTROL";
        case math::ControlState::kStopping:        return "STOPPING";
    }
    return "?";
}

// Engine status as a name — an operator reading a diagnostic line should not
// need an enum table to find out that the policy returned kSizeMismatch.
const char* infer_status_name(inference::InferenceStatus s) noexcept {
    switch (s) {
        case inference::InferenceStatus::kOk:           return "kOk";
        case inference::InferenceStatus::kSizeMismatch: return "kSizeMismatch";
        case inference::InferenceStatus::kRuntimeError: return "kRuntimeError";
        case inference::InferenceStatus::kNotReady:     return "kNotReady";
    }
    return "?";
}

}  // namespace

ControlLoop::ControlLoop(const ControlLoopConfig& cfg, ControlLoopDeps deps) noexcept
    : cfg_{cfg}, deps_{deps} {
    // Pre-allocate the MotionData scratch the loop processes RawMotionPlan
    // payloads into. Required when motion_inbox is wired (deps_.motion_inbox
    // != nullptr); harmless to leave zero-capacity if no inbox is attached.
    if (cfg_.motion_scratch_capacity > 0) {
        // motion.output_fps must equal the inference rate (== the control rate)
        // so `inference_tick` directly indexes motion rows — mirrors the Python
        // motion_loader where output_fps = 1/control_dt and controller.counter
        // (inference count) is 1:1 with rows.
        const std::int64_t period_ns = cfg_.control_period.count();
        const std::int32_t out_fps = (period_ns > 0)
            ? static_cast<std::int32_t>(std::lround(1.0e9 / static_cast<double>(period_ns)))
            : 50;
        processed_motion_scratch_ = policy::make_motion_scratch(
            cfg_.motion_scratch_capacity, out_fps);
    }

    // Map each observation-spec group (== ONNX input name) to its engine input
    // index once, so the hot path feeds the engine by name, in the engine's
    // declared input order regardless of the order the groups are listed in.
    {
        const auto&       groups = deps_.observation_builder.spec().groups;
        const std::size_t nin    = deps_.engine.input_count();
        engine_inputs_.assign(nin, std::span<const float>{});
        group_engine_idx_.assign(groups.size(), 0);
        // Which group (if any) claims each engine input, so an unfed input or a
        // second group landing on the same input can be reported below instead of
        // going out as an empty span / silently overwriting the first group.
        std::vector<int> engine_input_owner(nin, -1);
        for (std::size_t gi = 0; gi < groups.size(); ++gi) {
            int e = -1;
            for (std::size_t i = 0; i < nin; ++i) {
                if (deps_.engine.input_name_at(i) == groups[gi].name) {
                    e = static_cast<int>(i);
                    break;
                }
            }
            // A group whose name matches no engine input would alias to index 0
            // and silently overwrite the real input-0 tensor. Name it loudly:
            // this is almost always a typo in the config's `observation.groups`
            // or an obs spec paired with the wrong exported policy.
            if (e < 0) {
                std::fprintf(stderr,
                             "[control_loop] FATAL: obs group '%s' matches no engine input; "
                             "engine expects [",
                             groups[gi].name.c_str());
                for (std::size_t i = 0; i < nin; ++i) {
                    std::fprintf(stderr, "%s%.*s", i != 0U ? ", " : "",
                                 static_cast<int>(deps_.engine.input_name_at(i).size()),
                                 deps_.engine.input_name_at(i).data());
                }
                std::fprintf(stderr, "]\n");
                config_error_ = true;
            } else {
                const auto ei = static_cast<std::size_t>(e);
                // Two groups on one input: the later one overwrites the earlier
                // every tick. Only reachable through a spec with duplicate group
                // names.
                if (engine_input_owner[ei] >= 0) {
                    std::fprintf(stderr,
                                 "[control_loop] FATAL: obs groups '%s' and '%s' both feed "
                                 "engine input %zu ('%.*s'); the engine has %zu input(s) for "
                                 "%zu group(s)\n",
                                 groups[static_cast<std::size_t>(engine_input_owner[ei])]
                                     .name.c_str(),
                                 groups[gi].name.c_str(), ei,
                                 static_cast<int>(deps_.engine.input_name_at(ei).size()),
                                 deps_.engine.input_name_at(ei).data(), nin, groups.size());
                    config_error_ = true;
                } else {
                    engine_input_owner[ei] = static_cast<int>(gi);
                }
                // Right name, wrong size: infer_multi() would return
                // kSizeMismatch every tick and the zeroed action would hold the
                // default pose forever (or, through an engine that does not check,
                // read past the tensor). Typical cause: an image group of 48x84
                // paired with a policy exported at another resolution. Engines that
                // report 0 do not know their size and are not checked.
                const std::size_t want = deps_.engine.input_size_at(ei);
                const std::size_t have = deps_.observation_builder.group_size(gi);
                if (want != 0 && have != want) {
                    std::fprintf(stderr,
                                 "[control_loop] FATAL: obs group '%s' produces %zu values but "
                                 "engine input %zu ('%.*s') expects %zu\n",
                                 groups[gi].name.c_str(), have, ei,
                                 static_cast<int>(deps_.engine.input_name_at(ei).size()),
                                 deps_.engine.input_name_at(ei).data(), want);
                    config_error_ = true;
                }
            }
            group_engine_idx_[gi] = (e < 0) ? 0 : e;
        }
        // An engine input no group fills stays a default-constructed empty span.
        // infer_multi() would then return kSizeMismatch every tick, the action
        // would be zeroed, and the robot would hold its default pose FOREVER with
        // no log line. Refuse to start instead.
        for (std::size_t i = 0; i < nin; ++i) {
            if (engine_input_owner[i] < 0) {
                std::fprintf(stderr,
                             "[control_loop] FATAL: engine input %zu ('%.*s') is fed by no "
                             "observation group; check the config's observation.groups "
                             "against the exported policy's input names\n",
                             i, static_cast<int>(deps_.engine.input_name_at(i).size()),
                             deps_.engine.input_name_at(i).data());
                config_error_ = true;
            }
        }
        // An image group with no depth inbox is the most dangerous
        // misconfiguration this loop can be handed. The group stays *valid* —
        // the builder fills it with the far-plane constant log1p(clip_max) — so
        // nothing downstream errors: the engine gets a correctly-shaped tensor
        // and the policy runs happily on an image that says "open space in every
        // direction, forever". A depth policy told the world is empty will walk
        // the robot into whatever is actually there, and the staleness detector
        // (tick step 1c) cannot help because it is skipped entirely when the
        // inbox is null. The usual cause is an app that does not wire a depth source.
        for (const auto& g : groups) {
            if (g.is_image && deps_.depth_inbox == nullptr && cfg_.dry_run) {
                // Measurement harness, nothing actuating: the blank far-plane
                // image is the intended no-depth timing baseline, not a hazard.
                std::fprintf(stderr,
                             "[control_loop] WARNING: obs group '%s' is an image group with no "
                             "depth inbox.\n"
                             "               Permitted because this is a DRY RUN (no command "
                             "reaches the actuators):\n"
                             "               the group is filled with the far-plane constant, which "
                             "is the correct\n"
                             "               no-depth BASELINE for a timing comparison. It would be "
                             "refused on a\n"
                             "               powered run.\n",
                             g.name.c_str());
            } else if (g.is_image && deps_.depth_inbox == nullptr) {
                std::fprintf(stderr,
                             "[control_loop] FATAL: observation group '%s' is an image group but "
                             "no depth inbox is attached — the policy would run on a blank "
                             "far-plane image with no depth at all. Attach a depth source "
                             "(ControlLoopDeps::depth_inbox), or use a config without an image "
                             "group.\n",
                             g.name.c_str());
                config_error_ = true;
            }
        }
        // An image group whose clip never got set encodes every pixel through
        // log1p of a nonsense bound. The YAML parser already refuses this, but a
        // spec built in code (an embedder) bypasses that, and the symptom
        // downstream is a plausible-looking tensor rather than an error. This is
        // the last choke point before the policy sees it.
        for (const auto& g : groups) {
            if (!g.is_image) {
                continue;
            }
            if (!(g.clip_min > 0.0F) || !(g.clip_max > g.clip_min)) {
                std::fprintf(stderr,
                             "[control_loop] FATAL: image group '%s' has an unset or invalid encode "
                             "clip [%g, %g].\n"
                             "               It must be the clip this policy's checkpoint trained "
                             "with — 0.2/2.5 for the\n"
                             "               HALO depth student. There is no "
                             "default: a wrong clip mis-scales\n"
                             "               every pixel while every shape still matches.\n",
                             g.name.c_str(), static_cast<double>(g.clip_min),
                             static_cast<double>(g.clip_max));
                config_error_ = true;
            }
        }
        // Remember WHICH group is the image group, so the opt-in depth_stats
        // dump can index observation_builder.outputs() directly (that vector is
        // built index-aligned with spec().groups). First image group wins; the
        // depth student's spec declares exactly one.
        for (std::size_t gi = 0; gi < groups.size(); ++gi) {
            if (groups[gi].is_image) {
                depth_group_idx_ = static_cast<int>(gi);
                break;
            }
        }
    }
}

ControlLoopExit ControlLoop::run() noexcept {
    // Refuse to drive the robot on an obs-spec/engine mismatch. The ctor already
    // printed which group or input is at fault; ticking anyway would zero the
    // action every cycle or feed the wrong tensor to an input.
    if (config_error_) {
        std::fprintf(stderr, "[control_loop] refusing to start: observation spec does not match "
                              "the engine's inputs (see FATAL lines above)\n");
        return ControlLoopExit::kConfigError;
    }

    (void)core::configure_current_thread(cfg_.rt_thread_config);

    if (!deps_.backend.wait_for_state(cfg_.state_ready_timeout_ns)) {
        return ControlLoopExit::kStateTimeout;
    }

    // No odometry, no anchor frame. Refuse before the first tick rather than
    // sitting in INIT with an entry gate that can never open.
    {
        const std::int64_t deadline =
            core::MonotonicClock::now().count() + cfg_.odom_ready_timeout_ns;
        bool live = false;
        while (true) {
            live = deps_.backend.read_state(state_) != robot::ReadStatus::kNoData
                   && state_.odom_valid;
            if (live || core::MonotonicClock::now().count() >= deadline) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
        }
        if (!live) {
            std::fprintf(stderr,
                         "[control_loop] FATAL: no live odometry within %.1f s of the first "
                         "state; is the sport-mode state estimator running?\n",
                         static_cast<double>(cfg_.odom_ready_timeout_ns) * 1.0e-9);
            return ControlLoopExit::kOdomTimeout;
        }
    }

    const auto exit_code = run_body();
    print_depth_summary();
    print_swap_summary();
    return exit_code;
}

// Depth freshness over the whole run, printed once at exit. This is where a
// degraded run becomes visible after the fact: the per-stall stderr lines scroll
// away, and `frames` vs the producer's own published count is the only way to see
// how many frames the loop never got to read.
void ControlLoop::print_depth_summary(std::FILE* out) const noexcept {
    if (deps_.depth_inbox == nullptr) {
        return;  // no depth configured: nothing to report, not even zeros
    }
    const double period_ms = static_cast<double>(cfg_.control_period.count()) * 1.0e-6;
    std::fprintf(out,
                 "[rt_deploy] depth: frames_consumed=%lld  longest_stale=%d ticks (%.0f ms)  "
                 "far_plane_fallbacks=%lld\n",
                 static_cast<long long>(depth_frames_seen_), depth_max_stale_ticks_,
                 static_cast<double>(depth_max_stale_ticks_) * period_ms,
                 static_cast<long long>(depth_far_fallback_events_));
    if (depth_far_fallback_events_ > 0) {
        // Not a footnote: for part of this run the policy was fed an empty world.
        std::fprintf(out,
                     "[rt_deploy] WARNING: depth went stale past %d ticks %lld time(s); the policy "
                     "ran on the FAR PLANE (no obstacles anywhere) for those stretches. Treat any "
                     "behaviour recorded during them as depth-blind.\n",
                     cfg_.depth_stale_far_ticks,
                     static_cast<long long>(depth_far_fallback_events_));
    }
}


void ControlLoop::print_swap_summary(std::FILE* out) const noexcept {
    const std::size_t n = swap_joint_mean_.size();
    if (n == 0) {
        return;
    }
    auto stats = [n](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        const float mean = std::accumulate(v.begin(), v.end(), 0.0F)
                           / static_cast<float>(n);
        auto pct = [&](float p) -> float {
            const auto idx = std::min(
                static_cast<std::size_t>(p * static_cast<float>(n - 1)), n - 1);
            return v[idx];
        };
        return std::array<float, 5>{v.front(), pct(0.50F), pct(0.99F),
                                    v.back(), mean};
    };
    const auto jmn = stats(swap_joint_mean_);
    const auto jl  = stats(swap_joint_l2_);
    const auto bx  = stats(swap_base_xy_);
    const auto bz  = stats(swap_base_z_);
    std::fprintf(out,
                 "[rt_deploy] swap_err summary (n=%zu):\n"
                 "  joint_mean min/p50/p99/max=%.4f/%.4f/%.4f/%.4f rad  mean=%.4f\n"
                 "  joint_l2   min/p50/p99/max=%.4f/%.4f/%.4f/%.4f rad  mean=%.4f\n"
                 "  base_xy    min/p50/p99/max=%.4f/%.4f/%.4f/%.4f m    mean=%.4f\n"
                 "  base_z     min/p50/p99/max=%.4f/%.4f/%.4f/%.4f m    mean=%.4f\n",
                 n,
                 static_cast<double>(jmn[0]), static_cast<double>(jmn[1]),
                 static_cast<double>(jmn[2]), static_cast<double>(jmn[3]),
                 static_cast<double>(jmn[4]),
                 static_cast<double>(jl[0]), static_cast<double>(jl[1]),
                 static_cast<double>(jl[2]), static_cast<double>(jl[3]),
                 static_cast<double>(jl[4]),
                 static_cast<double>(bx[0]), static_cast<double>(bx[1]),
                 static_cast<double>(bx[2]), static_cast<double>(bx[3]),
                 static_cast<double>(bx[4]),
                 static_cast<double>(bz[0]), static_cast<double>(bz[1]),
                 static_cast<double>(bz[2]), static_cast<double>(bz[3]),
                 static_cast<double>(bz[4]));
}

// depth_stats diagnostic: print the ACTUAL numbers crossing the engine boundary.
//
// Cost, per invocation (i.e. once every cfg_.depth_stats_every inference ticks):
//   - one bounded pass over the encoded group tensor  — 4032 floats for 48x84
//   - one bounded pass over the raw metric DepthFrame — 4032 floats
//   - one bounded pass over the action vector         — 29 floats
//   - exactly one std::fprintf (one write(2) on stderr)
//   - zero clock reads: the caller's tick timestamp is the age reference
// ~8 k float loads is on the order of 10 µs on an Orin NX core, which is 0.05%
// of a 20 ms control period — negligible at any sane N. The fprintf is the part
// that actually matters: stderr is unbuffered, so if it is a pipe with a slow
// reader, the write BLOCKS THE RT THREAD. That is why this is opt-in, defaults
// to off, and is documented as "N >= one second's worth of ticks, or redirect
// stderr to a file". Nothing here allocates, locks or touches a shared word.
void ControlLoop::print_depth_stats(std::int64_t               inference_tick,
                                    std::int64_t               tick_mono_ns,
                                    inference::InferenceStatus infer_status) const noexcept {
    // ── 1. the ENCODED group tensor — byte-for-byte what the engine was fed ──
    // Read from observation_builder.outputs() AFTER build() + clip_observations().
    // Post-encode is the whole point: the raw frame can be perfect while the
    // encode is wrong (millimetres read as metres, log1p applied twice, a clip
    // range that no longer matches training) and nothing else in the system would
    // show it — the shape is still right and the engine still returns kOk.
    const auto& outs   = deps_.observation_builder.outputs();
    const auto& groups = deps_.observation_builder.spec().groups;
    const auto  gidx   = static_cast<std::size_t>(depth_group_idx_ < 0 ? 0
                                                                       : depth_group_idx_);
    const bool  have_group = depth_group_idx_ >= 0 && gidx < outs.size()
                             && gidx < groups.size();

    int    enc_rows      = 0;
    int    enc_cols      = 0;
    int    enc_n         = 0;
    int    img_rows      = 0;
    int    img_cols      = 0;
    int    enc_nonfinite = 0;
    int    enc_at_near   = 0;
    int    enc_at_far    = 0;
    double enc_min       = 0.0;
    double enc_max       = 0.0;
    double enc_mean      = 0.0;
    double clip_min      = 0.0;
    double clip_max      = 0.0;
    double near_bound    = 0.0;
    double far_bound     = 0.0;

    if (have_group) {
        const GroupTensor& g  = outs[gidx];
        const GroupCfg&    gc = groups[gidx];
        img_rows              = gc.img_rows;
        img_cols              = gc.img_cols;
        enc_rows              = g.rows;
        enc_cols              = g.cols;
        enc_n                 = static_cast<int>(g.data.size());
        // The builder encodes with std::log1p on the SAME clamped float, so these
        // are exact-equality tests, not tolerances: a pixel that hit the far plane
        // holds the identical bit pattern log1p(clip_max) produces here.
        const float near_f = std::log1p(gc.clip_min);
        const float far_f  = std::log1p(gc.clip_max);
        clip_min           = static_cast<double>(gc.clip_min);
        clip_max           = static_cast<double>(gc.clip_max);
        near_bound         = static_cast<double>(near_f);
        far_bound          = static_cast<double>(far_f);

        double sum = 0.0;
        float  mn  = std::numeric_limits<float>::infinity();
        float  mx  = -std::numeric_limits<float>::infinity();
        for (Eigen::Index i = 0; i < g.data.size(); ++i) {
            const float v = g.data[i];
            if (!std::isfinite(v)) {
                ++enc_nonfinite;
                continue;
            }
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += static_cast<double>(v);
            if (v == near_f) {
                ++enc_at_near;
            }
            if (v == far_f) {
                ++enc_at_far;
            }
        }
        const int finite_n = enc_n - enc_nonfinite;
        if (finite_n > 0) {
            enc_min  = static_cast<double>(mn);
            enc_max  = static_cast<double>(mx);
            enc_mean = sum / static_cast<double>(finite_n);
        }
    }
    const double far_frac =
        (enc_n > 0) ? (100.0 * static_cast<double>(enc_at_far) / static_cast<double>(enc_n))
                    : 0.0;

    // ── 2. the RAW metric frame the encode consumed — the units check ────────
    // metres in, log1p out. raw_max ≈ clip_max together with enc_max ==
    // log1p(clip_max) is the pair that proves the encode ran exactly once, in the
    // right direction. A raw_max of ~3000 would be millimetres; a raw_max of
    // ~1.386 would be an already-encoded frame being encoded again.
    int    raw_nonfinite = 0;
    int    raw_no_return = 0;  // the builder's invalid rule: <= 1e-6 m -> far
    double raw_min       = 0.0;
    double raw_max       = 0.0;
    double raw_mean      = 0.0;
    double raw_age_ms    = -1.0;
    if (last_depth_ != nullptr) {
        double sum = 0.0;
        float  mn  = std::numeric_limits<float>::infinity();
        float  mx  = -std::numeric_limits<float>::infinity();
        int    n   = 0;
        for (const float v : last_depth_->data) {
            if (!std::isfinite(v)) {
                ++raw_nonfinite;
                continue;
            }
            if (v <= 1.0e-6F) {
                ++raw_no_return;
            }
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += static_cast<double>(v);
            ++n;
        }
        if (n > 0) {
            raw_min  = static_cast<double>(mn);
            raw_max  = static_cast<double>(mx);
            raw_mean = sum / static_cast<double>(n);
        }
        if (last_depth_->stamp_ns > 0) {
            // Against the TICK's wake-up, not "now": no clock read of its own,
            // and every field in the line refers to the same instant.
            raw_age_ms =
                static_cast<double>(tick_mono_ns - last_depth_->stamp_ns) * 1.0e-6;
        }
    }

    // ── 3. the ACTION the engine returned — read BEFORE the clip ─────────────
    // Sampled at the call site, i.e. before clip_inplace(raw_action_train_,
    // clip_action), so a ±1e30 output is visible as ±1e30 instead of as a
    // perfectly plausible ±clip_action. NaN survives the clip either way (Eigen's
    // maxi/mini both return the first operand on an unordered compare), but a
    // huge finite value would not.
    int    act_nonfinite = 0;
    double act_min       = 0.0;
    double act_max       = 0.0;
    {
        float mn = std::numeric_limits<float>::infinity();
        float mx = -std::numeric_limits<float>::infinity();
        int   n  = 0;
        for (Eigen::Index i = 0; i < raw_action_train_.size(); ++i) {
            const float v = raw_action_train_[i];
            if (!std::isfinite(v)) {
                ++act_nonfinite;
                continue;
            }
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            ++n;
        }
        if (n > 0) {
            act_min = static_cast<double>(mn);
            act_max = static_cast<double>(mx);
        }
    }

    // Notes are string literals chosen by branch, never formatted — so the whole
    // report stays one fprintf with no intermediate buffer.
    const char* group_note =
        have_group ? "" : "   [NO IMAGE GROUP in the observation spec]";
    const char* far_note =
        (enc_n > 0 && enc_at_far == enc_n)
            ? "   [!! 100% FAR BOUND — depth source dead or blank]"
            : "";
    const char* raw_note =
        (last_depth_ == nullptr)
            ? "   [NO FRAME — image group on the far-plane fallback]"
            : "";
    const char* act_note = (act_nonfinite > 0) ? "   [!! NON-FINITE ACTION]" : "";

    std::fprintf(
        stderr,
        "[control_loop] depth_stats @ infer_tick=%lld  engine=%s\n"
        "  student_depth tensor  rows=%d cols=%d n=%d   (image %dx%d)%s\n"
        "    values             min=%.6f max=%.6f mean=%.6f nonfinite=%d\n"
        "    expected bounds    near=log1p(%.3f)=%.6f  far=log1p(%.3f)=%.6f\n"
        "    at bounds          at_near=%d at_far=%d  far_fraction=%.2f%%%s\n"
        "  raw DepthFrame (m)   seq=%lld stale=%d ticks  age=%.2f ms%s\n"
        "    values             min=%.4f max=%.4f mean=%.4f nonfinite=%d "
        "no_return=%d\n"
        "  action (train order) n=%d min=%.6f max=%.6f nonfinite=%d  "
        "clip=+/-%.3f%s\n",
        static_cast<long long>(inference_tick), infer_status_name(infer_status),
        enc_rows, enc_cols, enc_n, img_rows, img_cols, group_note,
        enc_min, enc_max, enc_mean, enc_nonfinite,
        clip_min, near_bound, clip_max, far_bound,
        enc_at_near, enc_at_far, far_frac, far_note,
        static_cast<long long>(last_depth_seq_), depth_stale_ticks_, raw_age_ms,
        raw_note,
        raw_min, raw_max, raw_mean, raw_nonfinite, raw_no_return,
        static_cast<int>(raw_action_train_.size()), act_min, act_max,
        act_nonfinite, static_cast<double>(cfg_.clip_action), act_note);
}

ControlLoopExit ControlLoop::run_body() noexcept {
    core::PeriodicTicker ticker{cfg_.control_period};
    static core::RateProbe probe{"control_loop"};

    // Per-phase probes — measure where time is spent inside each control
    // tick. Cost when --profile is off: one relaxed atomic load + branch.
    static core::RateProbe p_read_state {"phase.read_state"};
    static core::RateProbe p_poll_remote{"phase.poll_remote"};
    static core::RateProbe p_obs_build  {"phase.obs_build"};
    static core::RateProbe p_inference  {"phase.inference"};
    static core::RateProbe p_decode     {"phase.action_decode"};
    static core::RateProbe p_state_machine{"phase.state_machine"};
    static core::RateProbe p_safety     {"phase.safety"};
    static core::RateProbe p_send_command{"phase.send_command"};

    std::int64_t inference_tick = 0;
    inference_tick_.store(0, std::memory_order_relaxed);
    if (deps_.inference_tick_out != nullptr) {
        deps_.inference_tick_out->store(0, std::memory_order_release);
    }

    // Publish the current inference_tick into the private atomic and (if
    // attached) the caller-owned mirror, so off-loop readers (trim RPC) can
    // index the active motion without locks.
    auto publish_tick = [this, &inference_tick]() noexcept {
        const auto v = static_cast<std::int32_t>(inference_tick);
        inference_tick_.store(v, std::memory_order_release);
        if (deps_.inference_tick_out != nullptr) {
            deps_.inference_tick_out->store(v, std::memory_order_release);
        }
    };
    auto reset_tick = [&inference_tick, &publish_tick]() noexcept {
        inference_tick = 0;
        publish_tick();
    };

    // Snapshot publisher — copies the active motion's live window into the
    // ActivePlanSnapshot triple buffer so a transport thread can compute a trim
    // locally without waiting for the next control tick. Called only when the
    // active motion changes (initial bind, CONTROL entry, B/A swaps, inbox
    // install, trim commit).
    std::uint64_t snapshot_epoch = 0;
    auto publish_active_snapshot = [this, &snapshot_epoch]() noexcept {
        if (deps_.active_plan_snapshot_out == nullptr) {
            return;
        }
        const auto* m = deps_.observation_builder.motion();
        if (m == nullptr) {
            return;
        }
        auto& slot = deps_.active_plan_snapshot_out->producer_slot();
        slot.start_row    = 0;
        slot.frame_count  = m->frame_count;
        slot.output_fps   = m->output_fps;
        slot.input_fps    = m->input_fps;
        // The live window is indexed by inference_tick directly, so snapshot
        // row 0 (= window row 0) always corresponds to tick 0.
        slot.install_tick = 0;
        slot.epoch        = ++snapshot_epoch;
        slot.remap        = deps_.observation_builder.remap();
        slot.tail_held = true;
        if (m->frame_count > 0) {
            const Eigen::Index r0 = static_cast<Eigen::Index>(m->start_row);
            const Eigen::Index nn = std::min<Eigen::Index>(
                m->frame_count, kMaxSnapshotRows);
            slot.dof_poss.topRows(nn)  = m->dof_poss.middleRows(r0, nn);
            slot.base_poss.topRows(nn) = m->base_poss.middleRows(r0, nn);
            slot.base_rots.topRows(nn) = m->base_rots.middleRows(r0, nn);
            slot.frame_count = static_cast<std::int32_t>(nn);
            // Longer than the snapshot (only the padded standing pose in practice):
            // the cut is harmless when every later row repeats the last copied one,
            // because a trim past the window then answers with the held frame. Any
            // other tail makes the copy a truncated prefix (see tail_held).
            const Eigen::Index last = r0 + nn - 1;
            for (Eigen::Index r = r0 + nn; r < r0 + m->frame_count && slot.tail_held; ++r) {
                slot.tail_held = m->dof_poss.row(r) == m->dof_poss.row(last)
                                 && m->base_poss.row(r) == m->base_poss.row(last)
                                 && m->base_rots.row(r) == m->base_rots.row(last);
            }
        }
        // Zero-loss master clip (copied so the server-side trim can slice it
        // exactly, no resample). Absent for standing/CSV motions (master_rows==0).
        slot.master_fps  = m->master_fps;
        slot.master_offset = m->master_offset;
        slot.master_rows = 0;
        if (m->master_rows > 0) {
            const Eigen::Index mn = std::min<Eigen::Index>(
                m->master_rows,
                static_cast<Eigen::Index>(policy::RawMotionPlan::kMaxRows));
            slot.master_qpos.topRows(mn) = m->master_qpos.topRows(mn);
            slot.master_rows = static_cast<std::int32_t>(mn);
        }
        deps_.active_plan_snapshot_out->publish();
    };

    // Anchor the ACTIVE motion's frame 0 onto the robot's odom XY, keeping its own Z.
    // Used at TWO points: the initial publish below (so the very first reference external
    // consumers and the policy see is already correct) and CONTROL entry (where the yaw-align
    // happens). Both are skipped when cfg_.align_motion_to_robot is false (the motion is played
    // in its own frame).
    // `align_q` is the orientation the observation is measured against (the odometry attitude).
    auto anchor_active_to_robot = [&](policy::MotionData* m,
                                       const Eigen::Vector4f& align_q,
                                       const char* where) noexcept {
        if (!cfg_.align_motion_to_robot || m == nullptr || !state_.odom_valid
            || m->frame_count == 0) {
            return;
        }
        const Eigen::Vector3f anchor{
            state_.odom_position[0], state_.odom_position[1],
            m->base_poss(m->start_row, 2)};   // keep the authored standing height
        policy::align_motion_to_anchor(*m, anchor, align_q);
        std::fprintf(stderr,
                     "[control_loop] motion position-anchored to robot @ %s: "
                     "xy=(%+.3f,%+.3f) z=%.3f (kept)\n",
                     where,
                     static_cast<double>(anchor[0]),
                     static_cast<double>(anchor[1]),
                     static_cast<double>(anchor[2]));
    };

    // Initial publish: the standing motion is already bound by the caller.
    //
    // SEED state_ FIRST. `state_` is otherwise not populated until the loop's read_state()
    // below, so this publish would run against a default-constructed state -- odom_valid
    // false, odom_position zero -- and the FIRST reference anyone saw would be the raw standing
    // pose at the world origin (0, 0, 0.76), before CONTROL was ever entered.
    // wait_for_state() in run() already guarantees a state is available, and read_state is a
    // pure snapshot of the backend's current pose (no queue to consume), so reading it here is
    // free and cannot starve the loop. kStaleOk is a populated state too: on the robot the
    // odometry preflight in run() has usually just consumed the newest snapshot.
    if (const auto st = deps_.backend.read_state(state_);
        st == robot::ReadStatus::kOk || st == robot::ReadStatus::kStaleOk) {
        anchor_active_to_robot(
            deps_.motion_switcher != nullptr ? deps_.motion_switcher->active()
                                              : deps_.mutable_motion,
            state_.odom_quaternion_wxyz,
            "startup (initial publish)");
    }
    publish_active_snapshot();

    // Apply a switcher-driven motion swap: point the obs builder at the new
    // motion, reset the inference tick, publish a fresh snapshot, mark
    // operator_acted, and log. Used by CONTROL-entry yaw-align, B standup,
    // A replay. Like the inbox swap, this preserves obs history — matching
    // Python G1Controller.send_ref_motion, which never resets the history
    // buffers on a motion change.
    auto apply_swap = [&](policy::MotionData* m, bool& acted,
                          const char* msg) noexcept {
        deps_.observation_builder.set_motion(m);
        reset_tick();
        publish_active_snapshot();
        acted = true;
        std::fprintf(stderr, "[control_loop] %s\n", msg);
    };

    // Per-iteration timing trace (--trace_csv). `tracing` is a single load
    // hoisted out of the loop; when off, every `stamp(...)` below collapses to a
    // null PhaseStamp (one branch, no clock read) and nothing is recorded.
    const bool   tracing      = (deps_.loop_trace != nullptr);
    std::int64_t prev_tick_ns = 0;
    auto stamp = [tracing](std::int64_t& out) noexcept {
        return core::PhaseStamp{tracing ? &out : nullptr};
    };

    while (true) {
        if (stop_.load(std::memory_order_acquire)) {
            deps_.state_machine.request_stop();
        }

        const auto tick = ticker.wait_next();
        deps_.jitter.record(tick.jitter.count());
        const std::int64_t tc = tick_count_.load(std::memory_order_relaxed);
        tick_count_.store(tc + 1, std::memory_order_release);

        probe.tick();
        auto probe_scope = probe.scope();

        // Zero-init every tick so the accumulating PhaseStamps start from 0.
        core::LoopSample smp{};
        std::int64_t     body_begin_ns = 0;
        if (tracing) {
            body_begin_ns = core::MonotonicClock::now().count();
            smp.iter      = tc;
            smp.t_mono_ns = tick.actual.count();
            smp.period_ns = (prev_tick_ns != 0)
                                ? (tick.actual.count() - prev_tick_ns)
                                : 0;
            // Deviation of the MEASURED period from the nominal one, not the
            // RT-scheduler phase error (tick.jitter), so traces from loops with
            // and without a scheduled deadline stay comparable. tick.jitter is
            // still summarized by deps_.jitter at exit and under --profile.
            smp.jitter_ns = (smp.period_ns != 0)
                                ? (smp.period_ns - cfg_.control_period.count())
                                : 0;
            prev_tick_ns  = tick.actual.count();
        }

        // 1. Read latest state + remote.
        robot::ReadStatus read_status;
        {
            p_read_state.tick();
            auto s  = p_read_state.scope();
            auto ts = stamp(smp.obs_ns);
            read_status = deps_.backend.read_state(state_);
        }
        // kNoData can only happen after wait_for_state succeeded: state_ keeps
        // the last sample, and the safety monitor's staleness check decides
        // when to latch.
        (void)read_status;
        {
            p_poll_remote.tick();
            auto s  = p_poll_remote.scope();
            auto ts = stamp(smp.obs_ns);
            (void)deps_.backend.poll_remote(remote_);
            if (deps_.remote_buttons_out != nullptr) {
                std::uint32_t bits = 0;
                for (std::size_t i = 0; i < remote_.button.size() && i < 32; ++i) {
                    if (remote_.button[i] != 0) {
                        bits |= (1U << i);
                    }
                }
                deps_.remote_buttons_out->store(bits, std::memory_order_relaxed);
            }
        }

        // 1b. Publish the freshly-read state to the state outbox (if attached).
        //     Transports answer state queries from this snapshot via
        //     `try_acquire_latest()` (wait-free). Cost: one RobotState copy +
        //     one CAS — well under 1 µs on Orin.
        if (deps_.state_outbox != nullptr) {
            deps_.state_outbox->producer_slot() = state_;
            deps_.state_outbox->publish();
        }

        // 1c. Drain the depth inbox (WAIT-FREE) and hand the freshest frame to the
        //     observation builder's image group. Held across ticks: a tick with no
        //     new frame reuses last_depth_ (nullptr => far-plane fallback). On the
        //     REAL robot the depth producer lives entirely off this RT thread, so
        //     its rate never affects control-loop timing. A sim backend that
        //     renders depth inline in step() breaks that: the render runs on this
        //     thread, so sim loop timing is then not a proxy for real.
        //
        //     Staleness IS acted on, in two stages. `seq` is the producer's
        //     monotonic frame id, so a seq that stops advancing means the camera or
        //     the depth thread has died. First the stall is made loud
        //     (depth_stale_warn_ticks). Then, past depth_stale_far_ticks, the
        //     last-known-good frame is ABANDONED for the far plane: a frozen depth
        //     image is worse than no image, because the policy goes on confidently
        //     steering around geometry that is no longer there and nothing else in
        //     the system shows a symptom. Far = "the sensor is not telling me
        //     anything", which is the convention training and the producer already
        //     use. Neither stage blocks or skips a tick; only the image changes.
        //     Cost when depth is unattached: one predictable branch.
        if (deps_.depth_inbox != nullptr) {
            if (const auto* d = deps_.depth_inbox->try_acquire_latest()) {
                last_depth_ = d;
            }
            // try_acquire_latest() returns nullptr both for "no new frame" and,
            // transiently, between publishes — so compare seq rather than the
            // pointer to decide whether the producer actually advanced.
            const std::int64_t seq = (last_depth_ != nullptr) ? last_depth_->seq : -1;
            if (last_depth_ == nullptr) {
                // No frame received yet: the producer is still starting, not stalled,
                // so these ticks are not counted as stale (counting them would trip the
                // far-plane fallback for a camera that never ran). The CONTROL-entry
                // gate below keeps the policy from starting here; a loop that starts in
                // CONTROL or a dry run bypasses it, so say so once if the policy runs.
                if (!depth_none_warned_
                    && deps_.state_machine.state() == math::ControlState::kControl) {
                    std::fprintf(stderr,
                                 "[control_loop] WARNING: policy running with NO depth frame "
                                 "received yet — feeding the FAR PLANE (empty world)\n");
                    depth_none_warned_ = true;
                }
            } else if (seq != last_depth_seq_) {
                last_depth_seq_ = seq;
                ++depth_frames_seen_;
                if (depth_stale_warned_) {
                    std::fprintf(stderr,
                                 "[control_loop] depth recovered after %d stale ticks%s\n",
                                 depth_stale_ticks_,
                                 depth_far_fallback_ ? " (was on the far plane)" : "");
                    depth_stale_warned_ = false;
                }
                depth_stale_ticks_  = 0;
                depth_far_fallback_ = false;
            } else {
                ++depth_stale_ticks_;
                depth_max_stale_ticks_ = std::max(depth_max_stale_ticks_, depth_stale_ticks_);
                if (!depth_stale_warned_
                    && cfg_.depth_stale_warn_ticks > 0
                    && depth_stale_ticks_ >= cfg_.depth_stale_warn_ticks) {
                    std::fprintf(stderr,
                                 "[control_loop] WARNING: depth frozen for %d ticks (%.0f ms) — "
                                 "still feeding the last-known-good frame; check the depth "
                                 "producer\n",
                                 depth_stale_ticks_,
                                 static_cast<double>(depth_stale_ticks_)
                                     * static_cast<double>(cfg_.control_period.count()) * 1.0e-6);
                    depth_stale_warned_ = true;  // once per stall, not once per tick
                }
                // Announce the switch to the far plane exactly once per stall. It
                // changes what the policy sees, so it must never be inferred from
                // the absence of a message.
                if (!depth_far_fallback_
                    && cfg_.depth_stale_far_ticks > 0
                    && depth_stale_ticks_ >= cfg_.depth_stale_far_ticks) {
                    std::fprintf(stderr,
                                 "[control_loop] depth STALE for %d ticks (%.0f ms) — abandoning "
                                 "the frozen frame, feeding the FAR PLANE (empty world) until the "
                                 "producer recovers\n",
                                 depth_stale_ticks_,
                                 static_cast<double>(depth_stale_ticks_)
                                     * static_cast<double>(cfg_.control_period.count()) * 1.0e-6);
                    depth_far_fallback_ = true;
                    ++depth_far_fallback_events_;
                }
            }
            // nullptr is the builder's far-plane fill, so the fallback needs no
            // separate buffer and no copy — the image group already handles the
            // never-received-a-frame case this way.
            deps_.observation_builder.set_depth(depth_far_fallback_ ? nullptr : last_depth_);

            // Per-tick depth freshness for the trace. Three stores into a stack
            // POD and one subtraction — no clock read (the tick's own wake-up
            // timestamp is the reference), no branch on the producer. Every row
            // then carries the age of the frame that tick actually used,
            // alongside that tick's period and jitter, so a depth stall and a
            // loop overrun can be checked for coincidence.
            if (tracing) {
                smp.depth_seq   = last_depth_seq_;
                smp.depth_stale = depth_stale_ticks_;
                if (last_depth_ != nullptr && last_depth_->stamp_ns > 0) {
                    smp.depth_age_ns = tick.actual.count() - last_depth_->stamp_ns;
                }
            }
        }

        // 2. Inside CONTROL: build obs → infer → decode → cache cmd. Inference
        //    runs on every tick (control rate == inference rate).
        //
        //    Python parity: base_controller.apply_action decodes the action
        //    against the joint pos/vel read this tick. We decode with the
        //    just-read (fresh) state below, matching the per-tick inference of
        //    Python run_real_step.
        std::optional<math::MotorCommand<kDof>> fresh_cmd;
        if (deps_.state_machine.state() == math::ControlState::kControl) {
            // Build every observation group (one per engine input), then infer.
            inference::InferenceStatus infer_status;
            {
                p_obs_build.tick();
                auto s  = p_obs_build.scope();
                auto ts = stamp(smp.obs_ns);
                deps_.observation_builder.build(
                    state_, clipped_action_train_, static_cast<std::int32_t>(inference_tick));
                deps_.observation_builder.clip_observations(cfg_.clip_observation);
            }
            {
                p_inference.tick();
                auto s  = p_inference.scope();
                auto ts = stamp(smp.infer_ns);
                const auto& outs = deps_.observation_builder.outputs();
                for (std::size_t gi = 0; gi < outs.size(); ++gi) {
                    engine_inputs_[static_cast<std::size_t>(group_engine_idx_[gi])] =
                        std::span<const float>{outs[gi].data.data(),
                                               static_cast<std::size_t>(outs[gi].data.size())};
                }
                std::span<float> out{raw_action_train_.data(),
                                     static_cast<std::size_t>(raw_action_train_.size())};
                infer_status = deps_.engine.infer_multi(
                    std::span<const std::span<const float>>{engine_inputs_.data(),
                                                            engine_inputs_.size()},
                    out);
            }
            if (infer_status != inference::InferenceStatus::kOk) {
                // Treat as safety fault — set to zero so the torque-aware
                // clamp in the decoder produces the default pose. Announce the first
                // failure of a streak, and stop the run if the engine does not recover
                // within kEngineFailStopTicks: holding the default pose forever with a
                // dead engine is a silent failure.
                raw_action_train_.setZero();
                if (engine_fail_streak_ == 0) {
                    std::fprintf(stderr,
                                 "[control_loop] inference FAILED (%s) at inference tick %lld; "
                                 "holding the default pose\n",
                                 infer_status_name(infer_status),
                                 static_cast<long long>(inference_tick));
                }
                ++engine_fail_streak_;
                if (engine_fail_streak_ == kEngineFailStopTicks) {
                    std::fprintf(stderr,
                                 "[control_loop] FATAL: inference failed %d ticks in a row; "
                                 "stopping\n",
                                 engine_fail_streak_);
                    engine_failed_ = true;
                    request_stop();
                }
            } else if (engine_fail_streak_ > 0) {
                std::fprintf(stderr, "[control_loop] inference recovered after %d failed ticks\n",
                             engine_fail_streak_);
                engine_fail_streak_ = 0;
            }
            // depth_stats diagnostic (cfg_.depth_stats_every N, 0 = off). Placed HERE, in
            // between the engine returning and the action clip: the action reported
            // is the engine's own output on a kOk tick (a non-finite one is flagged
            // before the guard below replaces it), or the zeros the line above
            // substituted on a failed one (the printed status names which). Before
            // the clip, so a ±1e30 output shows as ±1e30 and not as a plausible
            // ±clip_action. `inference_tick` has not been incremented yet, so N=1
            // logs the very first inference — the frame most likely to be wrong.
            // Cost when disabled: this one integer compare.
            if (cfg_.depth_stats_every > 0
                && (inference_tick % cfg_.depth_stats_every) == 0) {
                print_depth_stats(inference_tick, tick.actual.count(), infer_status);
            }
            // A non-finite action (NaN from the engine, or from a NaN observation) must take
            // the same path: the decoder's clamp turns NaN into a finite limit
            // (std::max(lo, NaN) == lo), i.e. full torque on every joint, and the safety
            // monitor's finite-command check never sees it.
            const bool action_finite = raw_action_train_.allFinite();
            if (!action_finite) {
                raw_action_train_.setZero();
            }
            {
                static bool nonfinite_logged = false;
                if (!action_finite && !nonfinite_logged) {
                    std::fprintf(stderr,
                                 "[control_loop] non-finite policy action; holding the "
                                 "default pose until it is finite again\n");
                }
                nonfinite_logged = !action_finite;
            }
            clip_inplace(raw_action_train_, cfg_.clip_action);
            ++inference_tick;
            publish_tick();

            // Decode using the state just read at this inference tick (what
            // Python's compute_motor_targets sees for this tick's command). The
            // decode writes clipped_action_train_ (the torque-CLIPPED action,
            // training order), which the NEXT tick's observation feeds as
            // last_action — clipped, and with no first-step skip.
            {
                p_decode.tick();
                auto s  = p_decode.scope();
                auto ts = stamp(smp.act_ns);
                math::MotorCommand<kDof> policy_cmd;
                deps_.action_decoder.decode(raw_action_train_, state_.joint_positions,
                                             state_.joint_velocities, policy_cmd,
                                             clipped_action_train_);
                fresh_cmd   = policy_cmd;
                cached_cmd_ = policy_cmd;
            }
        }

        // 2b. CONTROL-entry gate. The policy must not start on an input it is
        //     not being given:
        //       depth — before the first frame (or while the stall fallback has
        //               it on the far plane) the image group says "open space
        //               everywhere". Dry runs actuate nothing and are exempt.
        //       odom  — the observation is measured against the odometry pose;
        //               without a live odom stream there is no anchor at all.
        //               Not exempt: a dry run must enter CONTROL on the same
        //               inputs as a powered one.
        //     Mask A in DEFAULT_POS so the state machine stays there; everything
        //     else (Start, Select, stop, damping, safety) sees the real remote. A
        //     press refused here stays masked until A is released, so an input
        //     arriving while A is held does not start the policy unasked. A loop
        //     that starts in CONTROL never passes through this gate; the odom
        //     check in step 4 covers it.
        robot::RemoteState        masked_remote;
        const robot::RemoteState* sm_remote = &remote_;
        {
            const bool depth_gated = deps_.depth_inbox != nullptr && depth_group_idx_ >= 0
                                     && !cfg_.dry_run;
            const bool depth_live  = last_depth_ != nullptr && !depth_far_fallback_;
            const bool odom_live   = state_.odom_valid;
            const char* blocked    = nullptr;
            if (depth_gated && !depth_live) {
                blocked = last_depth_ == nullptr
                              ? "[control_loop] waiting for the first depth frame; "
                                "staying in DEFAULT_POS\n"
                              : "[control_loop] depth stale on the far plane; not "
                                "entering CONTROL\n";
            } else if (!odom_live) {
                blocked = "[control_loop] no live odometry; staying in DEFAULT_POS\n";
            }
            if (blocked == nullptr) {
                entry_wait_logged_ = false;  // next wait is a new episode
            }
            const bool a_down = remote_.button[math::key_map::kA] != 0;
            if (!a_down) {
                entry_gate_holding_a_ = false;
            } else if ((blocked != nullptr || entry_gate_holding_a_)
                       && deps_.state_machine.state() == math::ControlState::kDefaultPos) {
                if (blocked != nullptr && !entry_wait_logged_) {
                    std::fprintf(stderr, "%s", blocked);
                    entry_wait_logged_ = true;
                }
                masked_remote = remote_;
                masked_remote.button[math::key_map::kA] = 0;
                sm_remote             = &masked_remote;
                entry_gate_holding_a_ = true;
            }
        }

        // 3. State machine drives the published command (respects its own state).
        math::MotorCommand<kDof> cmd_out = cached_cmd_;  // seed with last command
        bool cont;
        {
            p_state_machine.tick();
            auto s = p_state_machine.scope();
            cont = deps_.state_machine.update(state_, *sm_remote, fresh_cmd,
                                                tick.actual.count(), cmd_out);
        }

        // Diagnostic: on rising edges of the gating buttons, print the
        // resulting state transition (or note that no transition occurred).
        // Fires only on a button press — silent every other tick — so the
        // control loop's hot path stays clean.
        bool operator_acted = false;
        {
            static std::uint8_t       dbg_prev_start  = 0;
            static std::uint8_t       dbg_prev_a      = 0;
            static std::uint8_t       dbg_prev_b      = 0;
            static std::uint8_t       dbg_prev_select = 0;
            static math::ControlState dbg_prev_state  = math::ControlState::kInit;
            static bool               dbg_first_tick  = true;
            if (dbg_first_tick) {
                dbg_prev_state = deps_.state_machine.state();
                dbg_first_tick = false;
                std::fprintf(stderr, "[control_loop] start state=%s\n",
                             state_name(dbg_prev_state));
            }

            const auto& b = remote_.button;
            struct Edge { bool rising; const char* name; };
            const Edge edges[] = {
                {b[math::key_map::kStart]  != 0 && dbg_prev_start  == 0, "Start" },
                {b[math::key_map::kA]      != 0 && dbg_prev_a      == 0, "A"     },
                {b[math::key_map::kB]      != 0 && dbg_prev_b      == 0, "B"     },
                {b[math::key_map::kSelect] != 0 && dbg_prev_select == 0, "Select"},
            };
            char        names[48] = {};
            std::size_t pos       = 0;
            for (const auto& e : edges) {
                if (e.rising) {
                    const int n = std::snprintf(names + pos, sizeof(names) - pos,
                                                 pos == 0 ? "%s" : "+%s", e.name);
                    if (n > 0) {
                        pos += static_cast<std::size_t>(n);
                    }
                }
            }
            if (names[0] != '\0') {
                const auto state_after = deps_.state_machine.state();
                if (state_after != dbg_prev_state) {
                    std::fprintf(stderr,
                                 "[control_loop] %s pressed: transitioning from %s to %s\n",
                                 names, state_name(dbg_prev_state),
                                 state_name(state_after));
                } else {
                    std::fprintf(stderr,
                                 "[control_loop] %s pressed (state=%s, no transition)\n",
                                 names, state_name(state_after));
                }
            }

            // Yaw-align the motion to the robot's current orientation on the
            // single kDefaultPos → kControl transition. Without this
            // the motion-anchor orientation feature contains a large yaw
            // delta on tick 1 of CONTROL and the policy snaps the body to
            // match — visible to the operator as an abrupt initial movement.
            //
            // Routing:
            //   - With a motion_switcher: switch_to_original(robot_quat) does
            //     the yaw-align in place AND points the obs builder at the
            //     active (original, else standing) motion.
            //   - Without a switcher (fixtures with no swap support): align
            //     mutable_motion in place.
            const auto state_after = deps_.state_machine.state();
            if (state_after == math::ControlState::kControl
                && dbg_prev_state != math::ControlState::kControl
                && cfg_.align_motion_to_robot) {
                // Align to the orientation the policy observes (the odometry attitude),
                // or the policy sees a constant yaw error and spins the body to close it
                // the moment kControl is entered. Matches Python
                // G1Controller._orientation_quat. The entry gate (step 2b) guarantees
                // the odometry is live here.
                const Eigen::Vector4f align_q = state_.odom_quaternion_wxyz;
                //
                // Then anchor the POSITION too, so the reference starts at the robot's
                // odom XY instead of the world origin. The FIRST plan install does
                // exactly this too (cold start has `have_ref_anchor == false`, so
                // send_ref_motion takes the ROBOT anchor and snaps frame 0 onto odom).
                // This applies the same rule at CONTROL entry instead of leaving the
                // pre-plan window as a special case.
                //
                // XY AND YAW ONLY:
                //   * Z keeps the motion's own frame-0 value (passed through as the anchor Z, so
                //     the shift is exactly zero). Anchoring Z to odom would inherit odom's Z
                //     drift and make the reference height wander.
                //   * Roll/pitch are never touched -- align_motion_to_anchor carries yaw only.
                // The yaw here is already closed by the align above, so align_motion_to_anchor
                // sees |yaw_offset| < 1 mrad and degrades to a pure translation.
                //
                // Guarded on odom_valid inside anchor_active_to_robot: without a
                // trustworthy position there is nothing to anchor to.
                if (deps_.motion_switcher != nullptr) {
                    policy::MotionData* m =
                        deps_.motion_switcher->switch_to_original(align_q);
                    anchor_active_to_robot(m, align_q, "CONTROL entry");   // before apply_swap: it publishes the snapshot
                    apply_swap(
                        m,
                        operator_acted,
                        "motion yaw-aligned to robot @ CONTROL entry "
                        "(via switcher; counter reset)");
                } else if (deps_.mutable_motion != nullptr) {
                    policy::align_motion_to_yaw(*deps_.mutable_motion,
                                                  align_q);
                    anchor_active_to_robot(deps_.mutable_motion, align_q, "CONTROL entry");
                    // Republish: the yaw-align + anchor above changed the motion in place, and
                    // off-loop trims read the snapshot, not the motion. Without this they keep
                    // slicing the pre-anchor (origin) reference until the next install. The
                    // window, its indexing and the counter are unchanged (install_tick stays 0),
                    // so this only refreshes the content.
                    publish_active_snapshot();
                    operator_acted = true;
                    std::fprintf(stderr,
                                 "[control_loop] motion yaw-aligned to robot @ CONTROL entry\n");
                }
            }

            // Operator-driven motion swaps inside CONTROL. Mirrors the Python
            // do_standup / replay_original_motion paths in deploy_real.py.
            // The state machine sets these flags on rising-edge presses while
            // in kControl; we consume them here and clear immediately so each
            // press fires exactly one swap. Both swaps:
            //   1. yaw-align the chosen motion to the robot's current orient.
            //   2. point the observation builder at the new motion.
            //   3. reset the inference tick so the new motion plays from t=0.
            // Only the in-CONTROL transition uses these flags; the state
            // machine never sets them in any other state, so no extra guard.
            if (deps_.motion_switcher != nullptr
                && state_after == math::ControlState::kControl) {
                const auto& flags = deps_.state_machine.flags();
                // Same orientation source as the CONTROL-entry align above.
                const Eigen::Vector4f swap_q = state_.odom_quaternion_wxyz;
                if (flags.standup_requested) {
                    apply_swap(
                        deps_.motion_switcher->switch_to_standing(swap_q),
                        operator_acted,
                        "B → standup: motion → standing (yaw-aligned, counter reset)");
                    // In multi-process mode (original==nullptr) B ALSO latches the
                    // tracker-ready signal that tells the generator to start. Kept
                    // on B (not A) so the operator presses A only to stand, then a
                    // deliberate, separate B to release the generator — avoids an
                    // A double-press starting motion the instant the robot stands.
                    // Latched (never cleared); later B presses are standup-only.
                    if (deps_.motion_switcher->original() == nullptr
                        && deps_.tracker_ready_out != nullptr) {
                        bool expected = false;
                        deps_.tracker_ready_out->compare_exchange_strong(
                            expected, true, std::memory_order_release);
                        if (expected == false) {  // was false, now true
                            std::fprintf(stderr,
                                "[control_loop] B pressed: tracker_ready → true\n");
                        }
                    }
                }
                if (flags.motion_replay_requested) {
                    // Single-process: A replays the original CSV motion. In
                    // multi-process mode (original==nullptr) there is no original
                    // to replay and A is a no-op — the tracker-ready signal now
                    // lives on B (see the standup block above).
                    if (deps_.motion_switcher->original() != nullptr) {
                        apply_swap(
                            deps_.motion_switcher->switch_to_original(swap_q),
                            operator_acted,
                            "A → replay: motion → original (yaw-aligned, counter reset)");
                    }
                }
                if (flags.standup_requested || flags.motion_replay_requested) {
                    deps_.state_machine.clear_flags();
                }
            }

            dbg_prev_start  = b[math::key_map::kStart];
            dbg_prev_a      = b[math::key_map::kA];
            dbg_prev_b      = b[math::key_map::kB];
            dbg_prev_select = b[math::key_map::kSelect];
            dbg_prev_state  = deps_.state_machine.state();
        }

        // 4. SafetyMonitor — may rewrite the command to damping.
        bool safe;
        {
            p_safety.tick();
            auto s = p_safety.scope();
            safe = deps_.safety.check_and_guard(tick.actual.count(), state_,
                                                  remote_, cmd_out);
            // Odometry staleness, the twin of the stale-state check: the backend
            // clears odom_valid once the stream is older than its timeout
            // (UnitreeBackendConfig::odom_stale_ns). Without odometry the anchor
            // observation has no reference frame, and falling back to the IMU
            // would jump it by the IMU-odom yaw offset, so latch and damp exactly
            // as for a stale state.
            if (safe && !state_.odom_valid
                && deps_.state_machine.state() == math::ControlState::kControl) {
                std::fprintf(stderr,
                             "[control_loop] FATAL: the odometry went stale in CONTROL; "
                             "damping\n");
                deps_.safety.trigger_fault(robot::SafetyFault::kStaleOdom, cmd_out);
                safe = false;
            }
        }

        // 5. Publish.
        {
            p_send_command.tick();
            auto s  = p_send_command.scope();
            auto ts = stamp(smp.act_ns);
            deps_.backend.send_command(cmd_out);
        }

        // 6. Sim backends step here (no-op for real Unitree).
        {
            auto ts = stamp(smp.act_ns);
            deps_.backend.step();
        }

        // 7. Plan install + trim commit run AFTER the command is sent: neither
        //    changes this tick's command (it was decoded from the old plan
        //    above), so their cost lands in the idle time before the next tick
        //    instead of delaying the send. The counter only moves at the next
        //    inference, so start_offset / the trim cut see this tick's value.
        //
        // 7a. Inbox-driven motion swap (operator wins): skipped this tick if
        //     the operator block above fired.
        if (!operator_acted
            && deps_.motion_inbox != nullptr
            && deps_.motion_switcher != nullptr
            && deps_.state_machine.state() == math::ControlState::kControl) {
            if (const auto* raw = deps_.motion_inbox->try_acquire_latest()) {
                // Streaming pipeline (parity with G1Controller.send_ref_motion):
                // resample the full input, joint remap, slice the live window by
                // the tracker's inference_tick when kFlagAutoStartOffset is set
                // (continuation of a trim) or 0 (fresh plan), keep the master
                // clip, velocities, anchor rebase. The data transformation lives
                // in g1::send_ref_motion (runs without a loop); the
                // bookkeeping below touches loop-private state (swap metrics,
                // tick reset, snapshot republish, sim pause). No transition
                // blending: continuity at the seam is the generator's job
                // (overlapping chunks).
                const SwapResult swap = send_ref_motion(
                    *raw,
                    deps_.observation_builder.motion(),
                    state_,
                    static_cast<std::int32_t>(inference_tick),
                    processed_motion_scratch_,
                    deps_.observation_builder.remap());
                if (swap.installed) {
                    swap_joint_mean_.push_back(swap.metrics.joint_mean);
                    swap_joint_l2_.push_back(swap.metrics.joint_l2);
                    swap_base_xy_.push_back(swap.metrics.base_xy);
                    swap_base_z_.push_back(swap.metrics.base_z);

                    deps_.observation_builder.set_motion(&processed_motion_scratch_);
                    // Python parity: do NOT reset observation_builder history here.
                    // G1Controller.send_ref_motion never resets the history buffers on
                    // a motion change — they are observations of robot state (and
                    // motion-relative orientation) that remain valid because the swap
                    // changes the motion, not the robot. A reset would make the next
                    // push first-fill every slot with the current value, producing an
                    // internally inconsistent history block (constant pose + non-zero
                    // velocity) that the policy never saw in training.
                    reset_tick();
                    publish_active_snapshot();
                    // Installed and live: a transport waiting on plans_consumed() can reply now.
                    plans_installed_.fetch_add(1, std::memory_order_release);
                    plans_consumed_.fetch_add(1, std::memory_order_release);
                    // Sim-eval aid: BLOCK the entire control loop for
                    // sim_pause_after_plan right after a plan installs, so a
                    // slow external orchestrator in a separate process isn't
                    // outrun. Nothing runs — no state read,
                    // policy, command, or mj_step — while the orchestrator generates
                    // the next chunk, so the sim is frozen and only wall-clock
                    // passes. A transport thread keeps answering state queries
                    // from the just-published (frozen) snapshot, and
                    // PeriodicTicker resyncs on the next wait_next() after the
                    // overrun (no catch-up burst). Off by default (count() == 0).
                    if (cfg_.sim_pause_after_plan.count() > 0) {
                        (void)core::MonotonicClock::sleep_until(
                            core::MonotonicClock::now()
                            + cfg_.sim_pause_after_plan);
                    }
                } else {
                    // Rejected (late, non-finite, bad header): still consumed, so a waiting
                    // transport stops waiting and can report the rejection.
                    plans_consumed_.fetch_add(1, std::memory_order_release);
                }
            }
        }

        // 7b. Trim-install commit — runs when an off-loop transport thread has
        //     already replied to a trim request from its ActivePlanSnapshot copy
        //     and signaled trim_install_pending=true. The control loop only
        //     commits the side-effect: advance the live window and the exact
        //     master position by the cut, record the trim phase
        //     (policy::commit_trim_cut) and re-publish the snapshot. No reply,
        //     no compute (both happened on the transport thread).
        if (deps_.trim_install_pending != nullptr
            && deps_.trim_install_tick != nullptr
            && deps_.trim_install_pending->load(std::memory_order_acquire)) {
            const std::int32_t cut =
                deps_.trim_install_tick->load(std::memory_order_acquire);
            // The server computed `cut` from the counter it read (= the next row to play;
            // snapshots are always published with install_tick 0). By now this tick (and any
            // tick between the reply and here) has already PLAYED rows cut, cut+1, ...; the
            // counter says how many. Resetting the counter to 0 would replay row `cut` (a 20 ms
            // freeze at every trim), so continue from those already-played rows instead.
            std::int32_t played = 0;
            std::int32_t live_after = 0;
            // Stale trim: the server read a counter LARGER than the current one, i.e. it read the
            // old plan's counter and a plan install reset the tick in between. Applying that cut to
            // the new plan would jump it ahead (~0.4 s). Drop it (the reply came from the old plan).
            const bool stale = cut > static_cast<std::int32_t>(inference_tick);
            if (stale) {
                std::fprintf(stderr,
                             "[control_loop] stale trim dropped: cut=%d > counter=%lld "
                             "(a plan installed after the trim was read)\n",
                             cut, static_cast<long long>(inference_tick));
                deps_.trim_install_pending->store(false, std::memory_order_release);
            } else {
                auto* mut = deps_.observation_builder.mutable_motion();
                std::int32_t applied = 0;
                if (cut > 0 && mut != nullptr && mut->frame_count > 0) {
                    // A cut the server clamped to frame_count (playback ran past the window)
                    // keeps only the held final frame; the counter restarts there.
                    applied = std::min(cut, mut->frame_count - 1);
                    (void)policy::commit_trim_cut(*mut, cut);   // also records trim_phase
                    live_after = mut->frame_count;
                    played = std::clamp<std::int32_t>(
                        static_cast<std::int32_t>(inference_tick) - applied, 0,
                        std::max<std::int32_t>(live_after - 1, 0));
                } else {
                    // Nothing dropped (cut 0: the server read the counter right after an install).
                    // Keep the counter: resetting it would replay the rows played since the trim.
                    // The trim still sent whole frames from master_trim_start(master_offset), so
                    // record its phase for the next install.
                    if (mut != nullptr) {
                        (void)policy::commit_trim_cut(*mut, 0);
                        live_after = mut->frame_count;
                    }
                    played = static_cast<std::int32_t>(inference_tick);
                }
                reset_tick();
                publish_active_snapshot();          // install_tick 0: rows == counter values
                if (played > 0) {
                    inference_tick = played;
                    publish_tick();
                }
                deps_.trim_install_pending->store(false, std::memory_order_release);
                std::fprintf(stderr,
                             "[control_loop] trim install committed: cut=%d applied=%d → "
                             "%d frames remaining; counter -> %d (rows already played)\n",
                             cut, applied, live_after, played);
            }
        }

        // 7c. Reference outbox: the row the active motion holds at the counter, after
        //     any install/trim above, so a transport never reads the motion buffers.
        if (deps_.ref_outbox != nullptr) {
            if (const auto* m = deps_.observation_builder.motion();
                m != nullptr && m->frame_count > 0) {
                const std::int32_t lt = static_cast<std::int32_t>(inference_tick);
                const Eigen::Index r  = static_cast<Eigen::Index>(
                    m->start_row + std::min<std::int32_t>(lt, m->max_index()));
                auto& e      = deps_.ref_outbox->producer_slot();
                e.gtick      = tc;
                e.local_tick = lt;
                e.epoch      = snapshot_epoch;
                for (int k = 0; k < 3; ++k) e.pos[k] = m->base_poss(r, k);
                for (int k = 0; k < 4; ++k) e.quat_wxyz[k] = m->base_rots(r, k);
                for (std::size_t j = 0; j < policy::kG1Dof; ++j)
                    e.dof_train[j] = m->dof_poss(r, static_cast<Eigen::Index>(j));
                deps_.ref_outbox->publish();
            }
        }

        // 8. Record this iteration's timing. Placed before the exit checks so
        //    the final tick is captured too. One non-blocking ring push; all
        //    formatting and file I/O happen on the trace's writer thread.
        if (tracing) {
            smp.loop_ns = core::MonotonicClock::now().count() - body_begin_ns;
            smp.set_state(state_name(deps_.state_machine.state()));
            (void)deps_.loop_trace->record(smp);
        }

        if (!safe) {
            // The damping command was already published this tick; exit now.
            deps_.state_machine.request_stop();
            return ControlLoopExit::kSafetyLatched;
        }
        if (engine_failed_) {
            return ControlLoopExit::kEngineFailure;
        }
        if (!cont) {
            return ControlLoopExit::kStateMachineDone;
        }
        if (stop_.load(std::memory_order_acquire)) {
            return ControlLoopExit::kRequested;
        }
        if (cfg_.max_ticks > 0 && (tc + 1) >= cfg_.max_ticks) {
            return ControlLoopExit::kMaxTicks;
        }
    }
}

}  // namespace rt_deploy::g1
