//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// ControlLoop — the hard-real-time tick orchestrator for the G1.
//
// Runs at `control_period` (the apps derive it from the config's control_dt:
// 50 Hz for the G1). On every tick in CONTROL the policy pipeline runs
// (observation build → infer → action decode): inference and control share one
// rate. The MuJoCo sim keeps its fine physics step by
// sub-stepping inside the backend (MujocoBackendConfig::substeps), not by
// skipping inference here.
//
// The loop's own buffers are owned by the ControlLoop instance and reused (the
// config-driven ObservationBuilder::build still uses small Eigen temporaries).
// The only external I/O is:
//   - backend.read_state()  — reads from triple buffer (wait-free)
//   - backend.poll_remote() — reads from atomic snapshot
//   - engine.infer_multi()  — does its own blocking CUDA sync internally
//   - backend.send_command()— publishes via DDS (or, in sim, stores the
//                             command for backend.step())
//
// Ownership: the ControlLoop holds references to all components (non-owning).
// The caller constructs and outlives them.

#include "rt_deploy/core/clock.hpp"
#include "rt_deploy/core/jitter_histogram.hpp"
#include "rt_deploy/core/loop_trace.hpp"
#include "rt_deploy/core/rt_thread.hpp"
#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/g1/action_decoder.hpp"
#include "rt_deploy/g1/active_plan_snapshot.hpp"
#include "rt_deploy/g1/config.hpp"
#include "rt_deploy/g1/depth_frame.hpp"
#include "rt_deploy/g1/executed_ref_ring.hpp"
#include "rt_deploy/g1/motion_switcher.hpp"
#include "rt_deploy/g1/observation_builder.hpp"
#include "rt_deploy/g1/state_machine.hpp"
#include "rt_deploy/inference/i_inference_engine.hpp"
#include "rt_deploy/math/types.hpp"
#include "rt_deploy/policy/motion_loader.hpp"
#include "rt_deploy/policy/raw_motion_plan.hpp"
#include "rt_deploy/robot/i_robot_backend.hpp"
#include "rt_deploy/robot/safety_monitor.hpp"

#include <Eigen/Core>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <vector>

namespace rt_deploy::g1 {

struct ControlLoopConfig {
    std::chrono::nanoseconds control_period{20'000'000};  // 50 Hz; apps pass control_dt
    std::int64_t             state_ready_timeout_ns{10'000'000'000};  // 10 s
    // After the first state, wait this long for live odometry (the observation's
    // anchor frame) before ticking at all, and exit with kOdomTimeout if none
    // arrives (the Python deployment's 2 s preflight).
    std::int64_t             odom_ready_timeout_ns{2'000'000'000};    // 2 s
    core::RtThreadConfig     rt_thread_config{};
    // Safety / clipping bounds (duplicated from g1::Config for convenience)
    float                    clip_observation{100.0F};
    float                    clip_action{100.0F};
    // Optional: cap total ticks. 0 means unbounded.
    std::int64_t             max_ticks{0};
    // Pre-allocation size (in OUTPUT frames at the control rate) for the
    // internal MotionData scratch used when a RawMotionPlan arrives on the
    // motion inbox. Must include kMaxTransitionSteps of leading slack.
    // 0 disables inbox-driven motion processing (the inbox pointer in deps
    // must also be null in that case).
    std::int32_t             motion_scratch_capacity{0};
    // Anchor the active motion onto the robot (XY on odom, yaw on the anchor
    // orientation) at the initial publish and at CONTROL entry. false plays the
    // motion in its own frame (deploy_g1_mujoco --no_align_motion). Plan installs
    // from the motion inbox are anchored by send_ref_motion regardless.
    bool                     align_motion_to_robot{true};
    // Warn (once, then again on recovery) when the depth frame has not advanced
    // for this many control ticks. Only meaningful with depth_inbox attached.
    // Default 10 ticks = 200 ms at 50 Hz: the depth student was trained on
    // 40-120 ms of depth staleness, so 200 ms is already well out of distribution
    // and means the producer has stalled rather than merely run slow. Purely
    // diagnostic — it never changes what the policy is fed.
    std::int32_t             depth_stale_warn_ticks{10};
    // ABANDON the last-known-good depth frame after this many stale control ticks
    // and feed the far plane instead. 0 disables the fallback (hold forever).
    //
    // Holding a frozen frame indefinitely is worse than having no frame: the
    // policy keeps confidently steering around geometry that is no longer there,
    // and unlike a blank image that failure is completely silent. The far plane is
    // the honest answer to "the sensor is not telling me anything" — it is the
    // same convention the producer, training, and the never-received-a-frame path
    // already use, and it is what the policy saw for every dropped pixel during
    // training.
    //
    // This is NOT a stall and NOT a block: the loop still runs to time on the
    // tick it switches over. It only changes WHICH image is fed.
    //
    // Default 25 ticks = 500 ms at 50 Hz, matching the Python deployment's
    // RealSenseDepthSource stale_s. Deliberately far above depth_stale_warn_ticks
    // (200 ms) so the warning fires first and a short hiccup — the 40-120 ms of
    // staleness the policy trained on, or one dropped camera frame — never trips
    // it. Set 0 only for a bench experiment that must see the frozen frame.
    std::int32_t             depth_stale_far_ticks{25};
    // Diagnostic, OFF by default: every N INFERENCE ticks, dump the statistics of
    // the `student_depth` group tensor as the engine actually received it
    // (post-encode), of the raw metric DepthFrame it came from, and of the action
    // vector the engine returned. 0 = off, and off costs exactly one integer
    // compare per inference tick.
    //
    // This is the answer to "is the depth policy really being fed depth, in the
    // units and the encoding it was trained on?" — a question that cannot be
    // settled by reading the code, only by printing the numbers. A frame that is
    // 100% at the far bound, or a NaN in the action, is invisible everywhere else
    // in the system: the shapes are right, the engine returns kOk, and the robot
    // simply behaves as though the world were empty.
    //
    // NOT for a high-cadence powered run: it writes to stderr from the RT thread,
    // and if stderr is a pipe with a slow reader that write BLOCKS. Use N >= one
    // second's worth of ticks (50 at 50 Hz), or redirect stderr to a file.
    std::int32_t             depth_stats_every{0};
    // DRY RUN: no command reaches the actuators, so the loop is a measurement
    // harness rather than a controller. Set ONLY when actuation is genuinely
    // suppressed. Its only source is deploy_g1_real --no_send, which also
    // disables the backend's single DDS publish site (send_enabled = false).
    //
    // Its one effect: an `image` observation group with no depth inbox becomes a
    // loud WARNING instead of a refusal to start. That guard exists to stop a
    // depth policy driving the robot on a blank far-plane image — but a no-depth
    // timing baseline needs exactly that configuration: the same 3-input policy
    // and the same observation shape, with the depth producer's CPU/GPU cost
    // absent, so a with-depth run can be compared against it. With nothing
    // actuating, the hazard the guard prevents cannot occur.
    bool                     dry_run{false};
    // Sim-eval aid (NOT for the real robot): after a generated plan installs
    // from the motion inbox, hold the playback counter (don't advance
    // inference_tick) for this long so a slow external orchestrator in a separate
    // process isn't outrun by the tracker. 0 (default) = never pause, so the
    // real-robot and normal-sim paths are unaffected.
    std::chrono::nanoseconds sim_pause_after_plan{0};
};

enum class ControlLoopExit : std::uint8_t {
    kNormal         = 0,
    kStateTimeout   = 1,
    kSafetyLatched  = 2,
    kStateMachineDone = 3,
    kRequested      = 4,
    kMaxTicks       = 5,
    // Obs spec and engine inputs don't line up (a group matching no input, two
    // groups on one input, or an input no group fills). Detected in the ctor;
    // run() refuses to start (no tick, no command) rather than ticking with a
    // zeroed action or the wrong tensor. See the ctor for the diagnosis.
    kConfigError    = 6,
    // No live odometry within odom_ready_timeout_ns of the first state: the
    // policy has no anchor frame, so the loop never ticks.
    kOdomTimeout    = 7,
    // The policy engine failed kEngineFailStopTicks inferences in a row (e.g. an
    // ONNX/plan whose output does not match the 29-D action); the loop held the
    // default pose and stopped.
    kEngineFailure  = 8,
};

// Non-owning bag of references plumbed through the loop.
struct ControlLoopDeps {
    robot::IRobotBackend<kDof>&    backend;
    inference::IInferenceEngine&   engine;
    ObservationBuilder&            observation_builder;
    ActionDecoder&                 action_decoder;
    StateMachine&                  state_machine;
    robot::SafetyMonitor<kDof>&    safety;
    core::JitterHistogram&         jitter;
    // Optional non-owning pointer to the (mutable) motion data. When set,
    // the loop yaw-aligns the motion to the robot's current orientation (and
    // anchors its XY on odometry) exactly once on the kDefaultPos → kControl
    // transition. Pass nullptr to disable alignment (e.g. for a sim app whose
    // initial pose already matches the motion's authored yaw).
    //
    // Ignored when `motion_switcher` is non-null — the switcher takes over
    // CONTROL-entry alignment along with the runtime standup/replay swaps.
    policy::MotionData*            mutable_motion = nullptr;

    // Optional motion switcher. Enables operator-driven motion swaps inside
    // CONTROL via the state machine's button-edge flags:
    //   - B rising → standup_requested → switch to standing motion (and, in
    //     multi-process mode where original==nullptr, latch tracker_ready_out)
    //   - A rising → motion_replay_requested → switch back to original motion
    //     (single-process only; a no-op when original==nullptr)
    // Both switches yaw-align the chosen motion to the robot's current
    // orientation, swap the observation builder's motion pointer, and reset
    // the inference tick so the new motion plays from frame 0. Mirrors the
    // Python `do_standup` / `replay_original_motion` paths in deploy_real.py.
    //
    // When non-null, the switcher also handles the kDefaultPos → kControl
    // yaw-align (subsumes `mutable_motion`). Pass nullptr to use the
    // `mutable_motion` align path or to opt out entirely.
    MotionSwitcher*                motion_switcher = nullptr;

    // Optional outbox the loop publishes `state_` into once per tick. Used by
    // transports (e.g. the TrackerServer in system/cpp) to answer state queries
    // from external clients without contending the backend's own SPSC state
    // triple buffer. Must outlive the ControlLoop. Defaults nullptr.
    core::TripleBuffer<math::RobotState<kDof>>* state_outbox = nullptr;

    // Optional inbox for staged raw motion plans. Transport-neutral: any
    // producer (network transport, in-process driver) that wants
    // to push a new motion attaches here. Consumed only in CONTROL with a
    // motion_switcher attached, after the command has been sent. When non-null
    // AND no operator swap fired this tick (CONTROL-entry align, B-edge
    // standup, A-edge replay), the loop calls `try_acquire_latest()`; on a hit
    // it installs the plan via g1::send_ref_motion (resample, joint remap,
    // slice, derivatives, anchor rebase) into its `processed_motion_scratch_`
    // (~100 µs for a 5 s / 150-frame plan), points the observation builder at
    // it, and resets the inference tick to 0.
    //
    // Operator-wins precedence: when an operator action fires, the inbox is
    // skipped this tick; the buffered plan stays available for a later tick
    // (or gets overwritten by a newer one). Must outlive the ControlLoop.
    // Defaults nullptr.
    //
    // ControlLoopConfig::motion_scratch_capacity MUST be > kMaxTransitionSteps
    // when this pointer is non-null.
    core::TripleBuffer<policy::RawMotionPlan>*  motion_inbox = nullptr;

    // Optional caller-owned mirror of the loop's motion-reference tick. When
    // non-null, the control loop publishes every increment / reset of its
    // internal counter into this atomic (release ordering). Off-loop readers
    // (a transport's state/trim handlers) consume via
    // acquire. Defaults nullptr.
    std::atomic<std::int32_t>*                  inference_tick_out = nullptr;

    // Outbox for the active plan snapshot — published when the active motion
    // changes, NOT every tick. When non-null, the control loop copies the
    // post-processed active motion (resampled, aligned, training-order joints)
    // into the slot at: the initial bind, CONTROL entry, B-edge standup /
    // A-edge replay, a motion-inbox install, and a trim commit. Each publish
    // carries `install_tick` (always 0: snapshot row 0 is live-window row 0)
    // so the transport thread can compute its index in the snapshot via
    //   idx = inference_tick_mirror.load() - snap.install_tick
    // without locks. Must outlive the ControlLoop. Defaults nullptr.
    core::TripleBuffer<ActivePlanSnapshot>*     active_plan_snapshot_out = nullptr;

    // Atomic flag the trim initiator sets to true AFTER it has already
    // computed and replied to the trim request from the snapshot. The control
    // loop sees the flag on its next tick and commits the side-effect
    // (policy::commit_trim_cut): advances the active motion's start_row,
    // shrinks frame_count, re-bases the inference tick onto the rows already
    // played since the cut, and re-publishes the snapshot. A cut larger than
    // the current tick (read from a plan that has since been replaced) is
    // dropped. The flag is then cleared. Defaults nullptr.
    std::atomic<bool>*                          trim_install_pending = nullptr;
    // Companion atomic carrying the index into the live window where the
    // transport cut. Read when servicing the install flag. Defaults nullptr.
    std::atomic<std::int32_t>*                  trim_install_tick = nullptr;

    // Optional latched tracker-ready output. When non-null, the control loop
    // sets it to true (latched, never cleared) when the operator presses B
    // while in kControl in multi-process mode (original motion == nullptr; real
    // robot only). Read by the transport (TrackerServer) to populate
    // RobotState.tracker_ready so the generator knows when to start. Sim
    // trackers initialize it true at startup. Defaults nullptr.
    std::atomic<bool>*                          tracker_ready_out = nullptr;

    // Optional gamepad mirror: the 16 wireless-remote button bits (key_map order, bit i = button
    // i) stored once per tick right after poll_remote. Read by the transport to publish
    // RobotState.remote_buttons, so off-robot nodes (e.g. an orchestrator) can react to the gamepad
    // (e.g. a hold / resume button). nullptr = not mirrored. Relaxed store, no RT cost.
    std::atomic<std::uint32_t>*                 remote_buttons_out = nullptr;

    // Optional outbox for the reference frame the active motion holds at the
    // current tick (position, wxyz quaternion, training-order joints), published
    // once per tick in every state. Lets a transport report the reference (e.g. a
    // viewer ghost) without reading the motion buffers the control thread is
    // installing into and trimming. SPSC: one consumer thread. nullptr = off.
    core::TripleBuffer<ExecutedRefEntry>*       ref_outbox = nullptr;

    // Optional inbox for the latest depth frame (depth-student `student_depth`).
    // The loop drains it WAIT-FREE each tick (try_acquire_latest) and hands the
    // freshest frame to the ObservationBuilder's image group before build().
    // nullptr => no depth: a config with an image group is refused at start
    // (kConfigError) unless ControlLoopConfig::dry_run. Must outlive the
    // ControlLoop. Nothing on this RT thread copies, serializes or publishes a
    // depth frame; it only reads the newest slot.
    // Attached + image group (not dry run): A in DEFAULT_POS is ignored until a
    // live frame exists (none received yet, or on the stale far-plane fallback),
    // so the policy never starts on a blank image.
    //
    // Who fills it depends on the backend, and they are NOT equivalent:
    //   real — the RealSense reader, a genuinely separate non-RT thread. The
    //          loop pays only the try_acquire_latest.
    //   sim  — the MuJoCo backend (deploy_g1_mujoco --depth_render) renders the
    //          depth camera INLINE in step(), every obs_decim steps, on this
    //          control thread, so it DOES add per-tick work to the loop;
    //          acceptable in sim (no robot to fall over) but sim loop timing is
    //          then not a proxy for real.
    core::TripleBuffer<DepthFrame>*             depth_inbox = nullptr;

    // Optional per-iteration timing trace. When non-null, the loop records one
    // CSV row per control tick (period, total work time, and the obs / infer /
    // act phase split) via a lock-free push to the trace's writer thread. Costs
    // ~8 clock_gettime calls per tick (~200 ns on Orin) plus one ring push;
    // nothing on the hot path allocates, locks, or does I/O. Must be created on
    // a NON-RT thread (run_control_loop does) so its writer thread can never
    // preempt the control thread, and must outlive the ControlLoop.
    // Defaults nullptr — zero cost (one branch per tick) when unset.
    core::LoopTrace*                            loop_trace = nullptr;
};

class ControlLoop {
public:
    ControlLoop(const ControlLoopConfig& cfg, ControlLoopDeps deps) noexcept;

    void request_stop() noexcept { stop_.store(true, std::memory_order_release); }

    // Blocks in the current thread. Configures RT priority + pinning as its
    // first action. Returns the reason for exit.
    [[nodiscard]] ControlLoopExit run() noexcept;

    [[nodiscard]] std::int64_t tick_count() const noexcept {
        return tick_count_.load(std::memory_order_acquire);
    }

    // Current motion-reference tick (count of inferences run since the last
    // motion swap, equal to Python `controller.counter`). Atomic so off-loop
    // threads (trim handler) can read it cheaply.
    [[nodiscard]] std::int32_t current_tick() const noexcept {
        return inference_tick_.load(std::memory_order_acquire);
    }

    // Plan-install counters, for a transport that acknowledges a plan only once the loop has
    // acted on it: publish the plan, then wait (bounded) until plans_consumed() moves past its
    // value before the publish; the plan became the active reference iff plans_installed() moved
    // too. A plan published outside CONTROL is not consumed until CONTROL is entered.
    [[nodiscard]] std::uint64_t plans_consumed() const noexcept {
        return plans_consumed_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t plans_installed() const noexcept {
        return plans_installed_.load(std::memory_order_acquire);
    }

    // Pointer to the motion the observation builder is currently sourcing
    // command features from. May be nullptr if no motion is wired. CONTROL THREAD
    // ONLY: the loop rewrites this motion in place (plan install, trim commit,
    // alignment); other threads read the executed reference through ref_outbox.
    [[nodiscard]] const policy::MotionData* active_motion() const noexcept {
        return deps_.observation_builder.motion();
    }

    // Print min/p50/p99/max of the per-swap error metrics collected during
    // run(). No-op when no swaps have occurred. Called once automatically at
    // end of run(); exposed for callers that want it on a different stream.
    void print_swap_summary(std::FILE* out = stderr) const noexcept;
    // Depth freshness over the whole run: frames consumed (distinct producer
    // frames; at 50 Hz control and 30 Hz depth the loop re-reads frames), the
    // longest stall (healthy steady state: 1-2 ticks), and how many times the
    // far-plane fallback engaged. No-op without a depth inbox.
    void print_depth_summary(std::FILE* out = stderr) const noexcept;

private:
    ControlLoopExit run_body() noexcept;

    // Opt-in per-N-inference-tick dump of the ACTUAL tensor values crossing the
    // engine boundary: the built `student_depth` group tensor (post-encode), the
    // raw metric DepthFrame behind it, and the returned action vector. Called
    // only when cfg_.depth_stats_every > 0 (see that field for the cost and the
    // stderr-blocking caveat). const because it only reads loop state.
    //
    // `tick_mono_ns` is this tick's wake-up timestamp, passed in rather than
    // re-read so the diagnostic adds no clock read of its own.
    void print_depth_stats(std::int64_t               inference_tick,
                           std::int64_t               tick_mono_ns,
                           inference::InferenceStatus infer_status) const noexcept;


    ControlLoopConfig                  cfg_;
    ControlLoopDeps                    deps_;
    std::atomic<bool>                  stop_{false};
    std::atomic<std::int64_t>          tick_count_{0};
    // Motion-reference tick. Atomic so the trim handler can read it without
    // coordinating with the control thread.
    // Written only on the control thread (release); read on any thread.
    std::atomic<std::int32_t>          inference_tick_{0};

    // Hot-path scratch buffers (preallocated, reused).
    math::RobotState<kDof>             state_{};
    robot::RemoteState                 remote_{};
    // Latest depth frame drained from deps_.depth_inbox, held across ticks (a
    // tick with no new frame reuses the last; nullptr => image group far fallback).
    const DepthFrame*                  last_depth_{nullptr};
    // Depth staleness bookkeeping. DepthFrame::seq is the producer's monotonic
    // frame id; when it stops advancing the camera or the depth thread has died
    // and the policy is consuming a frozen image.
    //
    // A SHORT gap keeps the last-known-good frame: the policy trained on 40-120 ms
    // of staleness, so a slightly old image is in-distribution and blanking on
    // every dropped camera frame would be worse than holding.
    //
    // A LONG gap (cfg_.depth_stale_far_ticks) switches to the far plane. Past
    // that point the frame is not "slightly old", it is fiction, and the policy
    // steering around geometry that has gone is a failure with no other symptom.
    // Far is the honest encoding of "no information" and the one training used for
    // every dropped pixel. Both stages are announced once per stall; neither
    // blocks or skips a tick. Written only by the control thread.
    std::int64_t                       last_depth_seq_{-1};
    std::int32_t                       depth_stale_ticks_{0};      // since seq advanced
    std::int32_t                       depth_max_stale_ticks_{0};  // worst seen, for the summary
    std::int64_t                       depth_frames_seen_{0};
    bool                               depth_stale_warned_{false};
    bool                               depth_none_warned_{false};  // CONTROL before the first frame
    // True while the far-plane fallback is engaged (see depth_stale_far_ticks).
    bool                               depth_far_fallback_{false};
    // How many times the fallback engaged over the run. Nonzero means the policy
    // ran on an empty-world image for a while, which belongs in the run summary
    // rather than only in a scrolled-past stderr line.
    std::int64_t                       depth_far_fallback_events_{0};
    // CONTROL-entry gate (tick step 2b): A is masked in DEFAULT_POS until every
    // input the policy needs is live (odometry always, plus a depth frame for an
    // image group). A stale odometry stream in CONTROL latches the safety monitor
    // (SafetyFault::kStaleOdom). `wait_logged` = the refusal was printed for the
    // current not-live episode; `holding_a` = a refused A press is still held, so
    // it stays masked until released.
    bool                               entry_wait_logged_{false};
    bool                               entry_gate_holding_a_{false};
    // Consecutive failed inference ticks. A streak of kEngineFailStopTicks
    // (0.5 s at 50 Hz) stops the run instead of holding the default pose forever.
    static constexpr std::int32_t      kEngineFailStopTicks = 25;
    std::int32_t                       engine_fail_streak_{0};
    bool                               engine_failed_{false};
    std::atomic<std::uint64_t>         plans_consumed_{0};
    std::atomic<std::uint64_t>         plans_installed_{0};
    // Index into observation_builder.outputs() of the image ('student_depth')
    // group; -1 when the spec declares none. Resolved once in the ctor so the
    // depth_stats diagnostic can reach the built tensor by index — a per-tick
    // group("student_depth") lookup would construct a std::string temporary on
    // the RT thread, which is exactly the kind of "usually fine" allocation the
    // hot path is not allowed to contain.
    int                                depth_group_idx_{-1};
    // Seeded by the first inference before any CONTROL publish: inference runs
    // on every CONTROL tick, so cached_cmd_ always holds the command just
    // decoded this tick. It still seeds cmd_out before the state machine runs
    // (the state machine overwrites it in every non-CONTROL state).
    math::MotorCommand<kDof>           cached_cmd_{};
    // Policy output (training order) and its torque-clipped version. The
    // clipped action is what the next tick's observation feeds as last_action
    // (every tick, no first-step skip).
    ActionDecoder::JointVec            raw_action_train_{};
    ActionDecoder::JointVec            clipped_action_train_{
        ActionDecoder::JointVec::Zero()};

    // Per-group -> engine-input index map, and a reusable inputs vector sized
    // to engine.input_count().
    std::vector<int>                   group_engine_idx_{};
    // Set by the ctor when the obs spec and the engine's inputs are inconsistent.
    // run() returns kConfigError immediately rather than starting the robot.
    bool                               config_error_{false};
    std::vector<std::span<const float>> engine_inputs_{};

    // MotionData scratch the loop processes RawMotionPlan inbox payloads into.
    // Allocated once in the constructor when cfg_.motion_scratch_capacity > 0;
    // overwritten in place on each inbox hit. Lives in the loop (not the
    // inbox) because the inbox carries raw bytes — processing happens here.
    policy::MotionData                 processed_motion_scratch_{};

    // Per-swap error metrics, one entry pushed per successful inbox swap.
    // Summarized by print_swap_summary() at end of run(). Written + read only
    // on the control thread.
    std::vector<float>                 swap_joint_mean_;
    std::vector<float>                 swap_joint_l2_;
    std::vector<float>                 swap_base_xy_;
    std::vector<float>                 swap_base_z_;
};

}  // namespace rt_deploy::g1
