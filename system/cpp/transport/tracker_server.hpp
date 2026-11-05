//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// TrackerServer — the halo_system tracker-side transport node.
//
// Binds the four generator<->tracker channels on a halo_comm::Node (ROS2-style
// services + a topic, protobuf over ZMQ/TCP) and drives the tracker's control
// loop through its public TripleBuffer / RawMotionPlan / ActivePlanSnapshot /
// trim-atomic seam.
//
//   services (REQ/REP):
//     g1/get_state  StateRequest -> RobotState   (port 5555)
//     g1/put_plan   MotionPlan   -> PlanAck       (port 5556)
//     g1/get_trim   TrimRequest  -> TrimReply     (port 5557)
//   topic (PUB, 50 Hz, conflate):
//     g1/state      RobotState                    (port 5558)
//
// Threading: ONE server thread runs the unified loop — it is the SOLE consumer
// of `state_outbox` (refreshes a cached snapshot once per iteration; both the
// get_state handler and the 50 Hz publisher read that cache, so the SPSC
// TripleBuffer contract is never violated), and likewise of the reference
// outbox and the active-plan snapshot. It is the SOLE producer of
// `motion_inbox`. The 50 Hz control loop touches only TripleBuffers/atomics;
// protobuf (de)serialization happens here, off the control thread.

#include "halo_comm/node.hpp"
#include "motion.pb.h"
#include "state.pb.h"

#include "rt_deploy/core/rt_thread.hpp"
#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/g1/active_plan_snapshot.hpp"
#include "rt_deploy/g1/config.hpp"            // kDof
#include "rt_deploy/g1/control_loop.hpp"      // plans_consumed() / plans_installed() for the put_plan ack
#include "rt_deploy/g1/depth_frame.hpp"
#include "rt_deploy/g1/executed_ref_ring.hpp"
#include "rt_deploy/math/types.hpp"
#include "rt_deploy/policy/joint_remap.hpp"
#include "rt_deploy/policy/motion_loader.hpp"
#include "rt_deploy/policy/raw_motion_plan.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace halo_system {

namespace rd = rt_deploy;

struct TrackerServerConfig {
    std::string   bind_ip        = "0.0.0.0";
    int           state_port     = 5555;
    int           plan_port      = 5556;
    int           trim_port      = 5557;
    int           state_pub_port = 5558;   // 0 disables the topic
    int           state_pub_hz   = 50;     // matches the 50 Hz control loop
    int           poll_timeout_ms = 5;     // <= pub period so cadence holds
    float         fallback_base_height_m = 0.75F;  // when odom invalid

    // Optional ego-camera render callback (transport-neutral). When set, the get_state REP
    // reply carries an Image trailer (never the topic). Fills `rgb` (w*h*3
    // top-down RGB) and writes w/h; returns false to omit the image. Runs on
    // the server thread; the MuJoCo backend's render_ego_image is thread-safe.
    using RenderImageFn =
        std::function<bool(std::vector<std::uint8_t>& rgb, int& w, int& h)>;
    RenderImageFn render_image{};

    // Optional latched tracker-ready flag. When non-null, fill_robot_state()
    // reads it and sets RobotState.tracker_ready. The control loop sets it to
    // true (latched) when the operator presses A in CONTROL (real robot), or
    // it's initialized true for the sim tracker. The generator polls
    // get_state and waits for this before starting generation.
    const std::atomic<bool>* tracker_ready = nullptr;

    // Optional gamepad mirror written by the control loop (ControlLoopDeps::remote_buttons_out): the
    // wireless-remote button bits, published as RobotState.remote_buttons (the orchestrator's X hold).
    const std::atomic<std::uint32_t>* remote_buttons = nullptr;

    // Optional loop pointer, set by run_control_loop(control_loop_out). When set, put_plan replies only
    // after the loop took the plan from the inbox (normally <= one 20 ms tick), so a get_trim sent after
    // the reply always sees the new plan; outside CONTROL it gives up after kPlanAckWaitMs ("queued").
    rd::g1::ControlLoop* const* control_loop = nullptr;

    // Optional g1/depth topic (the depth policy's input frame, for the web viewer). When
    // `depth_provider` is set and depth_pub_port > 0, each NEW frame (by seq) is published at
    // up to depth_pub_hz on this server thread. The provider returns a COPY of the latest frame.
    using DepthProviderFn = std::function<bool(rd::g1::DepthFrame&)>;
    DepthProviderFn depth_provider{};
    int             depth_pub_port = 0;    // 0 disables the topic
    int             depth_pub_hz   = 10;
};

class TrackerServer {
public:
    // `state_outbox` : control loop publishes its state here once per tick;
    //                  this server is the sole consumer.
    // `motion_inbox` : this server is the sole producer of raw plans.
    // `snapshot_in` / `trim_install_pending` / `trim_install_tick` : the RTC
    //                  trim seam.
    // `current_tick` : the loop's motion-reference tick mirror.
    // `ref_in`       : the loop's per-tick reference outbox
    //                  (ControlLoopDeps::ref_outbox); this server is the sole
    //                  consumer.
    // All seam pointers must be non-null.
    TrackerServer(
        rd::core::TripleBuffer<rd::math::RobotState<rd::g1::kDof>>& state_outbox,
        rd::core::TripleBuffer<rd::policy::RawMotionPlan>&          motion_inbox,
        rd::core::TripleBuffer<rd::g1::ActivePlanSnapshot>*         snapshot_in,
        const std::atomic<std::int32_t>*                           current_tick,
        std::atomic<bool>*                                         trim_install_pending,
        std::atomic<std::int32_t>*                                 trim_install_tick,
        const rd::policy::JointRemap<rd::g1::kDof>&                remap,
        rd::core::TripleBuffer<rd::g1::ExecutedRefEntry>*          ref_in,
        TrackerServerConfig                                        cfg)
        : state_outbox_(state_outbox),
          motion_inbox_(motion_inbox),
          snapshot_in_(snapshot_in),
          current_tick_(current_tick),
          trim_install_pending_(trim_install_pending),
          trim_install_tick_(trim_install_tick),
          remap_(remap),
          ref_in_(ref_in),
          cfg_(std::move(cfg)),
          node_("tracker") {
        node_.create_service<motion_msgs::StateRequest, motion_msgs::RobotState>(
            "g1/get_state", cfg_.state_port,
            [this](const motion_msgs::StateRequest& req, motion_msgs::RobotState& out) {
                on_get_state(req, out);
            },
            cfg_.bind_ip);
        node_.create_service<motion_msgs::MotionPlan, motion_msgs::PlanAck>(
            "g1/put_plan", cfg_.plan_port,
            [this](const motion_msgs::MotionPlan& req, motion_msgs::PlanAck& out) {
                on_put_plan(req, out);
            },
            cfg_.bind_ip);
        node_.create_service<motion_msgs::TrimRequest, motion_msgs::TrimReply>(
            "g1/get_trim", cfg_.trim_port,
            [this](const motion_msgs::TrimRequest& req, motion_msgs::TrimReply& out) {
                on_get_trim(req, out);
            },
            cfg_.bind_ip);
        if (cfg_.state_pub_port > 0) {
            state_pub_ = node_.create_publisher<motion_msgs::RobotState>(
                "g1/state", cfg_.state_pub_port, cfg_.bind_ip);
        }
        if (cfg_.depth_pub_port > 0 && static_cast<bool>(cfg_.depth_provider)) {
            depth_pub_ = node_.create_publisher<motion_msgs::DepthImage>(
                "g1/depth", cfg_.depth_pub_port, cfg_.bind_ip);
        }
    }

    void start() {
        running_.store(true, std::memory_order_release);
        thread_ = std::thread([this]() { run(); });
    }
    void stop() {
        running_.store(false, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }
    ~TrackerServer() { stop(); }

    [[nodiscard]] std::uint64_t state_queries()  const noexcept { return state_queries_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t plans_received() const noexcept { return plans_received_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t trim_requests()  const noexcept { return trim_requests_.load(std::memory_order_relaxed); }

private:
    static constexpr int kDof = static_cast<int>(rd::g1::kDof);
    static constexpr int kPlanAckWaitMs = 60;   // put_plan: max wait for the install

    void run() {
        rd::core::RtThreadConfig tc;
        tc.name = "halo_comm-srv";
        (void)rd::core::configure_current_thread(tc);

        using clock = std::chrono::steady_clock;
        const int  hz = std::max(1, cfg_.state_pub_hz);
        const auto pub_period = std::chrono::nanoseconds(static_cast<long long>(1e9) / hz);
        auto       next_pub = clock::now();
        const bool pub_on = state_pub_.has_value();
        const auto depth_period = std::chrono::nanoseconds(
            static_cast<long long>(1e9) / std::max(1, cfg_.depth_pub_hz));
        auto               next_depth     = clock::now();
        std::int64_t       last_depth_seq = -1;
        rd::g1::DepthFrame depth_frame;

        while (running_.load(std::memory_order_acquire)) {
            // Refresh the cached state — sole consumer of state_outbox.
            if (const auto* s = state_outbox_.try_acquire_latest(); s != nullptr) {
                cached_state_ = *s;
                have_cached_  = true;
            }
            // Same for the reference outbox (sole consumer).
            if (const auto* r = ref_in_->try_acquire_latest(); r != nullptr) {
                cached_ref_ = *r;
                have_ref_   = true;
            }
            node_.spin_once(cfg_.poll_timeout_ms);   // dispatch ready REP requests
            if (pub_on) {
                const auto now = clock::now();
                if (now >= next_pub) {
                    next_pub = std::max(next_pub + pub_period, now);
                    motion_msgs::RobotState msg;
                    fill_robot_state(msg, /*with_image=*/false);  // topic never carries images
                    state_pub_->publish(msg);
                }
            }
            if (depth_pub_.has_value()) {
                const auto now = clock::now();
                if (now >= next_depth) {
                    next_depth = std::max(next_depth + depth_period, now);
                    if (cfg_.depth_provider(depth_frame) && depth_frame.seq != last_depth_seq) {
                        last_depth_seq = depth_frame.seq;
                        motion_msgs::DepthImage dmsg;
                        dmsg.set_width(rd::g1::DepthFrame::kCols);
                        dmsg.set_height(rd::g1::DepthFrame::kRows);
                        dmsg.set_data(depth_frame.data.data(),
                                      depth_frame.data.size() * sizeof(float));
                        dmsg.set_tick(static_cast<std::int32_t>(depth_frame.seq));
                        depth_pub_->publish(dmsg);
                    }
                }
            }
        }
    }

    // Service handlers.
    void on_get_state(const motion_msgs::StateRequest& /*req*/,
                       motion_msgs::RobotState&         out) {
        fill_robot_state(out, /*with_image=*/true);
        state_queries_.fetch_add(1, std::memory_order_relaxed);
    }

    // Pack the cached robot state (+ optional ref-motion trailer) into a
    // protobuf RobotState.
    void fill_robot_state(motion_msgs::RobotState& out, bool with_image) {
        out.Clear();
        if (have_cached_) {
            const auto& s = cached_state_;
            for (int i = 0; i < kDof; ++i)
                out.add_joint_pos(static_cast<double>(s.joint_positions[i]));
            // Publish the orientation source the server's ref rebase uses
            // (odom_quaternion_wxyz when valid) so the viewer
            // renders the robot and the ref_motion ghost in the SAME frame
            // — otherwise the imu↔odom yaw bias (~16-20° on a real G1) shows
            // up as a constant offset between the robot pose and the ref
            // ghost in the viewer, even when the policy is tracking
            // perfectly. The wire field keeps its name `imu_quat_wxyz` for
            // backward compat; the source is what changed. On MuJoCo both
            // fields alias the same floating-base qpos so this is bit-
            // identical there.
            const auto& q_pub = s.odom_valid ? s.odom_quaternion_wxyz
                                              : s.imu_quat_wxyz;
            for (int i = 0; i < 4; ++i)
                out.add_imu_quat_wxyz(static_cast<double>(q_pub[i]));
            if (s.odom_valid) {
                for (int i = 0; i < 3; ++i)
                    out.add_base_pos(static_cast<double>(s.odom_position[i]));
            } else {
                out.add_base_pos(0.0);
                out.add_base_pos(0.0);
                out.add_base_pos(static_cast<double>(cfg_.fallback_base_height_m));
            }
        } else {
            out.add_base_pos(0.0);
            out.add_base_pos(0.0);
            out.add_base_pos(static_cast<double>(cfg_.fallback_base_height_m));
        }
        out.set_tick(current_tick_->load(std::memory_order_acquire));
        if (cfg_.tracker_ready != nullptr
            && cfg_.tracker_ready->load(std::memory_order_acquire)) {
            out.set_tracker_ready(true);
        }
        if (cfg_.remote_buttons != nullptr) {
            out.set_remote_buttons(cfg_.remote_buttons->load(std::memory_order_relaxed));
        }
        fill_ref_motion(out);

        // Optional ego-camera Image trailer — only on the REP get_state path
        // (with_image), never the topic. The render callback (set by the node
        // from the MuJoCo backend) returns top-down w*h*3 RGB.
        if (with_image && cfg_.render_image) {
            int  w = 0, h = 0;
            bool ok = false;
            try {
                ok = cfg_.render_image(image_buf_, w, h);
            } catch (...) {
                ok = false;
            }
            if (ok && w > 0 && h > 0
                && image_buf_.size() == static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3) {
                auto* img = out.mutable_image();
                img->set_width(w);
                img->set_height(h);
                img->set_channels(3);
                img->set_rgb(image_buf_.data(), image_buf_.size());
            }
        }
    }

    // Optional ghost / reference-motion trailer: the reference row the control
    // loop published for its latest tick (ControlLoopDeps::ref_outbox). Read from
    // that copy, never from the loop's motion buffers, which the control thread
    // rewrites on every plan install and trim.
    void fill_ref_motion(motion_msgs::RobotState& out) {
        if (!have_ref_) return;
        const auto& e   = cached_ref_;
        auto*       ref = out.mutable_ref_motion();
        for (int i = 0; i < 3; ++i)
            ref->add_pos(static_cast<double>(e.pos[i]));
        for (int i = 0; i < 4; ++i)
            ref->add_quat_wxyz(static_cast<double>(e.quat_wxyz[i]));
        // dof_train is training order; convert to deployment order for the wire
        // so ref_motion.joints align with the actual-state joint_pos (the viewer's
        // ref ghost uses train_to_deploy too).
        for (int d = 0; d < kDof; ++d) {
            const std::int32_t t = remap_.train_to_deploy[static_cast<std::size_t>(d)];
            ref->add_joints(static_cast<double>(e.dof_train[static_cast<std::size_t>(t)]));
        }
    }

    // MotionPlan -> RawMotionPlan inbox slot.
    void on_put_plan(const motion_msgs::MotionPlan& req, motion_msgs::PlanAck& out) {
        const std::int32_t cols      = req.cols();
        const std::int32_t input_fps = req.input_fps();
        const std::uint32_t flags    = req.flags();
        const int          n_vals    = req.values_size();
        if (cols != rd::policy::RawMotionPlan::kCols || (n_vals % cols) != 0) {
            out.set_ok(false); out.set_reason("invalid cols / values length"); return;
        }
        const std::int32_t n_rows = static_cast<std::int32_t>(n_vals / cols);
        if (n_rows < 1 || input_fps <= 0) {
            out.set_ok(false); out.set_reason("invalid plan header"); return;
        }
        if (n_rows > rd::policy::RawMotionPlan::kMaxRows) {
            out.set_ok(false); out.set_reason("plan exceeds RawMotionPlan::kMaxRows"); return;
        }
        rd::g1::ControlLoop* loop = cfg_.control_loop != nullptr ? *cfg_.control_loop : nullptr;
        const std::uint64_t consumed0  = loop != nullptr ? loop->plans_consumed()  : 0;
        const std::uint64_t installed0 = loop != nullptr ? loop->plans_installed() : 0;
        auto& slot     = motion_inbox_.producer_slot();
        slot.n_rows    = n_rows;
        slot.input_fps = input_fps;
        slot.flags     = flags;
        for (int i = 0; i < n_vals; ++i)
            slot.data[static_cast<std::size_t>(i)] = req.values(i);
        motion_inbox_.publish();
        plans_received_.fetch_add(1, std::memory_order_relaxed);

        out.set_ok(true);
        // Ack after install (see TrackerServerConfig::control_loop). REP serves one request at a time, so
        // the next consumption is this plan (or a newer one that replaced it).
        if (loop == nullptr) return;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kPlanAckWaitMs);
        while (loop->plans_consumed() == consumed0) {
            if (std::chrono::steady_clock::now() >= deadline) {
                out.set_reason("queued: not installed yet (tracker not in CONTROL?)");
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(250));
        }
        if (loop->plans_installed() == installed0) {
            out.set_ok(false);
            out.set_reason("rejected by the tracker (start_offset >= plan length?)");
        }
    }

    // TrimRequest -> TrimReply, serviced locally from the active-plan snapshot.
    void on_get_trim(const motion_msgs::TrimRequest& /*req*/,
                      motion_msgs::TrimReply&         out) {
        trim_requests_.fetch_add(1, std::memory_order_relaxed);
        out.set_cols(rd::policy::RawMotionPlan::kCols);

        const auto* snap = snapshot_in_->try_acquire_latest();
        if (snap == nullptr || snap->epoch == 0 || snap->frame_count == 0) {
            out.set_input_fps(0);
            return;
        }
        const std::int32_t tick = current_tick_->load(std::memory_order_acquire);
        const std::int32_t idx_unclamped = tick - snap->install_tick;
        const std::int32_t idx =
            idx_unclamped < 0 ? 0
            : (idx_unclamped > snap->frame_count ? snap->frame_count : idx_unclamped);

        const std::int32_t target_input_fps =
            snap->input_fps > 0 ? snap->input_fps : snap->output_fps;
        const rd::policy::RawMotionPlan reply =
            rd::policy::trim_ref_motion_from_snapshot(*snap, idx, target_input_fps, snap->remap);
        if (reply.n_rows == 0) {
            // Refused (a truncated snapshot with the counter past its end): reply
            // empty and commit nothing.
            std::fprintf(stderr, "[tracker_server] trim refused: counter %d is past the %d-row "
                                 "active-plan snapshot\n", tick, snap->frame_count);
            out.set_input_fps(0);
            return;
        }

        // Signal the install side-effect for the loop's next tick.
        trim_install_tick_->store(idx, std::memory_order_release);
        trim_install_pending_->store(true, std::memory_order_release);

        const std::int32_t n_rows =
            std::min<std::int32_t>(reply.n_rows, rd::policy::RawMotionPlan::kMaxRows);
        out.set_input_fps(reply.input_fps);
        const int n_vals = n_rows * rd::policy::RawMotionPlan::kCols;
        out.mutable_values()->Reserve(n_vals);
        for (int i = 0; i < n_vals; ++i)
            out.add_values(reply.data[static_cast<std::size_t>(i)]);
    }

    rd::core::TripleBuffer<rd::math::RobotState<rd::g1::kDof>>& state_outbox_;
    rd::core::TripleBuffer<rd::policy::RawMotionPlan>&          motion_inbox_;
    rd::core::TripleBuffer<rd::g1::ActivePlanSnapshot>*         snapshot_in_;
    const std::atomic<std::int32_t>*                           current_tick_;
    std::atomic<bool>*                                         trim_install_pending_;
    std::atomic<std::int32_t>*                                 trim_install_tick_;
    rd::policy::JointRemap<rd::g1::kDof>                       remap_;
    rd::core::TripleBuffer<rd::g1::ExecutedRefEntry>*          ref_in_;
    TrackerServerConfig                                        cfg_;

    halo_comm::Node                                          node_;
    std::optional<halo_comm::Publisher<motion_msgs::RobotState>> state_pub_;
    std::optional<halo_comm::Publisher<motion_msgs::DepthImage>> depth_pub_;
    std::thread                                           thread_;
    std::atomic<bool>                                     running_{false};

    rd::math::RobotState<rd::g1::kDof> cached_state_{};
    bool                               have_cached_{false};
    rd::g1::ExecutedRefEntry           cached_ref_{};
    bool                               have_ref_{false};
    // Reused ego-render RGB scratch (server thread only; no lock needed).
    std::vector<std::uint8_t>          image_buf_;

    std::atomic<std::uint64_t> state_queries_{0};
    std::atomic<std::uint64_t> plans_received_{0};
    std::atomic<std::uint64_t> trim_requests_{0};
};

}  // namespace halo_system
