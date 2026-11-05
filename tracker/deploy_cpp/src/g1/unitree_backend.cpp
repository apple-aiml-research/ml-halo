//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// UnitreeBackend — real-robot backend over Unitree SDK2 DDS for the G1.
//
// Verified against upstream Unitree SDK2:
//   /unitree_sdk2/include/unitree/robot/channel/channel_publisher.hpp
//   /unitree_sdk2/include/unitree/robot/channel/channel_subscriber.hpp
//   /unitree_sdk2/include/unitree/robot/channel/channel_factory.hpp
//   /unitree_sdk2/include/unitree/idl/hg/LowCmd_.hpp
//   /unitree_sdk2/include/unitree/idl/hg/LowState_.hpp
//   /unitree_sdk2/include/unitree/dds_wrapper/common/crc.h
//   /unitree_sdk2/example/g1/low_level/g1_ankle_swing_example.cpp
//
// Topic convention (HG / G1):
//   rt/lowcmd    — publish  — LowCmd_  (sizeof(LowCmd_) / 4 - 1 CRC words)
//   rt/lowstate  — subscribe — LowState_ (CRC-validated inbound)
//
// Thread model:
//   The SDK invokes the subscriber callback on a DDS thread (serialized per
//   topic). We use a TripleBuffer<RobotState<29>> and a TripleBuffer<RemoteState>
//   to hand off to the control thread wait-free. ChannelPublisher::Write is
//   documented thread-safe, so the control thread publishes directly on tick.
//
// Hot path notes:
//   - send_command() writes into a persistent LowCmd_ member, computes CRC
//     with crc32_core(), and calls Write(). No heap.
//   - read_state() drains the latest state snapshot via TripleBuffer::try_acquire_latest.
//   - poll_remote() drains the latest remote snapshot identically.
//   - Both return ReadStatus so the control loop + safety monitor can reason
//     about staleness.

#include "rt_deploy/g1/unitree_backend.hpp"

#include "rt_deploy/core/clock.hpp"
#include "rt_deploy/core/rate_probe.hpp"
#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/robot/remote_controller.hpp"

#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace rt_deploy::g1 {

namespace {

using LowCmdHg         = unitree_hg::msg::dds_::LowCmd_;
using LowStateHg       = unitree_hg::msg::dds_::LowState_;
using SportModeStateGo = unitree_go::msg::dds_::SportModeState_;

constexpr std::size_t kUnitreeHgMotorSlots = 35;  // LowCmd_.motor_cmd array size
constexpr std::size_t kWirelessRemoteBytes  = 40;
constexpr const char* kCmdTopic   = "rt/lowcmd";
constexpr const char* kStateTopic = "rt/lowstate";
// Sport-mode state estimator topic (odometry). The robot only publishes here
// when the high-level sport mode is loaded; without it the odom_* fields stay
// invalid and the control loop refuses to run (the policy anchors on odometry).
constexpr const char* kOdomTopic  = "rt/odommodestate";

// Snapshot we hand off from the SportModeState_ DDS thread to the control
// thread. Keeps the merge in read_state() trivial and the producer side
// allocation-free.
struct OdomSnapshot {
    bool            valid{false};
    std::int64_t    rx_ns{0};   // MonotonicClock receive time, for the staleness check
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    Eigen::Vector3f velocity{Eigen::Vector3f::Zero()};
    Eigen::Vector4f quaternion_wxyz{1.0F, 0.0F, 0.0F, 0.0F};
    float           yaw_speed{0.0F};
};

// CRC covers every 32-bit word of LowCmd_ except the trailing crc_ field.
// Matches the formula used by every upstream example:
//   Crc32Core((uint32_t*)&cmd, (sizeof(cmd) >> 2) - 1)
[[nodiscard]] std::uint32_t compute_low_cmd_crc(const LowCmdHg& cmd) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* ptr = const_cast<std::uint32_t*>(reinterpret_cast<const std::uint32_t*>(&cmd));
    const std::uint32_t word_count
        = static_cast<std::uint32_t>(sizeof(LowCmdHg) / sizeof(std::uint32_t)) - 1U;
    return ::crc32_core(ptr, word_count);  // global scope per SDK header
}

[[nodiscard]] bool verify_low_state_crc(const LowStateHg& state) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* ptr = const_cast<std::uint32_t*>(reinterpret_cast<const std::uint32_t*>(&state));
    const std::uint32_t word_count
        = static_cast<std::uint32_t>(sizeof(LowStateHg) / sizeof(std::uint32_t)) - 1U;
    return state.crc() == ::crc32_core(ptr, word_count);
}

class UnitreeBackendImpl final : public robot::IRobotBackend<kDof> {
public:
    UnitreeBackendImpl(std::string network_interface, int domain, float damping_kd,
                        bool verify_inbound_crc, bool send_enabled, bool subscribe_only,
                        std::int64_t odom_stale_ns) noexcept
        : network_interface_{std::move(network_interface)},
          domain_{domain},
          damping_kd_{damping_kd},
          verify_inbound_crc_{verify_inbound_crc},
          odom_stale_ns_{odom_stale_ns},
          send_enabled_{send_enabled && !subscribe_only},
          subscribe_only_{subscribe_only} {
        // Reset the persistent LowCmd_ to a default-constructed state so the
        // first publish is well-defined. (Don't memset — LowCmd_ is a
        // generated DDS class with non-trivial members.) Enable all 35 motor
        // slots (mode=1 per the G1 example); unused slots past 29 keep
        // kp=0, kd=0, q=0 so they're inert.
        last_cmd_ = LowCmdHg{};
        auto& mc = last_cmd_.motor_cmd();
        for (std::size_t i = 0; i < kUnitreeHgMotorSlots; ++i) {
            mc[i].mode() = 1;  // enable the servo
        }
    }

    ~UnitreeBackendImpl() override { cleanup(); }

    [[nodiscard]] bool init() {
        try {
            // ChannelFactory is a process-wide singleton; guard against
            // double-init. Set the flag only AFTER Init() returns so a throwing
            // Init() (bad interface / DDS config) doesn't poison the guard and
            // make a later retry skip the real factory init.
            static std::atomic<bool> factory_initialized{false};
            if (!factory_initialized.load(std::memory_order_acquire)) {
                unitree::robot::ChannelFactory::Instance()->Init(domain_, network_interface_);
                factory_initialized.store(true, std::memory_order_release);
            }

            // Subscribe-only (read-only tools): no rt/lowcmd writer exists, so
            // nothing in this process can publish a command.
            if (!subscribe_only_) {
                publisher_
                    = std::make_unique<unitree::robot::ChannelPublisher<LowCmdHg>>(kCmdTopic);
                publisher_->InitChannel();
            }

            subscriber_
                = std::make_unique<unitree::robot::ChannelSubscriber<LowStateHg>>(kStateTopic);
            // The SDK callback signature is std::function<void(const void*)>; the
            // message must be cast at the call site. std::bind captures `this`
            // which is stable for the subscriber's lifetime.
            subscriber_->InitChannel(
                std::bind(&UnitreeBackendImpl::on_low_state_raw, this,
                           std::placeholders::_1),
                /*queuelen=*/1);

            // Sport-mode state estimator subscriber. Mirrors the Python
            // backend's third subscriber on rt/odommodestate. Without the
            // high-level sport mode the topic stays silent, odom_valid never
            // becomes true and the control loop's odometry preflight fails.
            odom_subscriber_
                = std::make_unique<unitree::robot::ChannelSubscriber<SportModeStateGo>>(
                    kOdomTopic);
            odom_subscriber_->InitChannel(
                std::bind(&UnitreeBackendImpl::on_odom_state_raw, this,
                           std::placeholders::_1),
                /*queuelen=*/1);
            return true;
        } catch (const std::exception&) {
            return false;
        } catch (...) {
            return false;
        }
    }

    [[nodiscard]] robot::ReadStatus read_state(math::RobotState<kDof>& out) noexcept override {
        robot::ReadStatus status;
        if (const auto* snap = state_tb_.try_acquire_latest(); snap != nullptr) {
            cached_state_ = *snap;
            status        = robot::ReadStatus::kOk;
        } else if (!has_first_snapshot_.load(std::memory_order_acquire)) {
            return robot::ReadStatus::kNoData;
        } else {
            status = robot::ReadStatus::kStaleOk;
        }

        // Merge in the latest sport-mode estimator snapshot if present. Until the
        // odom topic produces, cached_odom_.valid stays false (the control loop
        // gates on it).
        if (const auto* osnap = odom_tb_.try_acquire_latest(); osnap != nullptr) {
            cached_odom_ = *osnap;
        }
        // A stream that STOPPED is as invalid as one that never started: the
        // last pose would otherwise be served as current forever.
        const bool odom_fresh =
            cached_odom_.valid
            && core::MonotonicClock::now().count() - cached_odom_.rx_ns <= odom_stale_ns_;

        out = cached_state_;
        out.odom_valid           = odom_fresh;
        out.odom_timestamp_ns    = cached_odom_.rx_ns;
        out.odom_position        = cached_odom_.position;
        out.odom_velocity        = cached_odom_.velocity;
        out.odom_quaternion_wxyz = cached_odom_.quaternion_wxyz;
        out.odom_yaw_speed       = cached_odom_.yaw_speed;
        return status;
    }

    void send_command(const math::MotorCommand<kDof>& cmd) noexcept override {
        // mode_machine must echo whatever the robot announced in its last
        // LowStateHg (as in the SDK's g1_ankle_swing_example). mode_pr stays at 0
        // (PR/Series).
        last_cmd_.mode_machine() = mode_machine_.load(std::memory_order_relaxed);
        last_cmd_.mode_pr()      = 0;  // PR / Series control

        auto& mc = last_cmd_.motor_cmd();
        // Fill the G1's 29 active joints.
        for (std::size_t i = 0; i < kDof; ++i) {
            mc[i].mode() = 1;
            mc[i].q()    = cmd.positions[static_cast<Eigen::Index>(i)];
            mc[i].dq()   = cmd.velocities[static_cast<Eigen::Index>(i)];
            mc[i].kp()   = cmd.kp[static_cast<Eigen::Index>(i)];
            mc[i].kd()   = cmd.kd[static_cast<Eigen::Index>(i)];
            mc[i].tau()  = cmd.torques[static_cast<Eigen::Index>(i)];
        }
        // Leave the remaining slots (29..34) inert: kp=0 kd=0 q=0. They were
        // zeroed in the ctor and are untouched on the hot path.

        last_cmd_.crc() = compute_low_cmd_crc(last_cmd_);
        // DRY RUN choke point. Deliberately placed AFTER the full command build
        // and CRC so the withheld path does the same build + CRC work as the live
        // one (only the DDS Write is skipped) — a timing experiment that skipped
        // the build would measure the wrong thing. This is the single place a
        // LowCmd_ reaches the wire, so gating it here cannot be bypassed by
        // another call site (the state machine's damping, cleanup()'s
        // damping-on-exit, or anything added later).
        if (!send_enabled_ || publisher_ == nullptr) {
            suppressed_sends_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const bool ok = publisher_->Write(last_cmd_);
        if (!ok) {
            publish_failures_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] bool poll_remote(robot::RemoteState& out) noexcept override {
        if (const auto* snap = remote_tb_.try_acquire_latest(); snap != nullptr) {
            cached_remote_     = *snap;
            has_remote_data_   = true;
        }
        if (!has_remote_data_) {
            return false;
        }
        out = cached_remote_;
        return true;
    }

    void cleanup() noexcept override {
        bool expected = false;
        if (!cleanup_done_.compare_exchange_strong(expected, true)) {
            return;  // idempotent
        }
        // Best-effort damping command before we tear down the channel. In a dry
        // run this too is withheld at the choke point in send_command() — the
        // robot is never commanded, not even to damp — so report the total
        // withheld count as the audit trail for the run.
        if (publisher_ != nullptr) {
            math::MotorCommand<kDof> damping;
            damping.set_damping(damping_kd_);
            send_command(damping);
        }
        if (!send_enabled_ && !subscribe_only_) {
            std::fprintf(stderr,
                         "[unitree_backend] DRY RUN: %llu commands built and withheld; "
                         "0 published (damping-on-exit withheld too)\n",
                         static_cast<unsigned long long>(
                             suppressed_sends_.load(std::memory_order_relaxed)));
        }
        if (publisher_ != nullptr) {
            publisher_->CloseChannel();
            publisher_.reset();
        }
        if (subscriber_ != nullptr) {
            subscriber_->CloseChannel();
            subscriber_.reset();
        }
        if (odom_subscriber_ != nullptr) {
            odom_subscriber_->CloseChannel();
            odom_subscriber_.reset();
        }
    }

    [[nodiscard]] bool wait_for_state(std::int64_t timeout_ns) override {
        const auto              start_ns = core::MonotonicClock::now();
        const auto              deadline = start_ns + core::MonotonicClock::Nanos{timeout_ns};
        constexpr std::int64_t  kProgressIntervalNs = 2'000'000'000LL;  // 2 s
        std::int64_t            next_progress_ns
            = start_ns.count() + kProgressIntervalNs;

        while (core::MonotonicClock::now() < deadline) {
            if (has_first_snapshot_.load(std::memory_order_acquire)) {
                return true;
            }
            // Periodic progress line so a stalled interface is visible
            // instead of a silent 10-second hang.
            const std::int64_t now_ns = core::MonotonicClock::now().count();
            if (now_ns >= next_progress_ns) {
                next_progress_ns = now_ns + kProgressIntervalNs;
                const auto rx  = rx_count_.load(std::memory_order_relaxed);
                const auto crc = crc_rejects_.load(std::memory_order_relaxed);
                std::fprintf(stderr,
                             "[unitree_backend] waiting on %s (domain %d)  "
                             "rx=%llu  crc_rejects=%llu\n",
                             network_interface_.c_str(), domain_,
                             static_cast<unsigned long long>(rx),
                             static_cast<unsigned long long>(crc));
            }
            // Cheap poll — state arrives asynchronously on the DDS thread.
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
        }

        // Timed out — surface what we know so the operator can diagnose.
        const auto rx  = rx_count_.load(std::memory_order_relaxed);
        const auto crc = crc_rejects_.load(std::memory_order_relaxed);
        std::fprintf(stderr,
                     "[unitree_backend] wait_for_state timed out after %lld ms\n"
                     "  interface: %s  domain: %d  topic: %s\n"
                     "  rx_count: %llu  crc_rejects: %llu  verify_crc: %s\n",
                     timeout_ns / 1'000'000LL,
                     network_interface_.c_str(), domain_, kStateTopic,
                     static_cast<unsigned long long>(rx),
                     static_cast<unsigned long long>(crc),
                     verify_inbound_crc_ ? "on" : "off");
        if (rx > 0 && rx == crc) {
            std::fprintf(stderr,
                         "  All %llu inbound messages failed CRC. Either the Unitree SDK\n"
                         "  version compiled into this binary differs from the robot\n"
                         "  firmware, or the wire layout does. Rebuild with\n"
                         "  verify_inbound_crc=false to bypass the application CRC check\n"
                         "  (the Python backend already runs without it).\n",
                         static_cast<unsigned long long>(rx));
        } else if (rx == 0) {
            std::fprintf(stderr,
                         "  No DDS messages received. Check in order:\n"
                         "    1. Robot is powered and its lowstate publisher is running.\n"
                         "    2. `ip link show %s` reports UP with a link-local IPv4/6 address.\n"
                         "    3. DDS domain %d matches whatever the robot was started with.\n"
                         "    4. Try `cyclonedds ps` or `ddsperf pub` on the same interface\n"
                         "       to confirm CycloneDDS sees multicast traffic.\n"
                         "    5. `CYCLONEDDS_URI` is unset (or points at a config with this\n"
                         "       interface whitelisted).\n",
                         network_interface_.c_str(), domain_);
        }
        return false;
    }

    // Diagnostic accessors — not on the hot path.
    [[nodiscard]] std::uint64_t publish_failures() const noexcept {
        return publish_failures_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t rx_count() const noexcept {
        return rx_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t crc_rejects() const noexcept {
        return crc_rejects_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] robot::BackendDiagnostics diagnostics() const noexcept override {
        return robot::BackendDiagnostics{
            /*rx_count=*/rx_count_.load(std::memory_order_relaxed),
            /*crc_rejects=*/crc_rejects_.load(std::memory_order_relaxed),
            /*publish_failures=*/publish_failures_.load(std::memory_order_relaxed),
            /*odom_rx_count=*/odom_rx_count_.load(std::memory_order_relaxed),
        };
    }

private:
    // SDK callback thunk — unpacks const void* and forwards to the typed handler.
    void on_low_state_raw(const void* message) noexcept {
        if (message == nullptr) {
            return;
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        const auto& state = *static_cast<const LowStateHg*>(message);
        on_low_state(state);
    }

    // SDK callback thunk for the optional sport-mode estimator topic.
    void on_odom_state_raw(const void* message) noexcept {
        if (message == nullptr) {
            return;
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        const auto& sm = *static_cast<const SportModeStateGo*>(message);
        on_odom_state(sm);
    }

    void on_odom_state(const SportModeStateGo& sm) noexcept {
        // Per-callback profile probe — separate from dds_callback so a stalled
        // odom topic is visible distinct from a stalled lowstate.
        static core::RateProbe probe{"dds_odom_callback"};
        probe.tick();
        auto probe_scope = probe.scope();

        odom_rx_count_.fetch_add(1, std::memory_order_relaxed);

        // A non-finite pose counts as no message: the stream then goes stale and the
        // control loop's odometry gate / stale-odom latch handle it (as in Python).
        const auto& p_in = sm.position();
        const auto& q_in = sm.imu_state().quaternion();
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(p_in[i])) {
                return;
            }
        }
        for (int i = 0; i < 4; ++i) {
            if (!std::isfinite(q_in[i])) {
                return;
            }
        }

        auto& slot = odom_tb_.producer_slot();
        slot.valid = true;
        slot.rx_ns = core::MonotonicClock::now().count();

        const auto& pos = sm.position();
        slot.position[0] = pos[0];
        slot.position[1] = pos[1];
        slot.position[2] = pos[2];

        const auto& vel = sm.velocity();
        slot.velocity[0] = vel[0];
        slot.velocity[1] = vel[1];
        slot.velocity[2] = vel[2];

        // SportModeState_ embeds an IMUState_ whose quaternion is wxyz, same
        // convention as LowState_'s imu_state.quaternion.
        const auto& q = sm.imu_state().quaternion();
        slot.quaternion_wxyz[0] = q[0];
        slot.quaternion_wxyz[1] = q[1];
        slot.quaternion_wxyz[2] = q[2];
        slot.quaternion_wxyz[3] = q[3];

        slot.yaw_speed = sm.yaw_speed();
        odom_tb_.publish();
    }

    void on_low_state(const LowStateHg& state) noexcept {
        // Per-callback profile probe. Measures the time the SDK's DDS thread
        // spends inside our handler. tick() also gives us the message rate.
        // No-op when --profile is off.
        static core::RateProbe probe{"dds_callback"};
        probe.tick();
        auto probe_scope = probe.scope();

        // Count every callback invocation. This is the ground truth for "is
        // the DDS transport delivering anything at all" — it fires before the
        // CRC check so a high rx_count with zero published snapshots unambig-
        // uously points at a CRC-layout mismatch rather than a transport
        // failure.
        rx_count_.fetch_add(1, std::memory_order_relaxed);

        // Drop messages with bad CRC only when the operator opts in (default
        // off, matching the Python backend). DDS already checksums at the
        // transport level; the extra application CRC is defensive and can
        // silently discard every message if SDK versions skew between this
        // binary and the robot firmware.
        if (verify_inbound_crc_ && !verify_low_state_crc(state)) {
            crc_rejects_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        mode_machine_.store(state.mode_machine(), std::memory_order_relaxed);

        // Write the RobotState snapshot into the triple buffer.
        auto& slot = state_tb_.producer_slot();
        // Motor state: G1 uses the first 29 slots.
        const auto& motor_state = state.motor_state();
        for (std::size_t i = 0; i < kDof; ++i) {
            const auto ei = static_cast<Eigen::Index>(i);
            slot.joint_positions[ei]  = motor_state[i].q();
            slot.joint_velocities[ei] = motor_state[i].dq();
            slot.joint_torques[ei]    = motor_state[i].tau_est();
        }

        // IMU — quaternion is wxyz in the SDK layout (verified against
        // deploy/utils/backends/unitree_backend.py; the Python bindings
        // feed state.imu_quaternion straight into observation builders that
        // treat it as wxyz). Cache the reference so we don't re-fetch it.
        const auto& imu = state.imu_state();
        const auto& q   = imu.quaternion();
        slot.imu_quat_wxyz[0] = q[0];
        slot.imu_quat_wxyz[1] = q[1];
        slot.imu_quat_wxyz[2] = q[2];
        slot.imu_quat_wxyz[3] = q[3];
        const auto& gyro  = imu.gyroscope();
        const auto& accel = imu.accelerometer();
        for (int i = 0; i < 3; ++i) {
            slot.imu_angular_velocity[i]    = gyro[i];
            slot.imu_linear_acceleration[i] = accel[i];
        }
        slot.timestamp_ns = core::MonotonicClock::now().count();
        state_tb_.publish();

        // Publish ordering matters: the release-store on has_first_snapshot_
        // must happen AFTER state_tb_.publish() so any reader that sees the
        // flag set is guaranteed (via the TripleBuffer's own release-acquire)
        // to see the published snapshot.
        has_first_snapshot_.store(true, std::memory_order_release);

        // Parse the 40-byte wireless_remote blob (button bitmask + four sticks).
        const auto& remote_bytes = state.wireless_remote();
        robot::RemoteState parsed;
        if (robot::parse_wireless_remote(std::span<const std::uint8_t>{
                                              remote_bytes.data(), remote_bytes.size()},
                                          parsed)) {
            auto& rslot = remote_tb_.producer_slot();
            rslot       = parsed;
            remote_tb_.publish();
        }
    }

    std::string                                                    network_interface_;
    int                                                            domain_;
    float                                                          damping_kd_;
    bool                                                           verify_inbound_crc_;
    std::int64_t                                                   odom_stale_ns_;

    std::unique_ptr<unitree::robot::ChannelPublisher<LowCmdHg>>    publisher_;
    std::unique_ptr<unitree::robot::ChannelSubscriber<LowStateHg>> subscriber_;
    std::unique_ptr<unitree::robot::ChannelSubscriber<SportModeStateGo>> odom_subscriber_;

    LowCmdHg                                                       last_cmd_{};
    std::atomic<std::uint8_t>                                      mode_machine_{0};

    core::TripleBuffer<math::RobotState<kDof>> state_tb_{};
    math::RobotState<kDof>                     cached_state_{};
    std::atomic<bool>                          has_first_snapshot_{false};

    core::TripleBuffer<robot::RemoteState>     remote_tb_{};
    robot::RemoteState                         cached_remote_{};
    bool                                       has_remote_data_{false};

    // Sport-mode estimator handoff (mirrors Python's _odom_lock + odom_state).
    core::TripleBuffer<OdomSnapshot>           odom_tb_{};
    OdomSnapshot                               cached_odom_{};

    std::atomic<std::uint64_t>                 publish_failures_{0};
    // Dry run: false withholds every LowCmd_ from the wire. See
    // UnitreeBackendConfig::send_enabled for the rationale and the safety note.
    bool                                       send_enabled_{true};
    bool                                       subscribe_only_{false};
    std::atomic<std::uint64_t>                 suppressed_sends_{0};
    std::atomic<std::uint64_t>                 crc_rejects_{0};
    std::atomic<std::uint64_t>                 rx_count_{0};
    std::atomic<std::uint64_t>                 odom_rx_count_{0};
    std::atomic<bool>                          cleanup_done_{false};
};

}  // namespace

core::Result<std::unique_ptr<robot::IRobotBackend<kDof>>>
make_unitree_backend(const UnitreeBackendConfig& cfg) {
    using ResT = core::Result<std::unique_ptr<robot::IRobotBackend<kDof>>>;

    auto impl = std::unique_ptr<UnitreeBackendImpl>(
        new UnitreeBackendImpl{cfg.network_interface, cfg.domain, cfg.damping_kd,
                                cfg.verify_inbound_crc, cfg.send_enabled, cfg.subscribe_only,
                                cfg.odom_stale_ns});

    if (!impl->init()) {
        return ResT::Error{"unitree_backend: DDS channel init failed on interface '"
                            + cfg.network_interface + "'"};
    }
    return std::unique_ptr<robot::IRobotBackend<kDof>>{impl.release()};
}

}  // namespace rt_deploy::g1
