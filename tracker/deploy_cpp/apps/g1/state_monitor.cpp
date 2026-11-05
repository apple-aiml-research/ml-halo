//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// state_monitor — read-only Unitree G1 state monitor.
//
// Opens the DDS subscriber for rt/lowstate, prints a live dashboard of
// joints, IMU, wireless remote and rx counters, and exits cleanly on SIGINT.
//
// The backend is opened subscribe-only (UnitreeBackendConfig::subscribe_only):
// no rt/lowcmd publisher exists, so no LowCmd_ is ever published, not even the
// damping command a backend normally sends on exit. Safe to run next to a live
// controller. Use this to validate the network + DDS + CRC path before enabling
// deploy_g1_real.
//
// Build:   RT_DEPLOY_WITH_UNITREE=ON
// Usage:   g1_state_monitor --net eth0 [--domain 0] [--verify_crc]
//                           [--rate 10] [--once]

#include "rt_deploy/core/cli_args.hpp"
#include "rt_deploy/core/clock.hpp"
#include "rt_deploy/core/parse_int.hpp"
#include "rt_deploy/g1/config.hpp"
#include "rt_deploy/g1/unitree_backend.hpp"
#include "rt_deploy/math/quat.hpp"
#include "rt_deploy/math/types.hpp"
#include "rt_deploy/robot/i_robot_backend.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace {

// ─── Signal handling ─────────────────────────────────────────────────────────
std::atomic<bool> g_stop{false};
void on_signal(int /*signum*/) noexcept { g_stop.store(true, std::memory_order_release); }

// ─── ANSI helpers for the in-place dashboard ─────────────────────────────────
constexpr const char* kClearScreen  = "\x1b[2J";
constexpr const char* kCursorHome   = "\x1b[H";
constexpr const char* kCursorHide   = "\x1b[?25l";
constexpr const char* kCursorShow   = "\x1b[?25h";
constexpr const char* kClearEol     = "\x1b[K";
constexpr const char* kBold         = "\x1b[1m";
constexpr const char* kDim          = "\x1b[2m";
constexpr const char* kReset        = "\x1b[0m";
constexpr const char* kGreen        = "\x1b[32m";
constexpr const char* kYellow       = "\x1b[33m";
constexpr const char* kRed          = "\x1b[31m";

// G1 joint layout in deployment order (from configs/g1_v2.yaml).
//   [0..5]   left leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
//   [6..11]  right leg  (same pattern)
//   [12..14] torso  (waist_yaw, waist_roll, waist_pitch)
//   [15..21] left arm + wrist  (shoulder_pitch, _roll, _yaw, elbow, wrist_roll, _pitch, _yaw)
//   [22..28] right arm + wrist
constexpr int kSegments[] = {0, 6, 12, 15, 22, 29};
constexpr const char* kSegmentNames[] = {
    "left  leg ", "right leg ", "torso     ", "left  arm ", "right arm "};

// ─── CLI ─────────────────────────────────────────────────────────────────────
struct Args {
    std::string network_interface = "eth0";
    int         domain             = 0;
    bool        verify_crc         = false;
    int         rate_hz            = 10;
    bool        once               = false;  // print one snapshot and exit
    bool        verbose            = false;
};

void print_usage() {
    std::fprintf(stderr,
                 "usage: g1_state_monitor [options]\n"
                 "  --net IFACE     network interface (default eth0)\n"
                 "  --domain N      DDS domain id (default 0)\n"
                 "  --verify_crc    verify inbound LowState_ CRC (default off)\n"
                 "  --rate HZ       redraw rate (default 10 Hz)\n"
                 "  --once          print one snapshot and exit (no dashboard)\n"
                 "  --verbose       also print joint velocities + all buttons\n");
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string_view k{argv[i]};
        // Refuses a value that is itself a flag, so a forgotten value is an error.
        auto need = [&](int& j) -> const char* { return rt_deploy::core::take_value(argc, argv, j); };
        if (k == "-h" || k == "--help") {
            return false;
        } else if (k == "--net") {
            const char* v = need(i); if (v == nullptr) return false; a.network_interface = v;
        } else if (k == "--domain") {
            const char* v = need(i); if (v == nullptr) return false;
            if (!rt_deploy::core::parse_int_strict(v, a.domain) || a.domain < 0) {
                std::fprintf(stderr, "--domain expects a non-negative integer, got '%s'\n", v);
                return false;
            }
        } else if (k == "--verify_crc") {
            a.verify_crc = true;
        } else if (k == "--rate") {
            const char* v = need(i); if (v == nullptr) return false;
            if (!rt_deploy::core::parse_int_strict(v, a.rate_hz)) {
                std::fprintf(stderr, "--rate expects an integer, got '%s'\n", v);
                return false;
            }
        } else if (k == "--once") {
            a.once = true;
        } else if (k == "--verbose") {
            a.verbose = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    if (a.rate_hz <= 0 || a.rate_hz > 200) {
        std::fprintf(stderr, "--rate must be in [1, 200]\n");
        return false;
    }
    return true;
}

// ─── Pretty-printers ─────────────────────────────────────────────────────────

const char* status_tag(rt_deploy::robot::ReadStatus s) noexcept {
    using S = rt_deploy::robot::ReadStatus;
    switch (s) {
        case S::kOk:          return "OK    ";
        case S::kStaleOk:     return "STALE ";
        case S::kNoData:      return "NO-DATA";
        case S::kDisconnect:  return "DISC  ";
    }
    return "?     ";
}

const char* status_colour(rt_deploy::robot::ReadStatus s) noexcept {
    using S = rt_deploy::robot::ReadStatus;
    switch (s) {
        case S::kOk:          return kGreen;
        case S::kStaleOk:     return kYellow;
        case S::kNoData:
        case S::kDisconnect:  return kRed;
    }
    return kReset;
}

// Render buttons as "[A B  Y  Sel ]"-style string. Short enough to fit.
std::string render_buttons(const rt_deploy::robot::RemoteState& r, bool verbose) {
    namespace km = rt_deploy::math::key_map;
    const struct {
        std::size_t idx;
        const char* label;
    } btns[] = {
        {km::kR1,     "R1 "}, {km::kL1,     "L1 "},
        {km::kStart,  "St "}, {km::kSelect, "Sel"},
        {km::kR2,     "R2 "}, {km::kL2,     "L2 "},
        {km::kA,      "A  "}, {km::kB,      "B  "},
        {km::kX,      "X  "}, {km::kY,      "Y  "},
        {km::kUp,     "Up "}, {km::kRight,  "Rt "},
        {km::kDown,   "Dn "}, {km::kLeft,   "Lt "},
    };
    std::string out = "[";
    for (auto b : btns) {
        if (r.button[b.idx] != 0) {
            out += kBold;
            out += b.label;
            out += kReset;
        } else if (verbose) {
            out += kDim;
            out += b.label;
            out += kReset;
        }
    }
    out += "]";
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage();
        return 2;
    }

    std::signal(SIGINT, &on_signal);
    std::signal(SIGTERM, &on_signal);

    rt_deploy::g1::UnitreeBackendConfig cfg;
    cfg.network_interface   = args.network_interface;
    cfg.domain              = args.domain;
    cfg.verify_inbound_crc  = args.verify_crc;
    cfg.subscribe_only      = true;   // read-only: never create a LowCmd_ publisher

    auto r = rt_deploy::g1::make_unitree_backend(cfg);
    if (!r) {
        std::fprintf(stderr, "[error] backend init failed: %s\n", r.error().c_str());
        return 1;
    }
    auto backend = std::move(r.value());

    std::fprintf(stderr,
                 "[g1_state_monitor] interface=%s  domain=%d  verify_crc=%s\n"
                 "[g1_state_monitor] waiting up to 10s for first LowState_...\n",
                 args.network_interface.c_str(), args.domain,
                 args.verify_crc ? "on" : "off");

    if (!backend->wait_for_state(10'000'000'000LL)) {
        std::fprintf(stderr, "[g1_state_monitor] no state received. Exiting.\n");
        return 3;
    }
    std::fprintf(stderr, "%s[g1_state_monitor] state received, starting monitor%s\n",
                 kGreen, kReset);

    // Main read-print loop.
    const auto period
        = std::chrono::nanoseconds{1'000'000'000LL / args.rate_hz};
    rt_deploy::core::PeriodicTicker ticker{period};

    // Track rx rate across ticks (5 s EMA).
    std::uint64_t last_rx          = 0;
    std::int64_t  last_rx_time_ns  = rt_deploy::core::MonotonicClock::now().count();
    double        rx_rate_hz       = 0.0;

    if (!args.once) {
        std::printf("%s%s", kClearScreen, kCursorHide);
    }

    while (!g_stop.load(std::memory_order_acquire)) {
        (void)ticker.wait_next();

        rt_deploy::math::RobotState<rt_deploy::g1::kDof> state;
        const auto rs = backend->read_state(state);

        rt_deploy::robot::RemoteState remote;
        const bool remote_ok = backend->poll_remote(remote);

        const auto diag = backend->diagnostics();

        // Compute rx rate (exponential smoothing).
        const std::int64_t now_ns = rt_deploy::core::MonotonicClock::now().count();
        const double dt = static_cast<double>(now_ns - last_rx_time_ns) * 1e-9;
        if (dt > 0.1) {
            const double instant
                = static_cast<double>(diag.rx_count - last_rx) / dt;
            rx_rate_hz       = 0.7 * rx_rate_hz + 0.3 * instant;
            last_rx          = diag.rx_count;
            last_rx_time_ns  = now_ns;
        }

        if (!args.once) {
            std::printf("%s", kCursorHome);
        }

        // Header
        std::printf("%s%sG1 State Monitor%s   interface: %s   domain: %d   topic: rt/lowstate%s\n",
                    kBold, kGreen, kReset,
                    args.network_interface.c_str(), args.domain, kClearEol);
        std::printf("rx: %s%llu msgs%s  (%.1f Hz)   crc_rejects: %s%llu%s   "
                    "read: %s%s%s%s\n",
                    kBold,
                    static_cast<unsigned long long>(diag.rx_count),
                    kReset, rx_rate_hz,
                    diag.crc_rejects > 0 ? kRed : kReset,
                    static_cast<unsigned long long>(diag.crc_rejects), kReset,
                    status_colour(rs), status_tag(rs), kReset, kClearEol);
        std::printf("%s\n", kClearEol);

        // IMU
        const auto& q = state.imu_quat_wxyz;
        std::printf("IMU  quat wxyz: %s%+.4f  %+.4f  %+.4f  %+.4f%s%s\n",
                    kBold,
                    static_cast<double>(q[0]),
                    static_cast<double>(q[1]),
                    static_cast<double>(q[2]),
                    static_cast<double>(q[3]),
                    kReset, kClearEol);
        const auto rpy = rt_deploy::math::quat_to_euler(q);
        const double rad2deg = 180.0 / M_PI;
        std::printf("     rpy (deg): %+7.2f  %+7.2f  %+7.2f%s\n",
                    static_cast<double>(rpy[0]) * rad2deg,
                    static_cast<double>(rpy[1]) * rad2deg,
                    static_cast<double>(rpy[2]) * rad2deg, kClearEol);
        const auto& w = state.imu_angular_velocity;
        std::printf("     gyro (rad/s): %+6.3f  %+6.3f  %+6.3f%s\n",
                    static_cast<double>(w[0]),
                    static_cast<double>(w[1]),
                    static_cast<double>(w[2]), kClearEol);
        const auto& a = state.imu_linear_acceleration;
        std::printf("     accel (m/s²): %+6.2f  %+6.2f  %+6.2f%s\n",
                    static_cast<double>(a[0]),
                    static_cast<double>(a[1]),
                    static_cast<double>(a[2]), kClearEol);
        std::printf("%s\n", kClearEol);

        // Joints, grouped by body segment.
        std::printf("%sJoint positions (rad, deployment order):%s%s\n",
                    kBold, kReset, kClearEol);
        for (std::size_t seg = 0; seg < 5; ++seg) {
            std::printf("  %s[%2d..%2d]  %s", kSegmentNames[seg],
                        kSegments[seg], kSegments[seg + 1] - 1, "");
            for (int i = kSegments[seg]; i < kSegments[seg + 1]; ++i) {
                std::printf("%+6.3f ", static_cast<double>(state.joint_positions[i]));
            }
            std::printf("%s\n", kClearEol);
        }
        if (args.verbose) {
            std::printf("%sJoint velocities (rad/s):%s%s\n", kBold, kReset, kClearEol);
            for (std::size_t seg = 0; seg < 5; ++seg) {
                std::printf("  %s[%2d..%2d]  ", kSegmentNames[seg],
                            kSegments[seg], kSegments[seg + 1] - 1);
                for (int i = kSegments[seg]; i < kSegments[seg + 1]; ++i) {
                    std::printf("%+6.2f ", static_cast<double>(state.joint_velocities[i]));
                }
                std::printf("%s\n", kClearEol);
            }
        }
        std::printf("%s\n", kClearEol);

        // Remote
        if (remote_ok) {
            std::printf("Remote: %s   lx=%+.2f  ly=%+.2f  rx=%+.2f  ry=%+.2f%s\n",
                        render_buttons(remote, args.verbose).c_str(),
                        static_cast<double>(remote.lx),
                        static_cast<double>(remote.ly),
                        static_cast<double>(remote.rx),
                        static_cast<double>(remote.ry), kClearEol);
        } else {
            std::printf("Remote: %s(no data yet)%s%s\n", kDim, kReset, kClearEol);
        }
        std::printf("%s\n", kClearEol);

        // Footer
        std::printf("%sCtrl-C to exit%s%s\n", kDim, kReset, kClearEol);
        std::fflush(stdout);

        if (args.once) {
            break;
        }
    }

    if (!args.once) {
        std::printf("%s\n", kCursorShow);
    }
    std::fflush(stdout);

    backend->cleanup();
    return 0;
}
