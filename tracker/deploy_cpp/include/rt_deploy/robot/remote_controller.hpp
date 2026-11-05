//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// Builds on unitree_rl_gym (Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD., BSD-3-Clause).

#pragma once

// RemoteController — parser for Unitree `wireless_remote` bytes.
//
// Parity with deploy/utils/core/remote_controller.py:RemoteController.set.
//
// Byte layout (as parsed by the Python RemoteController.set):
//   bytes[2:4]   uint16 LE : packed button bitmask (16 buttons)
//   bytes[4:8]   float32 LE: lx
//   bytes[8:12]  float32 LE: rx        ← note: rx, not ly!
//   bytes[12:16] float32 LE: ry
//   bytes[20:24] float32 LE: ly        ← offset-20, NOT 16 (bytes 16-19 are a gap)
//
// Only bytes 0..23 are consumed. Data buffers shorter than 24 bytes are
// an error — returns false and leaves the output unchanged.

#include "rt_deploy/robot/i_robot_backend.hpp"

#include <cstdint>
#include <cstring>
#include <span>

namespace rt_deploy::robot {

// Decode the wireless_remote blob into a RemoteState. noexcept, hot-path safe.
[[nodiscard]] inline bool parse_wireless_remote(std::span<const std::uint8_t> bytes,
                                                  RemoteState&                   out) noexcept {
    if (bytes.size() < 24) {
        return false;
    }
    std::uint16_t keys = 0;
    std::memcpy(&keys, bytes.data() + 2, sizeof(keys));
    for (int i = 0; i < 16; ++i) {
        out.button[static_cast<std::size_t>(i)]
            = static_cast<std::uint8_t>((keys >> i) & 0x1U);
    }
    std::memcpy(&out.lx, bytes.data() + 4,  sizeof(float));
    std::memcpy(&out.rx, bytes.data() + 8,  sizeof(float));
    std::memcpy(&out.ry, bytes.data() + 12, sizeof(float));
    std::memcpy(&out.ly, bytes.data() + 20, sizeof(float));
    return true;
}

}  // namespace rt_deploy::robot
