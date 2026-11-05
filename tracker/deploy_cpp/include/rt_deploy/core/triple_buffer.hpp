//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Wait-free single-producer / single-consumer triple buffer.
//
// The producer always has a slot it can write to; the consumer always gets the
// most recent fully-published value. Exactly one of the three slots is "being
// written", one is "published/latest", and one is "being read" at any moment.
//
// The coordination word packs (latest_index | new_flag) into a single uint32.
// swap_on_publish and try_acquire_latest use CAS on that word, so there is no
// lock or blocking waiter and no heap allocation.
//
// Used wherever only the *latest* value matters and intermediate values can be
// dropped: robot state (I/O thread -> control thread), runtime-pushed motion
// plans, active-plan snapshots, and depth frames.

#include <array>
#include <atomic>
#include <cstdint>
#include <utility>

namespace rt_deploy::core {

template <typename T>
class TripleBuffer {
public:
    TripleBuffer() noexcept  = default;
    ~TripleBuffer() noexcept = default;

    TripleBuffer(const TripleBuffer&)            = delete;
    TripleBuffer& operator=(const TripleBuffer&) = delete;
    TripleBuffer(TripleBuffer&&)                 = delete;
    TripleBuffer& operator=(TripleBuffer&&)      = delete;

    // Producer side — slot to write into. Same slot is returned until
    // publish() is called.
    [[nodiscard]] T& producer_slot() noexcept { return slots_[producer_idx_]; }

    // Producer side — mark the current producer slot as the new latest. Swaps
    // producer_idx_ with the "latest" index stored in ctrl_; future calls to
    // producer_slot() write into the slot the consumer is *not* currently
    // reading.
    void publish() noexcept {
        std::uint32_t cur = ctrl_.load(std::memory_order_acquire);
        std::uint32_t next;
        do {
            next = static_cast<std::uint32_t>(producer_idx_) | kNewFlag;
            // Keep the "being read" slot intact; swap only the "latest" slot.
            // acq_rel: release our writes to the published slot, and acquire the consumer's reads of the
            // slot we take back, so our next writes cannot race them.
        } while (!ctrl_.compare_exchange_weak(cur, next, std::memory_order_acq_rel,
                                               std::memory_order_acquire));
        producer_idx_ = static_cast<std::uint8_t>(cur & kIndexMask);
    }

    // Consumer side — acquire the latest published value, if any new data is
    // available since the last acquire. Returns pointer to the stable slot, or
    // nullptr if nothing new. The pointer remains valid until the next call
    // to try_acquire_latest().
    [[nodiscard]] const T* try_acquire_latest() noexcept {
        std::uint32_t cur = ctrl_.load(std::memory_order_acquire);
        if ((cur & kNewFlag) == 0U) {
            return nullptr;
        }
        // Atomically swap consumer_idx_ with the latest slot, clear the new flag.
        std::uint32_t next;
        do {
            next = static_cast<std::uint32_t>(consumer_idx_);
            // acq_rel: acquire the producer's writes to the latest slot, and release our reads of the slot we
            // hand back before the producer reuses it.
        } while (!ctrl_.compare_exchange_weak(cur, next, std::memory_order_acq_rel,
                                                std::memory_order_acquire));
        consumer_idx_ = static_cast<std::uint8_t>(cur & kIndexMask);
        return &slots_[consumer_idx_];
    }

private:
    static constexpr std::uint32_t kIndexMask = 0x0000'00FFU;
    static constexpr std::uint32_t kNewFlag   = 0x0000'0100U;

    // Initial layout: slot 0 is "latest" (producer writes to slot 1, consumer
    // holds slot 2). No data is published yet, so kNewFlag starts clear.
    alignas(64) std::atomic<std::uint32_t> ctrl_{0U};
    alignas(64) std::uint8_t producer_idx_ = 1;
    std::uint8_t consumer_idx_ = 2;

    alignas(64) std::array<T, 3> slots_{};
};

}  // namespace rt_deploy::core
