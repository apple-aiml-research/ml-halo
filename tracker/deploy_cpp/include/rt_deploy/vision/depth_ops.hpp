//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// depth_ops — the disparity->metric-depth, invalid, and downsample rules, in
// ONE place.
//
// Single source of truth for the classical post-processing that turns the
// network's disparity (or the D435i's own Z16) into the fused 84x48 depth the
// depth student consumes. Shared by:
//   * the CUDA post-process kernel  (compiled by nvcc as __host__ __device__)
//   * the raw D435i CPU path        (vision::RawDepthSource)
// so the GPU result and the raw fallback cannot drift apart.
//
// Output contract — this MUST match what the policy was trained on
// (training/.../g1/mdp/observations.py::depth_image):
//   * float metres, clamped to [clip_min, clip_max].
//   * invalid pixels (no disparity, left-occlusion shadow where x-disp < 0, or
//     a D435i zero) -> clip_max, i.e. FAR / free space. NEVER 0. Training
//     encodes every missing sample as the far plane precisely so the CNN never
//     sees a spurious close obstacle; a real D435i reports 0 for "no data", so
//     the remap here is mandatory, not cosmetic.
//   * a valid surface closer than clip_min clamps UP to clip_min (np.clip
//     semantics) — it is NOT pushed to far.
//
// The g1::ObservationBuilder image group then applies clip -> log1p on top of
// this. Nothing here applies log1p: this file speaks metres.

#include <cstdint>

#if defined(__CUDACC__)
#    define RT_DEPLOY_HD __host__ __device__
#else
#    define RT_DEPLOY_HD
#endif

namespace rt_deploy::vision {

// How a block of full-resolution depth is reduced to one output cell.
// vision::DownsampleMode is an alias of this (see config.hpp) so the config
// enum and the device code can never disagree on the numeric values.
enum class Reduce : std::uint8_t {
    kMin   = 0,  // closest depth in the footprint (conservative)
    kArea  = 1,  // mean over the footprint
    kPoint = 2,  // nearest-neighbour at the footprint centre
    // Mean over the MEASURED samples only. Invalid samples (no disparity, the
    // occlusion shadow, a D435i zero) are excluded from the average instead of
    // entering it at clip_max; a cell reads far only when its whole footprint is
    // invalid. Genuine far readings ARE included — they are data, not holes.
    //
    // WHY IT EXISTS. kArea averages the far-filled holes, so every output cell is
    // biased FARTHER by an amount that tracks local hole density: at the D435i's
    // measured 15.4% dropout a 0.5 m wall reads 0.81 m (+0.32 m). Training never
    // saw that — it rendered 48x84 NATIVELY with no downsample at all, so its
    // pixel dropout set WHOLE output cells to far rather than mixing a far
    // fraction into every one. Excluding the holes removes the bias exactly,
    // invents no geometry (unlike hole filling) and adds no lag (unlike a
    // temporal filter).
    //
    // Mirrors the Python deployment's depth_sources.sanitize_depth.
    kAreaValid = 3,
};

// Does this metric sample carry depth information?
//
// Invalid = non-finite, or <= 0 (the D435i's "no return" code). A reading ABOVE
// clip_max is valid — it is real data, merely saturated — and callers far-clamp it
// before aggregating rather than dropping it, because letting a 65.54 m
// out-of-range sample into a mean turns a 0.5 m wall into 1.14 m.
//
// The comparisons are negated so NaN lands on "invalid": `!(z > eps)` is TRUE for
// NaN. `!(z < kSaneMaxDepthM)` catches +Inf. Same idiom, and same reason, as
// sanitize_depth_m below.
[[nodiscard]] RT_DEPLOY_HD inline bool depth_sample_valid(float z) noexcept {
    // 1e6 m is beyond any depth sensor's range by orders of magnitude, so this
    // rejects +Inf without needing <cmath> in device code. A finite value this
    // large is already garbage.
    constexpr float kSaneMaxDepthM = 1.0e6F;
    if (!(z > 1.0e-6F)) {  // zero, negative, or NaN
        return false;
    }
    return z < kSaneMaxDepthM;  // false for +Inf
}

// Is this disparity sample a real correspondence at column `x`?
//
// Invalid = no disparity (<= 0, or NaN, which fails the negated compare) or the
// left-occlusion shadow, where the matching right-image column x-disp falls off
// the sensor. Exactly the condition disp_to_depth_clamped sends to clip_max.
[[nodiscard]] RT_DEPLOY_HD inline bool disp_sample_valid(float disp, int x) noexcept {
    return (disp > 1.0e-3F) && ((static_cast<float>(x) - disp) >= 0.0F);
}

// Convert one disparity sample at image column `x` into clamped metric depth.
//
//   disp        : disparity at this pixel [px] (network output, net resolution)
//   x           : pixel column index [0 .. net_width)
//   fx_baseline : fx[px] * baseline[m] at the network resolution
//   clip_min    : near clip [m] (valid depths below clamp up to this)
//   clip_max    : far clip [m] (valid depths above, and all invalid, map here)
//
// Returns depth in metres in [clip_min, clip_max].
[[nodiscard]] RT_DEPLOY_HD inline float disp_to_depth_clamped(float disp, int x, float fx_baseline,
                                                              float clip_min,
                                                              float clip_max) noexcept {
    // Invalid: no disparity, or the left-occlusion shadow (no right
    // correspondence). Treat as far / free space.
    //
    // Written `!(disp > eps)` rather than `disp <= eps` so a NaN disparity —
    // which an FP16 cost-volume matcher can emit when the volume overflows —
    // also lands here. Bit-identical to `disp <= eps` for every non-NaN input,
    // so the fused GPU reduction stays in lockstep. It matters because a NaN
    // that survived would poison a whole output cell (kMin seeds from the
    // block's first sample and no comparison against NaN is ever true, kArea
    // sums it), turning a cell that contains a real close obstacle into the far
    // plane downstream — i.e. hiding an obstacle, the one direction this module
    // must never fail in.
    if (!(disp > 1.0e-3F) || (static_cast<float>(x) - disp) < 0.0F) {
        return clip_max;
    }
    float z = fx_baseline / disp;
    if (z < clip_min) {
        z = clip_min;
    }
    if (z > clip_max) {
        z = clip_max;
    }
    return z;
}

// Apply the SAME invalid + clamp rule to an already-metric depth sample. This
// is the raw-D435i entry point: Z16 counts have been scaled to metres, and 0
// (the D435i's "no data" code) must become far, not near.
//
// Both guards are written as negated comparisons so NaN falls through to the
// far plane rather than being passed on. `!(z > 1e-6)` is TRUE for NaN, so the
// FIRST branch already catches it and returns clip_max; the same branch also
// catches 0 (the D435i "no data" code) and negatives. `!(z < clip_max)` is what
// catches +Inf. Only a finite, positive sample can reach the near branch
// (`z < clip_min` → clamp UP) or be returned unchanged.
[[nodiscard]] RT_DEPLOY_HD inline float sanitize_depth_m(float z, float clip_min,
                                                         float clip_max) noexcept {
    if (!(z > 1.0e-6F)) {  // zero, negative, or NaN -> no data -> far
        return clip_max;
    }
    if (z < clip_min) {
        return clip_min;
    }
    if (!(z < clip_max)) {  // above far, or Inf
        return clip_max;
    }
    return z;
}

// Reduce the source block [y0,y1) x [x0,x1) of a row-major metric-depth plane
// to a single output sample, per `mode`.
//
//   src    : full-resolution depth [m], already sanitized (no zeros/NaN)
//   stride : row stride of `src` in floats (may exceed the valid width when the
//            plane is a padded network output)
//
// Blocks are always non-empty: callers derive them from a floor-division map
// with x1 > x0 and y1 > y0.
//
// kAreaValid is NOT supported here and is treated as kArea. A sanitized plane has
// already collapsed "invalid" and "genuinely at clip_max" onto the same value, so
// the mask-weighted mean is not recoverable from it. The production paths use
// the fused reducers instead (reduce_block_raw_m for raw metres,
// reduce_disp_block for disparity); this function is the two-pass reference
// they are defined against.
[[nodiscard]] RT_DEPLOY_HD inline float reduce_block(const float* src, int stride, int x0, int x1,
                                                     int y0, int y1, Reduce mode) noexcept {
    if (mode == Reduce::kPoint) {
        const int cy = y0 + ((y1 - y0) >> 1);
        const int cx = x0 + ((x1 - x0) >> 1);
        return src[(static_cast<long>(cy) * stride) + cx];
    }
    if (mode == Reduce::kArea) {
        float     acc = 0.0F;
        const int n   = (y1 - y0) * (x1 - x0);
        for (int y = y0; y < y1; ++y) {
            const float* row = src + (static_cast<long>(y) * stride);
            for (int x = x0; x < x1; ++x) {
                acc += row[x];
            }
        }
        return acc / static_cast<float>(n);
    }
    // kMin — the closest surface wins. Invalid samples sit at clip_max, so a
    // cell reads "far" only when its whole footprint is far/invalid.
    float best = src[(static_cast<long>(y0) * stride) + x0];
    for (int y = y0; y < y1; ++y) {
        const float* row = src + (static_cast<long>(y) * stride);
        for (int x = x0; x < x1; ++x) {
            if (row[x] < best) {
                best = row[x];
            }
        }
    }
    return best;
}

// Mean of the VALID samples in a block, given raw values and a per-sample
// validity flag already computed by the caller. Shared by the raw-metres and the
// disparity reducers so the mask-weighted mean itself exists once.
//
//   sum_valid : sum of the far-clamped valid samples
//   n_valid   : how many were valid
// Returns clip_max when nothing in the footprint was measured — the honest "no
// data here" answer, and the same code training used for a dropped pixel.
[[nodiscard]] RT_DEPLOY_HD inline float finish_valid_mean(float sum_valid, int n_valid,
                                                          float clip_min,
                                                          float clip_max) noexcept {
    if (n_valid == 0) {
        return clip_max;
    }
    const float mean = sum_valid / static_cast<float>(n_valid);
    // Near clamp on the RESULT, not per sample: averaging pre-clamped values
    // would drag a cell that straddles clip_min upward. np.clip semantics.
    return (mean < clip_min) ? clip_min : mean;
}

// Reduce a block of RAW (un-sanitized) metric depth, fusing the invalid/clamp
// rule into the reduction.
//
// This is the entry point that supports kAreaValid, and the reason it has to
// exist: once a plane has been through sanitize_depth_m, an invalid sample and a
// genuine far reading are BOTH clip_max and can no longer be told apart, so the
// mask-weighted mean is not computable from it. Validity has to be decided
// before the remap.
//
// LOCKSTEP: for kMin / kArea / kPoint this is bit-identical to
//   sanitize_depth_m over the block, then reduce_block(..., mode).
// Mirrors reduce_disp_block's relationship to the two-pass disparity reference.
[[nodiscard]] RT_DEPLOY_HD inline float reduce_block_raw_m(const float* raw, int stride, int x0,
                                                           int x1, int y0, int y1, float clip_min,
                                                           float clip_max, Reduce mode) noexcept {
    if (mode == Reduce::kAreaValid) {
        float acc = 0.0F;
        int   n   = 0;
        for (int y = y0; y < y1; ++y) {
            const float* row = raw + (static_cast<long>(y) * stride);
            for (int x = x0; x < x1; ++x) {
                const float z = row[x];
                if (!depth_sample_valid(z)) {
                    continue;  // a hole contributes nothing, not a far reading
                }
                // Far-clamp BEFORE the mean: a saturated out-of-range reading is
                // real but must not drag the whole cell (65.54 m in a mean turns a
                // 0.5 m wall into 1.14 m at the deployed downsample ratio).
                acc += (z > clip_max) ? clip_max : z;
                ++n;
            }
        }
        return finish_valid_mean(acc, n, clip_min, clip_max);
    }
    if (mode == Reduce::kPoint) {
        const int cy = y0 + ((y1 - y0) >> 1);
        const int cx = x0 + ((x1 - x0) >> 1);
        return sanitize_depth_m(raw[(static_cast<long>(cy) * stride) + cx], clip_min, clip_max);
    }
    if (mode == Reduce::kArea) {
        float     acc = 0.0F;
        const int n   = (y1 - y0) * (x1 - x0);
        for (int y = y0; y < y1; ++y) {
            const float* row = raw + (static_cast<long>(y) * stride);
            for (int x = x0; x < x1; ++x) {
                acc += sanitize_depth_m(row[x], clip_min, clip_max);
            }
        }
        return acc / static_cast<float>(n);
    }
    float best = sanitize_depth_m(raw[(static_cast<long>(y0) * stride) + x0], clip_min, clip_max);
    for (int y = y0; y < y1; ++y) {
        const float* row = raw + (static_cast<long>(y) * stride);
        for (int x = x0; x < x1; ++x) {
            const float z = sanitize_depth_m(row[x], clip_min, clip_max);
            if (z < best) {
                best = z;
            }
        }
    }
    return best;
}

// Reduce one output cell's DISPARITY footprint straight to a metric-depth
// sample, fusing the per-pixel disparity→depth conversion into the reduction.
//
//   disp              : network-resolution disparity plane [px], row-major
//   stride            : row stride of `disp` in floats (the padded engine width)
//   [x0,x1) x [y0,y1) : the cell's source footprint, from vision::block_span
//   fx_baseline       : fx[px] * baseline[m] at the network resolution
//   clip_min/clip_max : near/far clip [m] — see vision::disp_to_depth_clamped
//   mode              : kMin / kArea / kPoint / kAreaValid (vision::Reduce)
//
// LOCKSTEP REQUIREMENT — this must stay bit-identical to the two-pass host
// reference
//     depth[y][x] = disp_to_depth_clamped(disp[y][x], x, fx_b, lo, hi)
//     out         = reduce_block(depth, stride, x0, x1, y0, y1, mode)
// above. The fused form exists ONLY to avoid materialising a
// full-resolution depth plane, which would cost an extra global-memory round
// trip per frame on the Orin's shared LPDDR5; it is not licensed to differ
// numerically. Any edit to reduce_block must be mirrored here. kAreaValid has no
// two-pass equivalent (see reduce_block); its branch mirrors reduce_block_raw_m.
//
// Bit-exactness on the GPU additionally rests on nvcc's defaults
// (-prec-div=true, no --use_fast_math): the conversion is a single IEEE divide,
// kArea accumulates in the same row-major order into a float accumulator, and
// there is no multiply-add for FMA contraction to fuse. Do not add fast-math to
// the CUDA target.
[[nodiscard]] RT_DEPLOY_HD inline float reduce_disp_block(const float* disp, int stride, int x0,
                                                          int x1, int y0, int y1,
                                                          float fx_baseline, float clip_min,
                                                          float clip_max, Reduce mode) noexcept {
    if (mode == Reduce::kAreaValid) {
        // Mask-weighted mean: only samples with a real correspondence contribute.
        // An occluded / unmatched pixel is a HOLE, and averaging it in at clip_max
        // biases the cell farther with local hole density (see Reduce::kAreaValid).
        float acc = 0.0F;
        int   n   = 0;
        for (int y = y0; y < y1; ++y) {
            const float* row = disp + (static_cast<long>(y) * stride);
            for (int x = x0; x < x1; ++x) {
                if (!disp_sample_valid(row[x], x)) {
                    continue;
                }
                const float z = fx_baseline / row[x];
                // Far-clamp before the mean, exactly as reduce_block_raw_m does.
                acc += (z > clip_max) ? clip_max : z;
                ++n;
            }
        }
        return finish_valid_mean(acc, n, clip_min, clip_max);
    }
    if (mode == Reduce::kPoint) {
        const int cy = y0 + ((y1 - y0) >> 1);
        const int cx = x0 + ((x1 - x0) >> 1);
        return disp_to_depth_clamped(disp[(static_cast<long>(cy) * stride) + cx], cx, fx_baseline,
                                     clip_min, clip_max);
    }
    if (mode == Reduce::kArea) {
        float     acc = 0.0F;
        const int n   = (y1 - y0) * (x1 - x0);
        for (int y = y0; y < y1; ++y) {
            const float* row = disp + (static_cast<long>(y) * stride);
            for (int x = x0; x < x1; ++x) {
                acc += disp_to_depth_clamped(row[x], x, fx_baseline, clip_min, clip_max);
            }
        }
        return acc / static_cast<float>(n);
    }
    // kMin — the closest surface wins. Invalid samples convert to clip_max, so a
    // cell reads "far" only when its whole footprint is far/invalid.
    float best = disp_to_depth_clamped(disp[(static_cast<long>(y0) * stride) + x0], x0, fx_baseline,
                                       clip_min, clip_max);
    for (int y = y0; y < y1; ++y) {
        const float* row = disp + (static_cast<long>(y) * stride);
        for (int x = x0; x < x1; ++x) {
            const float z = disp_to_depth_clamped(row[x], x, fx_baseline, clip_min, clip_max);
            if (z < best) {
                best = z;
            }
        }
    }
    return best;
}

// Inclusive-exclusive source span covering output index `o` of `out_n` when the
// source has `src_n` samples. Floor-division map, guaranteed non-empty and
// in-bounds for out_n <= src_n; for out_n > src_n the block degenerates to a
// single replicated sample rather than going empty.
RT_DEPLOY_HD inline void block_span(int o, int out_n, int src_n, int* lo, int* hi) noexcept {
    int a = static_cast<int>((static_cast<long>(o) * src_n) / out_n);
    int b = static_cast<int>((static_cast<long>(o + 1) * src_n) / out_n);
    if (a >= src_n) {
        a = src_n - 1;
    }
    if (b <= a) {
        b = a + 1;
    }
    *lo = a;
    *hi = b;
}

// Host-side full-plane downsample from RAW metres, fusing the invalid/clamp rule
// into the reduction. This is what the raw D435i path uses: it supports every
// mode including kAreaValid, and it needs no intermediate sanitized plane.
inline void downsample_depth_raw_m(const float* raw, int src_stride, int src_w, int src_h,
                                   float* dst, int out_w, int out_h, float clip_min, float clip_max,
                                   Reduce mode) noexcept {
    for (int oy = 0; oy < out_h; ++oy) {
        int y0 = 0;
        int y1 = 0;
        block_span(oy, out_h, src_h, &y0, &y1);
        for (int ox = 0; ox < out_w; ++ox) {
            int x0 = 0;
            int x1 = 0;
            block_span(ox, out_w, src_w, &x0, &x1);
            dst[(static_cast<long>(oy) * out_w) + ox] =
                reduce_block_raw_m(raw, src_stride, x0, x1, y0, y1, clip_min, clip_max, mode);
        }
    }
}

}  // namespace rt_deploy::vision
