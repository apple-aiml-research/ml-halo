//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// stereo_kernels.cu — preprocess + postprocess kernels for the depth pipeline.
//
// Compiled by nvcc. Two trivially-parallel kernels bracket the TensorRT
// inference; together they keep the whole disparity→depth path on the GPU so the
// only host traffic is the IR bytes in and the small metric-depth grid out.
//
// The depth maths is NOT duplicated here: disp_to_depth_reduce_kernel calls
// vision::block_span and vision::reduce_disp_block, which in turn call
// vision::disp_to_depth_clamped from depth_ops.hpp — nvcc compiles all three as
// __host__ __device__. That header is the single source of truth shared with the
// raw-D435i CPU path, so the GPU result cannot drift from the reference. See the LOCKSTEP note on reduce_disp_block before editing.

#include "rt_deploy/vision/stereo_kernels.hpp"

#include "rt_deploy/vision/depth_ops.hpp"

namespace rt_deploy::vision {

namespace {

// ─── Preprocess ──────────────────────────────────────────────────────────────

// One thread per output pixel of the padded CHW plane. Replicates the single IR
// channel to all 3 and bottom-right replicate-pads beyond the live image (the
// same edge clamp as cv2.BORDER_REPLICATE).
__global__ void ir_to_chw_kernel(const std::uint8_t* __restrict__ ir, float* __restrict__ chw,
                                 int width, int height, int pad_width, int pad_height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= pad_width || y >= pad_height) {
        return;
    }
    // Replicate-clamp into the live image for the padded border region.
    const int sx = x < width ? x : width - 1;
    const int sy = y < height ? y : height - 1;
    const float v = static_cast<float>(ir[sy * width + sx]);  // raw 0-255

    const int plane = pad_width * pad_height;
    const int off   = y * pad_width + x;
    chw[0 * plane + off] = v;
    chw[1 * plane + off] = v;
    chw[2 * plane + off] = v;
}

// ─── Postprocess ─────────────────────────────────────────────────────────────

// One thread per OUTPUT cell (84×48 = 4032 for the student_depth grid). Each
// thread owns its cell's entire footprint, so the reduction needs no atomics, no
// shared memory and no second pass; `mode` is uniform across the launch, so the
// mode branch inside reduce_disp_block never diverges within a warp.
//
// The footprint comes from vision::block_span — the identical map the host-side
// raw path uses — and the reduction itself is the shared
// vision::reduce_disp_block, so this kernel contributes no arithmetic of its own
// that could drift from the reference.
__global__ void disp_to_depth_reduce_kernel(const float* __restrict__ disp,
                                            float* __restrict__ depth_out, int net_w, int net_h,
                                            int disp_stride, int out_w, int out_h,
                                            float fx_baseline, float clip_min, float clip_max,
                                            Reduce mode) {
    const int ox = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int oy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (ox >= out_w || oy >= out_h) {
        return;
    }
    // Footprint of this output cell in the network-resolution disparity map.
    int x0 = 0;
    int x1 = 0;
    int y0 = 0;
    int y1 = 0;
    block_span(ox, out_w, net_w, &x0, &x1);
    block_span(oy, out_h, net_h, &y0, &y1);

    depth_out[(static_cast<long>(oy) * out_w) + ox] = reduce_disp_block(
        disp, disp_stride, x0, x1, y0, y1, fx_baseline, clip_min, clip_max, mode);
}

constexpr dim3 kBlock{16, 16};

dim3 grid_for(int w, int h) noexcept {
    return dim3{(static_cast<unsigned>(w) + kBlock.x - 1) / kBlock.x,
                (static_cast<unsigned>(h) + kBlock.y - 1) / kBlock.y};
}

}  // namespace

cudaError_t launch_ir_to_chw(const std::uint8_t* ir, float* chw, int width, int height,
                             int pad_width, int pad_height, cudaStream_t stream) noexcept {
    ir_to_chw_kernel<<<grid_for(pad_width, pad_height), kBlock, 0, stream>>>(
        ir, chw, width, height, pad_width, pad_height);
    return cudaGetLastError();
}

cudaError_t launch_disp_to_depth_reduce(const float* disp, float* depth_out, int net_w, int net_h,
                                        int disp_stride, int out_w, int out_h, float fx_baseline,
                                        float clip_min, float clip_max, Reduce mode,
                                        cudaStream_t stream) noexcept {
    // block_span assumes a non-empty source: with src_n <= 0 it would hand the
    // kernel a [-1, 0) span and read outside the plane. Reject here rather than
    // corrupt memory — the dims are validated at engine construction, so this is
    // a backstop, not an expected path.
    if (net_w <= 0 || net_h <= 0 || out_w <= 0 || out_h <= 0 || disp_stride < net_w) {
        return cudaErrorInvalidValue;
    }
    disp_to_depth_reduce_kernel<<<grid_for(out_w, out_h), kBlock, 0, stream>>>(
        disp, depth_out, net_w, net_h, disp_stride, out_w, out_h, fx_baseline, clip_min, clip_max,
        mode);
    return cudaGetLastError();
}

}  // namespace rt_deploy::vision
