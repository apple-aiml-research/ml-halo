//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// CUDA kernel launchers for the stereo-depth pipeline. Declared with plain C++
// signatures so stereo_depth_engine.cpp (compiled by the host compiler) can
// call them while the definitions live in stereo_kernels.cu (compiled by nvcc).
//
// All pointers are DEVICE-visible addresses (either a cudaMalloc'd device
// pointer on a discrete GPU, or the device alias of a mapped host allocation on
// an integrated Orin GPU — see stereo_depth_engine.cpp). Every launcher is
// asynchronous on `stream`; the caller synchronises.
//
// No depth arithmetic is defined here. The post-process routes every footprint
// through vision::block_span and vision::reduce_disp_block (depth_ops.hpp,
// compiled __host__ __device__ by nvcc), so the GPU path and the raw-D435i CPU
// path share ONE rule and cannot drift apart. In particular invalid / occluded
// disparity lands on clip_max (FAR), never 0, and the result is RAW METRES — no
// clip+log1p, that is the g1::ObservationBuilder image group's job downstream.

#include "rt_deploy/vision/depth_ops.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace rt_deploy::vision {

// ─── Launchers ───────────────────────────────────────────────────────────────

// IR (1-channel uint8) → network input tensor (1,3,padH,padW) FP32, raw 0-255.
// The single grey channel is replicated to all three (the model is RGB-trained)
// and the image is bottom-right replicate-padded to the 32-divisible (padH,padW)
// the engine was built for. Normalisation (2*x/255-1) is expected to be part of
// the exported network, so the kernel feeds raw 0-255 values.
cudaError_t launch_ir_to_chw(const std::uint8_t* ir, float* chw, int width, int height,
                             int pad_width, int pad_height, cudaStream_t stream) noexcept;

// Disparity (1,1,padH,padW) FP32 → fused depth, reduced to the small output grid
// (out_w × out_h) as float32 METRES. One thread per output cell; each reduces
// its own footprint via reduce_disp_block, so there are no atomics and no shared
// memory.
//
// `mode` picks the reduction and is uniform across the launch, so the branch
// costs nothing in warp divergence:
//   kMin       — closest depth in the footprint; a cell is far only when the
//                whole footprint is far/invalid. Conservative.
//   kArea      — mean over the footprint.
//   kPoint     — nearest-neighbour at the footprint centre (closest to the
//                training renderer, which point-samples one ray per pixel).
//   kAreaValid — mean over the matched samples only (the config default).
// The footprint map is vision::block_span, byte-for-byte the map the host-side
// raw path uses, and out-of-range output threads return early.
//
// `disp_stride` is the padded row stride (pad_width); only the top-left
// net_w × net_h region is read. Returns cudaErrorInvalidValue for non-positive
// extents rather than launching a kernel that would index outside the plane.
cudaError_t launch_disp_to_depth_reduce(const float* disp, float* depth_out, int net_w, int net_h,
                                        int disp_stride, int out_w, int out_h, float fx_baseline,
                                        float clip_min, float clip_max, Reduce mode,
                                        cudaStream_t stream) noexcept;

}  // namespace rt_deploy::vision
