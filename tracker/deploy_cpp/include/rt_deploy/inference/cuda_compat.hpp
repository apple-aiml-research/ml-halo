//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Cross-version CUDA runtime shims.
//
// A tiny translation layer for CUDA runtime APIs whose names or struct
// layouts changed between CUDA 12 (Orin / JetPack 6) and CUDA 13 (Thor /
// JetPack 7). Keeps the rest of the codebase free of version-guards.
//
// Only add shims here when:
//   (a) the API was renamed across versions, and
//   (b) -Werror would break one of the versions without the shim.
//
// Header-only; no ODR issues because all functions are static-inline.

#include <cstdint>
#include <cuda_runtime_api.h>

namespace rt_deploy::inference::cuda_compat {

// Wrapper for cudaGetDeviceProperties. The struct layout grew in CUDA 11.0
// and the runtime headers added a `_v2` symbol; CUDA 11/12 expose the
// unversioned name as a macro that resolves to `_v2`, while CUDA 13 dropped
// the `_v2` alias entirely. Calling the unversioned name works across all
// three because the macro is harmless when absent.
[[nodiscard]] inline cudaError_t get_device_properties(cudaDeviceProp& prop,
                                                        int device_id) noexcept {
    return cudaGetDeviceProperties(&prop, device_id);
}

// Compute-capability → "has fast native FP16" decision used by the plan
// builder. Pascal (sm_60) and later have native FP16; the first card with
// fast FP16 is sm_53 (Jetson TX1), so we use that as the cutoff.
//
// Blackwell (Thor, sm_101) trivially satisfies this, but we run the query
// anyway for diagnostic consistency.
[[nodiscard]] inline bool device_has_fast_fp16(int device_id) noexcept {
    cudaDeviceProp prop{};
    if (get_device_properties(prop, device_id) != cudaSuccess) {
        return false;
    }
    return prop.major > 6
           || (prop.major == 6 && prop.minor >= 0)
           || (prop.major == 5 && prop.minor >= 3);
}

}  // namespace rt_deploy::inference::cuda_compat
