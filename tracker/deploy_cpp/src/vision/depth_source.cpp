//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// depth_source — the factory + availability probe behind `depth_source: ...`.
//
// One place decides which producer fills the control loop's depth inbox, and
// one place decides what happens when the build cannot provide it. Both
// producers need librealsense2 (RT_DEPLOY_WITH_REALSENSE, without which this
// library is not built at all); kCleaned additionally needs TensorRT + CUDA
// (RT_DEPLOY_WITH_TRT). A RealSense-only build is valid, so every
// backend-specific include, type and construction below is compile-guarded and
// a missing backend is reported at runtime with the CMake option that enables
// it, instead of failing deep inside a driver.
//
// The factory hands back a started source or an error, never anything in
// between: a partially-started producer would leave a device open and a thread
// running behind a caller that thinks depth is off.

#include "rt_deploy/vision/depth_source.hpp"

#if defined(RT_DEPLOY_WITH_REALSENSE)
#    include "rt_deploy/vision/raw_depth_source.hpp"
#    if defined(RT_DEPLOY_WITH_TRT)
#        include "rt_deploy/vision/depth_pipeline.hpp"
#    endif
#endif

#include <memory>
#include <string>
#include <utility>

namespace rt_deploy::vision {

namespace {

// Backend presence as ordinary constants rather than #ifs threaded through the
// logic below, so the dispatch reads the same in every build configuration.
#if defined(RT_DEPLOY_WITH_REALSENSE)
constexpr bool kHaveRealSense = true;
#else
constexpr bool kHaveRealSense = false;
#endif

#if defined(RT_DEPLOY_WITH_TRT)
constexpr bool kHaveTrt = true;
#else
constexpr bool kHaveTrt = false;
#endif

// Name the exact CMake switch that is missing and how to turn it on. This
// string is what an operator sees when `depth_source: cleaned` meets a
// developer build, and "not available" on its own is not actionable.
[[nodiscard]] std::string unavailable_error(DepthSource s) {
    std::string msg{"depth source '"};
    msg += to_string(s);
    msg += "' is not compiled into this build — reconfigure cmake with ";
    if (!kHaveRealSense) {
        msg += "-DRT_DEPLOY_WITH_REALSENSE=ON (librealsense2)";
    }
    if (s == DepthSource::kCleaned && !kHaveTrt) {
        if (!kHaveRealSense) {
            msg += " and ";
        }
        msg += "-DRT_DEPLOY_WITH_TRT=ON (TensorRT + CUDA)";
    }
    msg += " and rebuild";
    return msg;
}

// Construct (but do not start) the producer named by cfg.source, or nullptr
// when this build cannot provide it. Callers gate on depth_source_available()
// first; the nullptr return is the belt to that braces.
[[nodiscard]] std::unique_ptr<IDepthSource> construct(const VisionConfig&                 cfg,
                                                      core::TripleBuffer<g1::DepthFrame>& inbox) {
#if defined(RT_DEPLOY_WITH_REALSENSE)
    switch (cfg.source) {
        case DepthSource::kCleaned:
#    if defined(RT_DEPLOY_WITH_TRT)
            return std::make_unique<DepthPipeline>(cfg, inbox);
#    else
            break;
#    endif
        case DepthSource::kRaw:
            return std::make_unique<RawDepthSource>(cfg, inbox);
    }
#else
    (void)cfg;
    (void)inbox;
#endif
    return nullptr;
}

}  // namespace

// ─── Availability ─────────────────────────────────────────────────────────────

bool depth_source_available(DepthSource s) noexcept {
    switch (s) {
        case DepthSource::kCleaned:
            return kHaveRealSense && kHaveTrt;
        case DepthSource::kRaw:
            return kHaveRealSense;
    }
    return false;
}

// ─── Factory ──────────────────────────────────────────────────────────────────

core::Result<std::unique_ptr<IDepthSource>>
make_depth_source(const VisionConfig& cfg, core::TripleBuffer<g1::DepthFrame>& inbox) {
    using ResT = core::Result<std::unique_ptr<IDepthSource>>;

    // Validate before anything is constructed, so a config that would silently
    // produce a wrong observation fails as a config error rather than as a
    // confusing driver error three layers down. Propagate the message verbatim:
    // validate() already names the offending field, and prefixing would bury it.
    if (auto v = cfg.validate(); !v) {
        return ResT::Error{v.error()};
    }

    if (!depth_source_available(cfg.source)) {
        return ResT::Error{unavailable_error(cfg.source)};
    }

    std::unique_ptr<IDepthSource> src = construct(cfg, inbox);
    if (src == nullptr) {
        return ResT::Error{unavailable_error(cfg.source)};
    }

    if (auto r = src->start(); !r) {
        std::string err{to_string(cfg.source)};
        err += " depth source: ";
        err += r.error();
        // Destroy before returning: the header promises no partially-started
        // source escapes, and ~IDepthSource() stops whatever start() managed to
        // spin up before it failed.
        src.reset();
        return ResT::Error{std::move(err)};
    }
    return ResT{std::move(src)};
}

}  // namespace rt_deploy::vision
