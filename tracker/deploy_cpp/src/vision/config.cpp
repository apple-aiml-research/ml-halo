//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/vision/config.hpp"

#include "rt_deploy/g1/depth_frame.hpp"
#include "rt_deploy/policy/yaml_helpers.hpp"
#include "rt_deploy/vision/stereo_frame.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <sstream>
#include <string>

namespace rt_deploy::vision {

namespace {

// {ROOT_DIR} in plan_path is expanded exactly like g1::load_config expands
// xml_path, so a depth YAML can be moved next to the robot configs and still
// point at the same TensorRT plan.
using policy::yaml_helpers::replace_all;
using policy::yaml_helpers::resolve_root_dir;

// Read an optional scalar; leaves `out` untouched when the key is absent, so a
// partial YAML inherits the struct's defaults rather than zeroing fields.
template <typename T>
void get(const YAML::Node& node, const char* key, T& out) {
    if (const auto n = node[key]; n && !n.IsNull()) {
        out = n.as<T>();
    }
}

void load_thread_rt(const YAML::Node& node, ThreadRtConfig& out) {
    if (!node || !node.IsMap()) {
        return;
    }
    get(node, "fifo_priority", out.fifo_priority);
    get(node, "cpu_affinity", out.cpu_affinity);
    get(node, "lock_memory", out.lock_memory);
}

// ─── Enum spellings ──────────────────────────────────────────────────────────
// Both parsers reject an unrecognised value instead of falling back to the
// default. A typo in `depth_source` would otherwise silently run the *other*
// estimator, and a typo in `downsample` would silently change the reduction the
// policy sees — both produce a plausible-looking depth image and a robot that
// misbehaves for no visible reason.

[[nodiscard]] core::Result<DepthSource> parse_depth_source(const std::string& raw) {
    if (raw == "cleaned") {
        return DepthSource::kCleaned;
    }
    if (raw == "raw") {
        return DepthSource::kRaw;
    }
    return core::unexpected<std::string>(
        "vision config: depth_source must be 'cleaned' or 'raw', got '" + raw + "'");
}

[[nodiscard]] core::Result<DownsampleMode> parse_downsample(const std::string& raw) {
    if (raw == "min") {
        return DownsampleMode::kMin;
    }
    if (raw == "area") {
        return DownsampleMode::kArea;
    }
    if (raw == "point") {
        return DownsampleMode::kPoint;
    }
    if (raw == "area_valid") {
        return DownsampleMode::kAreaValid;
    }
    return core::unexpected<std::string>(
        "vision config: downsample must be 'min', 'area', 'point' or 'area_valid', got '" + raw
        + "'");
}

[[nodiscard]] core::Result<void, std::string> validate_thread_rt(const ThreadRtConfig& rt,
                                                                 const char*           which) {
    if (rt.fifo_priority < 0 || rt.fifo_priority > 99) {
        std::ostringstream os;
        os << "vision config: " << which << ".fifo_priority=" << rt.fifo_priority
           << " is out of range; use 0 (SCHED_OTHER) or 1..99, and keep it strictly below the "
              "control loop's --rt_prio";
        return core::unexpected<std::string>(os.str());
    }
    return {};
}

}  // namespace

// ─── Enum names ──────────────────────────────────────────────────────────────

std::string_view to_string(DepthSource s) noexcept {
    switch (s) {
        case DepthSource::kCleaned: return "cleaned";
        case DepthSource::kRaw:     return "raw";
    }
    return "unknown";
}

std::string_view to_string(DownsampleMode m) noexcept {
    switch (m) {
        case DownsampleMode::kMin:   return "min";
        case DownsampleMode::kArea:  return "area";
        case DownsampleMode::kPoint: return "point";
        case DownsampleMode::kAreaValid: return "area_valid";
    }
    return "unknown";
}

// ─── Validation ──────────────────────────────────────────────────────────────

core::Result<void, std::string> VisionConfig::validate() const {
    // The output geometry is the one mismatch that cannot be detected later: a
    // wrong-shaped plane still copies into the DepthFrame slot and the policy
    // still runs, on a silently scrambled image. Compare against the sink's own
    // constants so the two can never drift.
    if (out_width != g1::DepthFrame::kCols || out_height != g1::DepthFrame::kRows) {
        std::ostringstream os;
        os << "vision config: out_width x out_height = " << out_width << "x" << out_height
           << ", but the depth student's student_depth observation is fixed at "
           << g1::DepthFrame::kCols << "x" << g1::DepthFrame::kRows
           << " (g1::DepthFrame::kCols x kRows); set out_width: " << g1::DepthFrame::kCols
           << " and out_height: " << g1::DepthFrame::kRows;
        return core::unexpected<std::string>(os.str());
    }

    if (clip_min_m <= 0.0F) {
        std::ostringstream os;
        os << "vision config: clip_min_m=" << clip_min_m
           << " must be > 0 (it is a metric near plane; 0 would make the far-plane "
              "invalid encoding indistinguishable from a valid sample)";
        return core::unexpected<std::string>(os.str());
    }
    if (clip_max_m <= clip_min_m) {
        std::ostringstream os;
        os << "vision config: clip_max_m=" << clip_max_m << " must be > clip_min_m=" << clip_min_m;
        return core::unexpected<std::string>(os.str());
    }

    if (width <= 0 || height <= 0 || fps <= 0) {
        std::ostringstream os;
        os << "vision config: width/height/fps must be positive, got " << width << "x" << height
           << " @" << fps;
        return core::unexpected<std::string>(os.str());
    }
    // The capture path memcpys width*height bytes into a fixed-size StereoFrame
    // slot with no runtime extent check, so an oversized mode would corrupt the
    // TripleBuffer rather than fail. Unreachable today (rs2::pipeline::start()
    // rejects an unsupported IR mode first), but this is the cheap guard that
    // keeps it unreachable if a larger sensor is ever added.
    if (static_cast<std::size_t>(width) * static_cast<std::size_t>(height) > kMaxPixels) {
        std::ostringstream os;
        os << "vision config: " << width << "x" << height << " exceeds the StereoFrame capacity of "
           << kMaxWidth << "x" << kMaxHeight << " (" << kMaxPixels << " px)";
        return core::unexpected<std::string>(os.str());
    }
    // NOTE — deliberately NOT rejecting a capture size that is off a multiple of
    // 32. The /32 rule belongs to the PLAN, not to the capture: the plan is built
    // at 32-divisible padded dims and the engine replicate-pads the live IR up to
    // them (launch_ir_to_chw + the `cfg.width > pad_w` check in
    // make_stereo_depth_engine). The default mode 424x240 is itself off
    // /32 (424 = 13.25*32), as are native D435i modes like 480x270, so a /32
    // reject here would refuse every resolution the camera actually offers.
    // Whether the capture fits the plan is checked where pad_w/pad_h are known.

    if (source == DepthSource::kCleaned && plan_path.empty()) {
        return core::unexpected<std::string>(
            "vision config: depth_source 'cleaned' needs plan_path (a stereo TensorRT .plan "
            "built with onnx_to_trt); use depth_source 'raw' for the no-GPU on-chip path");
    }

    if (rate_hz < 0.0F) {
        std::ostringstream os;
        os << "vision config: rate_hz=" << rate_hz << " must be >= 0 (0 means free-run)";
        return core::unexpected<std::string>(os.str());
    }

    if (auto r = validate_thread_rt(capture_rt, "capture_rt"); !r) {
        return core::unexpected<std::string>(r.error());
    }
    if (auto r = validate_thread_rt(inference_rt, "inference_rt"); !r) {
        return core::unexpected<std::string>(r.error());
    }

    // A clip other than the depth student's 0.2/2.5 is deliberately NOT
    // rejected here: retrained policies legitimately move it, and this file has no
    // way to know which policy will run. What IS enforced is agreement — the
    // deploy app cross-checks these against the policy config's image-group clip,
    // and that clip against the training clip (g1/depth_clip_registry).
    // (Range/ordering of the two bounds is checked at the top of this function.)
    return {};
}

// ─── Loader ──────────────────────────────────────────────────────────────────

core::Result<VisionConfig> load_vision_config(std::string_view yaml_path) {
    namespace fs = std::filesystem;
    const fs::path path{std::string{yaml_path}};
    if (!fs::exists(path)) {
        return core::unexpected<std::string>("vision config file not found: " + path.string());
    }

    YAML::Node node;
    try {
        node = YAML::LoadFile(path.string());
    } catch (const YAML::Exception& e) {
        return core::unexpected<std::string>(std::string{"vision config: YAML parse error: "}
                                             + e.what());
    }

    VisionConfig cfg;
    try {
        // ── Which producer ──
        if (const auto n = node["depth_source"]; n && !n.IsNull()) {
            auto r = parse_depth_source(n.as<std::string>());
            if (!r) {
                return core::unexpected<std::string>(r.error());
            }
            cfg.source = r.value();
        }

        // ── Camera / network input resolution ──
        get(node, "width", cfg.width);
        get(node, "height", cfg.height);
        get(node, "fps", cfg.fps);
        get(node, "emitter", cfg.emitter);
        get(node, "serial", cfg.serial);

        // ── Inference (kCleaned only) ──
        if (const auto n = node["plan_path"]; n && n.IsScalar()) {
            cfg.plan_path = replace_all(n.as<std::string>(), "{ROOT_DIR}", resolve_root_dir(path));
        }
        get(node, "device_id", cfg.device_id);
        get(node, "use_cuda_graph", cfg.use_cuda_graph);

        // ── Output ──
        get(node, "out_width", cfg.out_width);
        get(node, "out_height", cfg.out_height);
        get(node, "clip_min_m", cfg.clip_min_m);
        get(node, "clip_max_m", cfg.clip_max_m);
        if (const auto n = node["downsample"]; n && !n.IsNull()) {
            auto r = parse_downsample(n.as<std::string>());
            if (!r) {
                return core::unexpected<std::string>(r.error());
            }
            cfg.downsample = r.value();
        }
        get(node, "fx_override", cfg.fx_override);
        get(node, "baseline_m_override", cfg.baseline_m_override);

        // ── Real-time thread placement ──
        load_thread_rt(node["capture_rt"], cfg.capture_rt);
        load_thread_rt(node["inference_rt"], cfg.inference_rt);

        // ── Run control ──
        get(node, "rate_hz", cfg.rate_hz);
        get(node, "profile_period_s", cfg.profile_period_s);
    } catch (const YAML::Exception& e) {
        return core::unexpected<std::string>(std::string{"vision config: field error: "}
                                             + e.what());
    }

    // validate() is deliberately NOT called here. `--depth_source` overrides the
    // YAML's choice after loading, so a plan-less file that selects 'cleaned' is
    // still a legal input to a run that overrides to 'raw'. The factory
    // (make_depth_source) validates once the effective config is settled.
    return cfg;
}

}  // namespace rt_deploy::vision
