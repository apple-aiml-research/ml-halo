//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Config-driven ObservationBuilder — C++ parity with the Python
// deploy/utils/core/observation_builder.py.
//
// The observation layout is declared in the robot config's `observation:` YAML
// block (groups -> ordered terms). This builder assembles the ONNX feed tensors
// from that declaration, so a new policy iteration only needs a new config (+
// matching ONNX) — no code change — as long as every referenced term exists in
// the registries below. The SAME spec is intended to be emitted by training.
//
// Term kinds (mirror the Python registry):
//   STATE terms      — per-frame robot signals (base_ang_vel, proj_gravity,
//                      joint_pos_rel, joint_vel_rel, last_action,
//                      motion_anchor_ori_b). Computed each frame from
//                      RobotState; buffered for history.
//   REFERENCE terms  — tracking_command_partial (67, body-frame): sampled from
//                      the motion over a set of time indices; never buffered.
//
// Group layouts:
//   flat  -> concatenate each term for the current frame. A reference term may
//            set horizon:K: the current frame and the next K-1 frames, ordered
//            t+K-1 ... t (current frame last), flattened term_major (each
//            subfield over all K frames, then the next subfield).
//   stack -> (history, feature): state terms from their history buffer, oldest
//            row first; reference terms sampled over the full time-index
//            window (frames t+H-1 ... t, current frame last).
//
// The orientation anchor is the odometry pose (`anchor_mode: odom`, the only
// supported mode): motion_anchor_ori_b is measured against
// RobotState::odom_quaternion_wxyz.
//
// Dims are DERIVED from the spec (never hard-coded) and can be validated against
// the ONNX input names/shapes.

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/g1/config.hpp"
#include "rt_deploy/g1/depth_frame.hpp"
#include "rt_deploy/math/quat.hpp"
#include "rt_deploy/math/types.hpp"
#include "rt_deploy/policy/joint_remap.hpp"
#include "rt_deploy/policy/motion_loader.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rt_deploy::g1 {

// --------------------------------------------------------------------------- //
// Spec (parsed from the YAML `observation:` block)
// --------------------------------------------------------------------------- //
struct TermCfg {
    std::string name;
    int         horizon{0};            // reference terms, flat groups
};

struct GroupCfg {
    // Sentinel for "the config did not set a clip". Negative so it is not a
    // distance and cannot be mistaken for one: 0.0 would read as an obstacle
    // pressed against the lens, which is the most dangerous value this field
    // can take.
    static constexpr float kClipUnset = -1.0F;

    std::string          name;
    bool                 is_stack{false};         // stack vs flat
    int                  history{0};
    std::vector<TermCfg> terms{};
    // image-layout ('image'): a raw sensor frame (depth) fed via set_depth(),
    // NOT assembled from terms (terms stays empty). img_rows*img_cols must equal
    // DepthFrame::kCount; encode = clip([clip_min,clip_max]) -> log1p.
    bool                 is_image{false};
    int                  img_rows{0};
    int                  img_cols{0};
    // Depth encode clip [m]. MANDATORY for an image group — there is
    // deliberately NO usable default.
    //
    // This must equal the clip the policy's own training env used. Anything else
    // mis-scales every pixel while the shape, num_obs and the ONNX signature all
    // still match, so NOTHING downstream can detect it. Encoding with 3.0 m a
    // policy trained with a 2.5 m far clip shifts every invalid / no-return pixel
    // by ~12% of the encoded range, and invalid pixels dominate a real D435.
    //
    // The HALO depth student trains with DEPTH_CLIP = 0.2/2.5
    // (motion_tracking_scene_distill_cfg.py); g1/depth_clip_registry.hpp enforces
    // that value. This struct only carries what the config declared.
    float                clip_min{kClipUnset};
    float                clip_max{kClipUnset};
    // Control steps each depth frame is HELD for => the policy is fed depth at
    // control_hz / obs_decim. 1 = a fresh frame every control step.
    //
    // The depth student trains with obs_decim=2 in its depth observation term
    // (25 Hz at a 50 Hz control rate), so feeding depth at the full control rate
    // is a train/deploy mismatch on the ONLY exteroceptive input.
    //
    // WHERE IT APPLIES — and this is the part that is easy to get backwards:
    //   SIM      the renderer is SYNCHRONOUS with the control loop, so without
    //            this it would produce a fresh frame every tick, i.e. twice the
    //            rate training used. Applied by decimating the render cadence.
    //   HARDWARE NOT applied. The camera is asynchronous and already slower than
    //            the control loop, so the natural sensor rate IS the hold; the
    //            right frame is simply the newest one, and holding on top of that
    //            would only add latency for nothing.
    // Default 1 (inert) because 1 is a harmless value, unlike a defaulted clip
    // which is a wrong one. The resolved rate is printed at startup so an omitted
    // obs_decim stays visible rather than silently doubling the depth rate.
    int                  obs_decim{1};
};

struct ObsSpec {
    int                   history_length{1};
    std::vector<GroupCfg> groups;                 // ordered; group name == ONNX input name
};

// Parse the `observation:` node of a config YAML file.
[[nodiscard]] core::Result<ObsSpec> load_observation_spec(std::string_view yaml_path);

// --------------------------------------------------------------------------- //
// Term registry
// --------------------------------------------------------------------------- //
struct Subfield { const char* name; int dim; };

// dim of a state term (0 => not a state term).
[[nodiscard]] int state_term_dim(const std::string& name);
// reference term subfield layout (for term-major flatten); nullptr if not a ref term.
[[nodiscard]] const std::vector<Subfield>* reference_subfields(const std::string& name);
[[nodiscard]] inline bool is_reference_term(const std::string& name) {
    return reference_subfields(name) != nullptr;
}

// --------------------------------------------------------------------------- //
// Dynamic circular history buffer (runtime feature dim + length).
// Mirrors the Python HistoryBuffer: first push fills all slots; get_history
// returns rows oldest->newest.
// --------------------------------------------------------------------------- //
class DynHistoryBuffer {
public:
    void init(int feat, int len) {
        feat_ = feat; len_ = len; head_ = 0; num_pushes_ = 0;
        buffer_.setZero(len, feat);
    }
    void reset() noexcept { head_ = 0; num_pushes_ = 0; buffer_.setZero(); }

    void push(const Eigen::Ref<const Eigen::VectorXf>& v) noexcept {
        if (num_pushes_ == 0) {
            for (int r = 0; r < len_; ++r) buffer_.row(r) = v.transpose();
            head_ = 0; num_pushes_ = len_;
        } else {
            buffer_.row(head_) = v.transpose();
            head_ = (head_ + 1) % len_;
            ++num_pushes_;
        }
    }
    void get_history(Eigen::Ref<Eigen::MatrixXf> out) const noexcept {
        const int tail = len_ - head_;
        if (tail > 0) out.topRows(tail) = buffer_.bottomRows(tail);
        if (head_ > 0) out.bottomRows(head_) = buffer_.topRows(head_);
    }
    [[nodiscard]] int feat() const noexcept { return feat_; }
    [[nodiscard]] int len() const noexcept { return len_; }

private:
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> buffer_{};
    int feat_{0}, len_{0}, head_{0};
    long num_pushes_{0};
};

// --------------------------------------------------------------------------- //
// One built ONNX input tensor.
// flat  -> rows=1, cols=dim.   stack -> rows=history, cols=feature.
// data is row-major, length rows*cols.
// --------------------------------------------------------------------------- //
struct GroupTensor {
    std::string     name;
    int             rows{0};
    int             cols{0};
    Eigen::VectorXf data;   // row-major
};

// --------------------------------------------------------------------------- //
// Config-driven builder.
// --------------------------------------------------------------------------- //
class ObservationBuilder {
public:
    ObservationBuilder(const Config& cfg, ObsSpec spec,
                       const policy::JointRemap<kDof>& remap,
                       const policy::MotionData* motion);

    void set_motion(const policy::MotionData* motion) noexcept { motion_ = motion; }
    [[nodiscard]] const policy::MotionData* motion() const noexcept { return motion_; }
    [[nodiscard]] policy::MotionData* mutable_motion() const noexcept {
        return const_cast<policy::MotionData*>(motion_);
    }
    [[nodiscard]] const policy::JointRemap<kDof>& remap() const noexcept { return remap_; }
    [[nodiscard]] const ObsSpec& spec() const noexcept { return spec_; }
    // Number of floats build() will produce for spec().groups[gi] (0 if out of
    // range), computed from the spec alone so it can be checked against the
    // engine's input sizes before the first tick.
    [[nodiscard]] std::size_t group_size(std::size_t gi) const;
    void reset() noexcept { for (auto& kv : buffers_) kv.second.reset(); }

    // Latest metric-depth frame for the 'image' group (e.g. student_depth). The
    // caller (the control loop, once it drains a depth inbox) points this at the
    // freshest frame each tick; nullptr => the image group emits the far-plane
    // fallback. Not owned; must outlive the next build().
    void set_depth(const DepthFrame* d) noexcept { depth_ = d; }
    [[nodiscard]] const DepthFrame* depth() const noexcept { return depth_; }

    // Assemble all groups for this tick. `last_action_train` is the previous
    // tick's action in training order (the torque-clipped action from the
    // decoder; zeros on the first tick).
    void build(const math::RobotState<kDof>& state,
               const Eigen::Ref<const Eigen::Matrix<float, kDof, 1>>& last_action_train,
               std::int32_t inference_tick);

    [[nodiscard]] const std::vector<GroupTensor>& outputs() const noexcept { return outputs_; }
    // Clip every built group tensor to [-bound, bound] in place.
    void clip_observations(float bound) noexcept {
        for (auto& g : outputs_) g.data = g.data.cwiseMax(-bound).cwiseMin(bound);
    }
    [[nodiscard]] const GroupTensor* group(const std::string& name) const noexcept {
        for (const auto& g : outputs_) if (g.name == name) return &g;
        return nullptr;
    }

private:
    void compute_frame(const math::RobotState<kDof>& state,
                       const Eigen::Ref<const Eigen::Matrix<float, kDof, 1>>& last_action_train,
                       std::int32_t inference_tick);
    [[nodiscard]] const Eigen::VectorXf& frame_value(const std::string& name) const;
    void reference_rows(const std::string& name, const std::int32_t* idx, int n,
                        Eigen::MatrixXf& out) const;

    Config                          cfg_;
    ObsSpec                         spec_;
    policy::JointRemap<kDof>        remap_;
    const policy::MotionData*       motion_{nullptr};
    Eigen::Matrix<float, kDof, 1>   default_angles_{};
    float                           control_dt_{0.02F};
    int                             max_history_{1};

    std::unordered_map<std::string, Eigen::VectorXf>  frame_;   // state-term values this tick
    std::unordered_map<std::string, DynHistoryBuffer> buffers_;
    std::vector<std::int32_t>       time_indices_;              // max_history_ long
    std::vector<GroupTensor>        outputs_;
    const DepthFrame*               depth_{nullptr};            // latest depth for image groups
    bool                            allocated_{false};
};

}  // namespace rt_deploy::g1
