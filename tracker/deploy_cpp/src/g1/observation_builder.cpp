//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/g1/observation_builder.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace rt_deploy::g1 {

// --------------------------------------------------------------------------- //
// Term registry
// --------------------------------------------------------------------------- //
int state_term_dim(const std::string& name) {
    if (name == "base_ang_vel")         return 3;
    if (name == "proj_gravity")         return 3;
    if (name == "joint_pos_rel")        return static_cast<int>(kDof);
    if (name == "joint_vel_rel")        return static_cast<int>(kDof);
    if (name == "last_action")          return static_cast<int>(kDof);
    if (name == "motion_anchor_ori_b")  return 6;
    return 0;  // not a state term
}

const std::vector<Subfield>* reference_subfields(const std::string& name) {
    static const std::vector<Subfield> kPartial{
        {"joint_pos", static_cast<int>(kDof)}, {"joint_vel", static_cast<int>(kDof)},
        {"root_gravity_b", 3}, {"root_lin_vel_b", 3}, {"root_ang_vel_b", 3}};      // 67
    if (name == "tracking_command_partial") return &kPartial;
    return nullptr;
}

namespace {
int reference_dim(const std::string& name) {
    const auto* sf = reference_subfields(name);
    if (sf == nullptr) return 0;
    int d = 0;
    for (const auto& s : *sf) d += s.dim;
    return d;
}
}  // namespace

// --------------------------------------------------------------------------- //
// Spec parsing
// --------------------------------------------------------------------------- //
namespace {

// Sanity ceiling on an image group's far clip, in metres. Same value as the
// Python ObservationSpec (_MAX_PLAUSIBLE_DEPTH_FAR_M). Not the contract (that is
// the training clip, see depth_clip_registry); it catches a unit slip such as
// millimetres for metres.
constexpr float kMaxPlausibleDepthFarM = 10.0F;

// Body of load_observation_spec. yaml-cpp reports a wrongly typed value by
// throwing YAML::Exception; the wrapper below turns that into a config error.
core::Result<ObsSpec> parse_observation_spec(const YAML::Node& root) {
    if (!root["observation"]) {
        return typename core::Result<ObsSpec>::Error{"config has no 'observation:' block"};
    }
    const YAML::Node obs = root["observation"];

    ObsSpec spec;
    spec.history_length = obs["history_length"] ? obs["history_length"].as<int>() : 1;
    // The anchor is the odometry pose; `odom` is the only mode and it is required,
    // so a config written for another anchor is refused rather than reinterpreted.
    const std::string anchor_mode =
        obs["anchor_mode"] ? obs["anchor_mode"].as<std::string>() : std::string{};
    if (anchor_mode != "odom") {
        return typename core::Result<ObsSpec>::Error{
            "observation.anchor_mode must be 'odom', got '" + anchor_mode + "'"};
    }

    if (!obs["groups"]) {
        return typename core::Result<ObsSpec>::Error{"observation has no 'groups'"};
    }
    for (const auto& gkv : obs["groups"]) {
        GroupCfg g;
        g.name = gkv.first.as<std::string>();
        const YAML::Node gn = gkv.second;
        // `layout` is required, as in the Python ObservationSpec: a group whose
        // layout is missing would otherwise be read as flat without a word.
        if (!gn["layout"]) {
            return typename core::Result<ObsSpec>::Error{
                "group '" + g.name + "': `layout` is required (flat|stack|image)"};
        }
        const std::string layout = gn["layout"].as<std::string>();
        // image layout: a raw depth frame fed via set_depth(); no proprio/reference
        // terms. Group name must equal the ONNX input name (e.g. student_depth).
        if (layout == "image") {
            g.is_image = true;
            if (!gn["height"] || !gn["width"]) {
                return typename core::Result<ObsSpec>::Error{
                    "image group '" + g.name + "' needs explicit `height` and `width`"};
            }
            g.img_rows = gn["height"].as<int>();
            g.img_cols = gn["width"].as<int>();
            if (g.img_rows != DepthFrame::kRows || g.img_cols != DepthFrame::kCols) {
                return typename core::Result<ObsSpec>::Error{
                    "image group '" + g.name + "' is " + std::to_string(g.img_rows) + "x"
                    + std::to_string(g.img_cols) + " (height x width); the depth frame is "
                    + std::to_string(DepthFrame::kRows) + "x" + std::to_string(DepthFrame::kCols)};
            }
            // `clip` is MANDATORY and has no default. It must equal the clip the
            // policy's own training env used; a wrong value mis-scales every
            // pixel while every shape still matches, so nothing downstream can
            // catch it. See GroupCfg::clip_min for why a default is unsafe.
            if (!gn["clip"] || !gn["clip"].IsSequence() || gn["clip"].size() != 2) {
                return typename core::Result<ObsSpec>::Error{
                    "image group '" + g.name
                    + "' must set an explicit `clip: [near_m, far_m]` (metres), equal to the clip "
                      "its checkpoint trained with — 0.2/2.5 for the HALO depth student "
                      "(DEPTH_CLIP in motion_tracking_scene_distill_cfg.py). There is deliberately "
                      "no default: encoding with the wrong one mis-scales every pixel with no "
                      "detectable symptom."};
            }
            g.clip_min = gn["clip"][0].as<float>();
            g.clip_max = gn["clip"][1].as<float>();
            // Range-check the VALUE, not just its presence. A non-positive near
            // clip makes log1p's input span meaningless, and near >= far inverts
            // the encode. Both would otherwise sail through to the policy.
            if (!(g.clip_min > 0.0F) || !std::isfinite(g.clip_min)
                || !std::isfinite(g.clip_max)) {
                return typename core::Result<ObsSpec>::Error{
                    "image group '" + g.name + "': clip bounds must be finite positive distances, got ["
                    + std::to_string(g.clip_min) + ", " + std::to_string(g.clip_max) + "]"};
            }
            if (!(g.clip_max > g.clip_min)) {
                return typename core::Result<ObsSpec>::Error{
                    "image group '" + g.name + "': clip far must exceed clip near, got ["
                    + std::to_string(g.clip_min) + ", " + std::to_string(g.clip_max) + "]"};
            }
            if (g.clip_max > kMaxPlausibleDepthFarM) {
                return typename core::Result<ObsSpec>::Error{
                    "image group '" + g.name + "': clip far " + std::to_string(g.clip_max)
                    + " m exceeds " + std::to_string(kMaxPlausibleDepthFarM)
                    + " m (a unit slip? the clip is in metres)"};
            }
            if (gn["encode"] && gn["encode"].as<std::string>() != "log1p") {
                return typename core::Result<ObsSpec>::Error{
                    "image group '" + g.name + "': only encode=log1p is supported"};
            }
            // obs_decim: control steps to hold each frame. Range-checked rather
            // than clamped — max(1, x) alone would quietly turn 0, -1 and a typo'd
            // value all into 1, which is the value that means "no hold" and so
            // hides the mistake completely.
            if (gn["obs_decim"] && !gn["obs_decim"].IsNull()) {
                g.obs_decim = gn["obs_decim"].as<int>();
                if (g.obs_decim < 1) {
                    return typename core::Result<ObsSpec>::Error{
                        "image group '" + g.name + "': obs_decim must be >= 1, got "
                        + std::to_string(g.obs_decim)
                        + ". 1 = a fresh frame every control step; 2 = 25 Hz at a 50 Hz control "
                          "rate, which is what the depth student trains at."};
                }
            }
            spec.groups.push_back(std::move(g));
            continue;  // no terms for image groups
        }
        g.is_stack = (layout == "stack");
        if (layout != "flat" && layout != "stack") {
            return typename core::Result<ObsSpec>::Error{"group '" + g.name + "': layout must be flat|stack|image"};
        }
        g.history = gn["history"] ? gn["history"].as<int>() : spec.history_length;
        if (g.is_stack && g.history < 1) {
            return typename core::Result<ObsSpec>::Error{
                "stack group '" + g.name + "': history must be >= 1"};
        }
        if (!gn["terms"] || !gn["terms"].IsSequence()) {
            return typename core::Result<ObsSpec>::Error{
                "group '" + g.name + "': `terms` must be a list"};
        }
        for (const auto& tn : gn["terms"]) {
            TermCfg t;
            if (tn.IsScalar()) {
                t.name = tn.as<std::string>();
            } else {
                t.name = tn["name"].as<std::string>();
                for (const auto& kv : tn) {
                    const std::string key = kv.first.as<std::string>();
                    if (key != "name" && key != "horizon" && key != "flatten") {
                        return typename core::Result<ObsSpec>::Error{
                            "term '" + t.name + "': unsupported key '" + key
                            + "' (allowed: name, horizon, flatten)"};
                    }
                }
                if (tn["horizon"]) t.horizon = tn["horizon"].as<int>();
                if (tn["flatten"] && tn["flatten"].as<std::string>() != "term_major") {
                    return typename core::Result<ObsSpec>::Error{
                        "term '" + t.name + "': flatten must be term_major, got '"
                        + tn["flatten"].as<std::string>() + "'"};
                }
            }
            if (state_term_dim(t.name) == 0 && !is_reference_term(t.name)) {
                return typename core::Result<ObsSpec>::Error{"unknown observation term '" + t.name + "'"};
            }
            g.terms.push_back(std::move(t));
        }
        spec.groups.push_back(std::move(g));
    }

    // A stack group reads its rows from the shared time-index window and the
    // shared per-term history buffers, whose length is the longest history any
    // group asks for. A stack group shorter than that would read the wrong rows,
    // so every stack group must use exactly that length (all shipped configs do).
    int window = std::max(1, spec.history_length);
    for (const auto& g : spec.groups) {
        if (g.is_stack) window = std::max(window, g.history);
        for (const auto& t : g.terms) {
            window = std::max(window, t.horizon);
        }
    }
    for (const auto& g : spec.groups) {
        if (g.is_stack && g.history != window) {
            return typename core::Result<ObsSpec>::Error{
                "stack group '" + g.name + "' has history " + std::to_string(g.history)
                + " but the longest history/horizon in the spec is " + std::to_string(window)
                + "; mixed history lengths are not supported"};
        }
    }
    return spec;
}

}  // namespace

core::Result<ObsSpec> load_observation_spec(std::string_view yaml_path) {
    try {
        return parse_observation_spec(YAML::LoadFile(std::string{yaml_path}));
    } catch (const YAML::Exception& e) {
        return typename core::Result<ObsSpec>::Error{"observation spec " + std::string{yaml_path}
                                                     + ": " + e.what()};
    }
}

// --------------------------------------------------------------------------- //
// Builder
// --------------------------------------------------------------------------- //
ObservationBuilder::ObservationBuilder(const Config& cfg, ObsSpec spec,
                                       const policy::JointRemap<kDof>& remap,
                                       const policy::MotionData* motion)
    : cfg_{cfg}, spec_{std::move(spec)}, remap_{remap}, motion_{motion},
      default_angles_{cfg.default_angles}, control_dt_{cfg.control_dt} {
    // time-index window length = the largest history any group needs.
    max_history_ = std::max(1, spec_.history_length);
    for (const auto& g : spec_.groups) {
        if (g.is_stack) max_history_ = std::max(max_history_, g.history);
        for (const auto& t : g.terms) {
            if (is_reference_term(t.name)) max_history_ = std::max(max_history_, t.horizon);
        }
    }
    time_indices_.assign(static_cast<std::size_t>(max_history_), 0);

    // Allocate a history buffer for each state term of a stack group.
    for (const auto& g : spec_.groups) {
        if (!g.is_stack) continue;
        for (const auto& t : g.terms) {
            if (is_reference_term(t.name)) continue;
            auto& buf = buffers_[t.name];
            if (buf.len() < g.history) buf.init(state_term_dim(t.name), g.history);
        }
    }
}

const Eigen::VectorXf& ObservationBuilder::frame_value(const std::string& name) const {
    return frame_.at(name);
}

// Mirrors build()'s three layouts exactly.
std::size_t ObservationBuilder::group_size(std::size_t gi) const {
    if (gi >= spec_.groups.size()) return 0;
    const GroupCfg& g = spec_.groups[gi];
    if (g.is_image) {
        return static_cast<std::size_t>(g.img_rows) * static_cast<std::size_t>(g.img_cols);
    }
    std::size_t n = 0;
    if (g.is_stack) {
        for (const auto& t : g.terms)
            n += static_cast<std::size_t>(is_reference_term(t.name) ? reference_dim(t.name)
                                                                    : state_term_dim(t.name));
        return static_cast<std::size_t>(std::max(g.history, 0)) * n;
    }
    for (const auto& t : g.terms) {
        if (is_reference_term(t.name)) {
            const int k = t.horizon > 0 ? t.horizon : 1;
            n += static_cast<std::size_t>(k) * static_cast<std::size_t>(reference_dim(t.name));
        } else {
            n += static_cast<std::size_t>(state_term_dim(t.name));
        }
    }
    return n;
}

void ObservationBuilder::compute_frame(
    const math::RobotState<kDof>& state,
    const Eigen::Ref<const Eigen::Matrix<float, kDof, 1>>& last_action_train,
    std::int32_t inference_tick) {
    // --- proprio state terms ---
    Eigen::VectorXf ang_vel(3);  ang_vel = state.imu_angular_velocity;
    Eigen::VectorXf gravity(3);  gravity = math::gravity_body(state.imu_quat_wxyz);

    Eigen::Matrix<float, kDof, 1> qj_norm = state.joint_positions - default_angles_;
    Eigen::VectorXf qj_train(static_cast<int>(kDof));
    Eigen::VectorXf dqj_train(static_cast<int>(kDof));
    for (std::size_t i = 0; i < kDof; ++i) {
        qj_train[static_cast<int>(i)]  = qj_norm[remap_.deploy_to_train[i]];
        dqj_train[static_cast<int>(i)] = state.joint_velocities[remap_.deploy_to_train[i]];
    }
    Eigen::VectorXf last_action(static_cast<int>(kDof));
    for (int i = 0; i < static_cast<int>(kDof); ++i) last_action[i] = last_action_train[i];

    // --- time indices + anchor ---
    // time_indices_[k] = frame t + (max_history_ - 1 - k): furthest future first,
    // the current frame t last.
    std::int32_t t_now = 0;
    Eigen::VectorXf anchor_ori(6);         anchor_ori.setZero();
    if (motion_ != nullptr) {
        policy::get_time_indices(inference_tick, /*decimation=*/1,
                                 static_cast<double>(control_dt_),
                                 motion_->output_fps, static_cast<std::size_t>(max_history_),
                                 motion_->max_index(), time_indices_.data(), motion_->start_row);
        t_now = time_indices_[static_cast<std::size_t>(max_history_) - 1];

        // Measured against the odometry attitude. No IMU fallback: the control loop
        // never builds an observation without live odometry.
        const Eigen::Vector4f& orient_quat = state.odom_quaternion_wxyz;
        const math::Quat ref_quat{motion_->base_rots(t_now, 0), motion_->base_rots(t_now, 1),
                                  motion_->base_rots(t_now, 2), motion_->base_rots(t_now, 3)};
        const math::Quat q_rel = math::quat_difference(orient_quat, ref_quat);
        const math::Mat3 rot   = math::quat_to_rot(q_rel);
        anchor_ori << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    } else {
        std::fill(time_indices_.begin(), time_indices_.end(), 0);
    }

    frame_["base_ang_vel"]        = std::move(ang_vel);
    frame_["proj_gravity"]        = std::move(gravity);
    frame_["joint_pos_rel"]       = std::move(qj_train);
    frame_["joint_vel_rel"]       = std::move(dqj_train);
    frame_["last_action"]         = std::move(last_action);
    frame_["motion_anchor_ori_b"] = std::move(anchor_ori);

    // push buffered state terms (current value)
    for (auto& kv : buffers_) kv.second.push(frame_.at(kv.first));
}

void ObservationBuilder::reference_rows(const std::string& name, const std::int32_t* idx, int n,
                                        Eigen::MatrixXf& out) const {
    const int dim = reference_dim(name);
    out.resize(n, dim);
    if (motion_ == nullptr) { out.setZero(); return; }  // no reference: zero command
    const Eigen::Index last = motion_->dof_poss.rows() - 1;
    for (int j = 0; j < n; ++j) {
        Eigen::Index r = static_cast<Eigen::Index>(idx[j]);
        if (r < 0) r = 0;
        if (r > last) r = last;
        out.block(j, 0, 1, static_cast<int>(kDof)) = motion_->dof_poss.row(r);
        out.block(j, static_cast<int>(kDof), 1, static_cast<int>(kDof)) = motion_->dof_vels.row(r);
        const math::Quat q{motion_->base_rots(r, 0), motion_->base_rots(r, 1),
                           motion_->base_rots(r, 2), motion_->base_rots(r, 3)};
        const math::Vec3 lv_w{motion_->base_lin_vels(r, 0), motion_->base_lin_vels(r, 1),
                              motion_->base_lin_vels(r, 2)};
        const math::Vec3 av_w{motion_->base_ang_vels(r, 0), motion_->base_ang_vels(r, 1),
                              motion_->base_ang_vels(r, 2)};
        // tracking_command_partial: root gravity / velocities in the reference body frame.
        const int j2 = 2 * static_cast<int>(kDof);
        out.block(j, j2, 1, 3)     = math::gravity_body(q).transpose();
        out.block(j, j2 + 3, 1, 3) = math::quat_rotate_inverse(q, lv_w).transpose();
        out.block(j, j2 + 6, 1, 3) = math::quat_rotate_inverse(q, av_w).transpose();
    }
}

void ObservationBuilder::build(
    const math::RobotState<kDof>& state,
    const Eigen::Ref<const Eigen::Matrix<float, kDof, 1>>& last_action_train,
    std::int32_t inference_tick) {
    compute_frame(state, last_action_train, inference_tick);

    if (!allocated_) {
        outputs_.clear();
        outputs_.reserve(spec_.groups.size());
        for (const auto& g : spec_.groups) outputs_.push_back(GroupTensor{g.name, 0, 0, {}});
        allocated_ = true;
    }

    for (std::size_t gi = 0; gi < spec_.groups.size(); ++gi) {
        const GroupCfg& g = spec_.groups[gi];
        GroupTensor& out = outputs_[gi];

        if (g.is_image) {
            // ---- image group: raw depth frame -> clip([min,max]) -> log1p ----
            // (1, H*W) row-major; the ONNX declares (1,1,H,W)=H*W. depth_ null or
            // any invalid pixel -> far (clip_max) so the group is always valid.
            const int n = g.img_rows * g.img_cols;
            out.rows = 1;
            out.cols = n;
            out.data.resize(n);
            const float lo = g.clip_min, hi = g.clip_max;
            if (depth_ == nullptr) {
                out.data.setConstant(std::log1p(hi));  // far-plane fallback (blank)
            } else {
                const float* s = depth_->data.data();  // row-major meters, length kCount
                for (int i = 0; i < n; ++i) {
                    float v = s[i];
                    if (!std::isfinite(v) || v <= 1.0e-6F) v = hi;  // invalid/no-return -> far
                    v = v < lo ? lo : (v > hi ? hi : v);
                    out.data[i] = std::log1p(v);
                }
            }
        } else if (!g.is_stack) {
            // ---- flat group: concat each term's current-frame contribution ----
            std::vector<Eigen::VectorXf> parts;
            for (const auto& t : g.terms) {
                if (is_reference_term(t.name)) {
                    const int k = t.horizon > 0 ? t.horizon : 1;
                    Eigen::MatrixXf blk;
                    // The last k time indices: frames t+k-1 ... t, current frame last.
                    reference_rows(t.name, time_indices_.data() + (max_history_ - k), k, blk);
                    const auto* sf = reference_subfields(t.name);
                    // term_major: each subfield over all k frames, then the next subfield.
                    Eigen::VectorXf flat(blk.rows() * blk.cols());
                    Eigen::Index w = 0, off = 0;
                    for (const auto& s : *sf) {
                        for (int r = 0; r < blk.rows(); ++r)
                            for (int c = 0; c < s.dim; ++c) flat[w++] = blk(r, off + c);
                        off += s.dim;
                    }
                    parts.push_back(std::move(flat));
                } else {
                    parts.push_back(frame_value(t.name));
                }
            }
            int total = 0;
            for (const auto& p : parts) total += static_cast<int>(p.size());
            out.rows = 1; out.cols = total;
            out.data.resize(total);
            Eigen::Index w = 0;
            for (const auto& p : parts) { out.data.segment(w, p.size()) = p; w += p.size(); }
        } else {
            // ---- stack group: (history, sum of term dims) ----
            const int h = g.history;
            int F = 0;
            for (const auto& t : g.terms)
                F += is_reference_term(t.name) ? reference_dim(t.name) : state_term_dim(t.name);
            Eigen::MatrixXf feat(h, F);
            int col = 0;
            for (const auto& t : g.terms) {
                if (is_reference_term(t.name)) {
                    const int dim = reference_dim(t.name);
                    Eigen::MatrixXf blk;
                    reference_rows(t.name, time_indices_.data(), h, blk);
                    feat.block(0, col, h, dim) = blk;
                    col += dim;
                } else {
                    const int dim = state_term_dim(t.name);
                    Eigen::MatrixXf hist(buffers_.at(t.name).len(), buffers_.at(t.name).feat());
                    buffers_.at(t.name).get_history(hist);
                    feat.block(0, col, h, dim) = hist;
                    col += dim;
                }
            }
            out.rows = h; out.cols = F;
            out.data.resize(h * F);  // row-major flatten
            Eigen::Index w = 0;
            for (int r = 0; r < h; ++r)
                for (int c = 0; c < F; ++c) out.data[w++] = feat(r, c);
        }
    }
}

}  // namespace rt_deploy::g1
