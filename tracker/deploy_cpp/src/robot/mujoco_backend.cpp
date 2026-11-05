//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// MujocoBackend implementation, templated on joint count N.
//
// Uses the MuJoCo C API (mujoco.h) — mj_loadXML / mj_step / mjData access.
// Only compiled when RT_DEPLOY_WITH_MUJOCO=ON.
//
// When enable_viewer is set, a GLFW + mjr_render viewer runs on its own
// thread at ~60 Hz. The viewer owns a private mjData (so it can call
// mj_forward to compute derived rendering quantities) and reads qpos/qvel
// snapshots from the control thread under a brief mutex.
//
// The viewer also optionally renders a semi-transparent "ghost" robot that
// plays back the desired motion, for visually inspecting tracking error.
// Ghost rendering is gated on N == 29 because the only consumer
// (deploy_g1_mujoco) is G1-only and `policy::MotionData::dof_poss` is fixed
// at 29 columns.
//
// Mouse controls (matches MuJoCo `simulate` defaults):
//   Left-drag           — orbit camera
//   Right-drag          — pan camera
//   Scroll              — zoom
//   Ctrl + Left-drag    — apply rotational force (torque) to clicked body
//   Ctrl + Right-drag   — apply translational force to clicked body

#include "rt_deploy/robot/mujoco_backend.hpp"

#include "rt_deploy/core/clock.hpp"
#include "rt_deploy/core/rate_probe.hpp"
#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/g1/depth_frame.hpp"
#include "rt_deploy/policy/joint_remap.hpp"
#include "rt_deploy/policy/motion_loader.hpp"

#include <GLFW/glfw3.h>
#include <chrono>
#include <mujoco/mujoco.h>

// Headless ego-camera offscreen rendering uses EGL on Linux (GPU-backed, no
// display server — the same backend mujoco.Renderer selects via MUJOCO_GL=egl).
// macOS keeps the hidden-GLFW-window path (it isn't headless). See ego_gl_*().
#if defined(__linux__)
#    define RT_DEPLOY_EGL_EGO 1
#    include <EGL/egl.h>
#    include <EGL/eglext.h>   // EGL_PLATFORM_DEVICE_EXT, eglQueryDevicesEXT, …
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace rt_deploy::robot {

namespace {

// ─── GLFW input callbacks ────────────────────────────────────────────────────
// All shared mutable state lives in MujocoBackendImpl; the callbacks reach it
// via this context struct stashed in the GLFW window's user pointer.

struct ViewerCtx {
    // Read-only across threads
    const mjModel* model{nullptr};
    // Owned by viewer thread (this thread only)
    mjvCamera*  camera{nullptr};
    mjvScene*   scene{nullptr};
    mjvOption*  option{nullptr};        // for mjv_select (must not be NULL)
    mjData*     viewer_data{nullptr};   // for picking / movePerturb
    // Shared with control thread
    mjvPerturb* perturb{nullptr};
    std::mutex* perturb_mu{nullptr};
    // Mouse state (viewer thread only)
    bool   btn_left{false};
    bool   btn_right{false};
    bool   ctrl_held{false};
    double last_x{0.0};
    double last_y{0.0};
};

void on_mouse_button(GLFWwindow* w, int button, int action, int mods) {
    auto* ctx = static_cast<ViewerCtx*>(glfwGetWindowUserPointer(w));
    if (ctx == nullptr) return;

    const bool press = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_LEFT)  ctx->btn_left  = press;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) ctx->btn_right = press;
    ctx->ctrl_held = (mods & GLFW_MOD_CONTROL) != 0;

    glfwGetCursorPos(w, &ctx->last_x, &ctx->last_y);

    // CTRL + click → start a perturbation by selecting the body under the cursor.
    if (press && ctx->ctrl_held && (button == GLFW_MOUSE_BUTTON_LEFT
                                     || button == GLFW_MOUSE_BUTTON_RIGHT)) {
        int width  = 0;
        int height = 0;
        glfwGetWindowSize(w, &width, &height);
        if (width <= 0 || height <= 0) return;
        const mjtNum aspect = static_cast<mjtNum>(width) / static_cast<mjtNum>(height);
        const mjtNum relx   = ctx->last_x / static_cast<mjtNum>(width);
        const mjtNum rely   = 1.0 - ctx->last_y / static_cast<mjtNum>(height);
        mjtNum selpnt[3]    = {0, 0, 0};
        int    geomid[1]    = {-1};
        int    flexid[1]    = {-1};
        int    skinid[1]    = {-1};

        const std::scoped_lock lock{*ctx->perturb_mu};
        const int selbody = mjv_select(ctx->model, ctx->viewer_data, ctx->option,
                                         aspect, relx, rely, ctx->scene, selpnt,
                                         geomid, flexid, skinid);
        if (selbody > 0) {
            ctx->perturb->select     = selbody;
            ctx->perturb->skinselect = skinid[0];
            ctx->perturb->flexselect = flexid[0];

            // localpos = body-frame coordinates of the click point.
            mjtNum tmp[3];
            mju_sub3(tmp, selpnt, ctx->viewer_data->xpos + 3 * selbody);
            mju_mulMatTVec(ctx->perturb->localpos, ctx->viewer_data->xmat + 9 * selbody,
                            tmp, 3, 3);

            ctx->perturb->active = (button == GLFW_MOUSE_BUTTON_LEFT)
                                       ? mjPERT_ROTATE
                                       : mjPERT_TRANSLATE;
            mjv_initPerturb(ctx->model, ctx->viewer_data, ctx->scene, ctx->perturb);
        }
    }

    // Release of either mouse button while perturbing → end perturbation.
    if (!press && (ctx->perturb != nullptr)) {
        const std::scoped_lock lock{*ctx->perturb_mu};
        if (!ctx->btn_left && !ctx->btn_right) {
            ctx->perturb->active = 0;
            ctx->perturb->select = 0;
        }
    }
}

void on_cursor_pos(GLFWwindow* w, double x, double y) {
    auto* ctx = static_cast<ViewerCtx*>(glfwGetWindowUserPointer(w));
    if (ctx == nullptr) return;

    int width  = 0;
    int height = 0;
    glfwGetWindowSize(w, &width, &height);
    if (height <= 0) {
        ctx->last_x = x;
        ctx->last_y = y;
        return;
    }
    const double dx = (x - ctx->last_x) / static_cast<double>(height);
    const double dy = (y - ctx->last_y) / static_cast<double>(height);
    ctx->last_x     = x;
    ctx->last_y     = y;

    if (!ctx->btn_left && !ctx->btn_right) return;

    // CTRL held → moving the perturbation; otherwise → moving the camera.
    if (ctx->ctrl_held && ctx->perturb != nullptr && ctx->perturb->active != 0) {
        const mjtMouse action = ctx->btn_left ? mjMOUSE_ROTATE_V : mjMOUSE_MOVE_V;
        const std::scoped_lock lock{*ctx->perturb_mu};
        mjv_movePerturb(ctx->model, ctx->viewer_data, action, dx, dy,
                         ctx->scene, ctx->perturb);
    } else {
        const mjtMouse action = ctx->btn_left ? mjMOUSE_ROTATE_V : mjMOUSE_MOVE_V;
        mjv_moveCamera(ctx->model, action, dx, dy, ctx->scene, ctx->camera);
    }
}

void on_scroll(GLFWwindow* w, double /*xoff*/, double yoff) {
    auto* ctx = static_cast<ViewerCtx*>(glfwGetWindowUserPointer(w));
    if (ctx == nullptr) return;
    mjv_moveCamera(ctx->model, mjMOUSE_ZOOM, 0.0, -0.05 * yoff, ctx->scene, ctx->camera);
}

void on_key(GLFWwindow* w, int key, int /*scancode*/, int action, int /*mods*/) {
    auto* ctx = static_cast<ViewerCtx*>(glfwGetWindowUserPointer(w));
    if (ctx == nullptr) return;
    if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
        ctx->ctrl_held = (action != GLFW_RELEASE);
    }
}

// ─── Backend implementation ──────────────────────────────────────────────────
template <std::size_t N>
class MujocoBackendImpl final : public IRobotBackend<N> {
public:
    using JointVec = Eigen::Matrix<float, static_cast<Eigen::Index>(N), 1>;

    MujocoBackendImpl(mjModel* m, mjData* d, int substeps, bool enable_viewer,
                       bool spawn_viewer_thread,
                       int viewer_w, int viewer_h,
                       const std::optional<JointVec>& max_torque,
                       const std::optional<JointVec>& kp_scale,
                       const std::optional<JointVec>& kd_scale,
                       const policy::MotionData*     ghost_motion,
                       const policy::JointRemap<29>* ghost_remap,
                       const std::array<float, 4>&   ghost_rgba) noexcept
        : m_{m}, d_{d}, substeps_{substeps},
          viewer_enabled_{enable_viewer},
          viewer_w_{viewer_w}, viewer_h_{viewer_h},
          ghost_motion_{ghost_motion},
          ghost_remap_{ghost_remap},
          ghost_rgba_{ghost_rgba} {
        if (max_torque) {
            max_torque_  = *max_torque;
            clip_torque_ = true;
        }
        if (kp_scale) kp_scale_ = *kp_scale;
        if (kd_scale) kd_scale_ = *kd_scale;
        mjv_defaultPerturb(&perturb_);
        mj_forward(m_, d_);
        if (viewer_enabled_) {
            snapshot_qpos_.assign(static_cast<std::size_t>(m_->nq), 0.0);
            snapshot_qvel_.assign(static_cast<std::size_t>(m_->nv), 0.0);
            // Seed snapshot so the viewer renders a valid pose immediately.
            std::memcpy(snapshot_qpos_.data(), d_->qpos,
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nq));
            std::memcpy(snapshot_qvel_.data(), d_->qvel,
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nv));
            if (spawn_viewer_thread) {
                viewer_thread_ = std::thread{[this]() { viewer_loop(); }};
            }
        }
    }

    ~MujocoBackendImpl() override {
        viewer_should_stop_.store(true, std::memory_order_release);
        if (viewer_thread_.joinable()) {
            viewer_thread_.join();
        }
        // Free ego-renderer GL resources while the context is current
        // (mjr_freeContext touches GL state), then tear the context down.
        if (ego_initialized_) {
            ego_gl_make_current();
            if (ego_scene_   != nullptr) { mjv_freeScene(ego_scene_);   delete ego_scene_;   ego_scene_   = nullptr; }
            if (ego_context_ != nullptr) { mjr_freeContext(ego_context_); delete ego_context_; ego_context_ = nullptr; }
            if (ego_camera_  != nullptr) { delete ego_camera_;  ego_camera_  = nullptr; }
            if (ego_option_  != nullptr) { delete ego_option_;  ego_option_  = nullptr; }
            if (ego_data_    != nullptr) { mj_deleteData(ego_data_); ego_data_ = nullptr; }
            ego_gl_release();
            ego_gl_destroy();
            ego_initialized_ = false;
        }
        // Single global GLFW teardown, AFTER both the viewer (joined above) and
        // the ego renderer have released their windows/contexts. viewer_loop()
        // must NOT call glfwTerminate() itself: GLFW state is process-global, so
        // terminating from the viewer orphans the still-live ego renderer and
        // its later teardown dereferences freed GLFW state → crash on close.
        // Safe to call even if GLFW was never initialized (documented no-op).
        glfwTerminate();
        if (d_ != nullptr) {
            mj_deleteData(d_);
        }
        if (m_ != nullptr) {
            mj_deleteModel(m_);
        }
    }

    [[nodiscard]] ReadStatus read_state(math::RobotState<N>& out) noexcept override {
        for (std::size_t i = 0; i < N; ++i) {
            out.joint_positions[static_cast<Eigen::Index>(i)]  =
                static_cast<float>(d_->qpos[7 + i]);
            out.joint_velocities[static_cast<Eigen::Index>(i)] =
                static_cast<float>(d_->qvel[6 + i]);
        }
        out.imu_quat_wxyz[0] = static_cast<float>(d_->qpos[3]);
        out.imu_quat_wxyz[1] = static_cast<float>(d_->qpos[4]);
        out.imu_quat_wxyz[2] = static_cast<float>(d_->qpos[5]);
        out.imu_quat_wxyz[3] = static_cast<float>(d_->qpos[6]);
        for (int i = 0; i < 3; ++i) {
            out.imu_angular_velocity[i]    = static_cast<float>(d_->qvel[3 + i]);
            out.imu_linear_acceleration[i] = 0.0F;
        }
        // MuJoCo world IS the odom world. Expose qpos[0..2] so consumers
        // (state outbox, transport state replies, plan anchoring) can read the
        // robot's actual base position. Without this they fall back to
        // standing_base_pos and the anchor never tracks the simulated robot
        // as it walks.
        out.odom_position[0] = static_cast<float>(d_->qpos[0]);
        out.odom_position[1] = static_cast<float>(d_->qpos[1]);
        out.odom_position[2] = static_cast<float>(d_->qpos[2]);
        // MuJoCo world == odom world, so qpos[3..7] (base quat, wxyz) is also
        // the odom orientation. Expose it as part of the robot state (odom is
        // still reported even though the tracker now does yaw-only alignment;
        // on hardware imu_quat and odom_quat differ by a fixed yaw, so any
        // consumer doing SE(2) work MUST read odom, not imu).
        out.odom_quaternion_wxyz[0] = static_cast<float>(d_->qpos[3]);
        out.odom_quaternion_wxyz[1] = static_cast<float>(d_->qpos[4]);
        out.odom_quaternion_wxyz[2] = static_cast<float>(d_->qpos[5]);
        out.odom_quaternion_wxyz[3] = static_cast<float>(d_->qpos[6]);
        out.odom_valid       = true;
        out.timestamp_ns = core::MonotonicClock::now().count();
        out.odom_timestamp_ns = out.timestamp_ns;
        return ReadStatus::kOk;
    }

    void send_command(const math::MotorCommand<N>& cmd) noexcept override {
        target_pos_ = cmd.positions;
        target_vel_ = cmd.velocities;
        kp_         = cmd.kp.cwiseProduct(kp_scale_);  // the plant's identified drive (1 unless configured)
        kd_         = cmd.kd.cwiseProduct(kd_scale_);
        ff_torque_  = cmd.torques;
    }

    [[nodiscard]] bool poll_remote(RemoteState& out) noexcept override {
        out = {};
        return false;
    }

    // ── Ego-camera offscreen render ──────────────────────────────────────────
    // Create a headless GL context (EGL on Linux, hidden GLFW window on macOS) +
    // mjrContext + a private mjData bound to the named MJCF camera
    // (mjCAMERA_FIXED). Call once after construction, on the main thread on
    // macOS (GLFW window-creation constraint). Idempotent.
    // True when the camera name denotes the OAK mount (case-insensitive
    // substring "oak"), used to auto-enable the fisheye path for `oak1`.
    static bool camera_name_is_oak(const std::string& name) {
        std::string lower;
        lower.reserve(name.size());
        for (char c : name) {
            lower.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(c))));
        }
        return lower.find("oak") != std::string::npos;
    }

    [[nodiscard]] bool configure_oak_fisheye(int render_width,
                                              double target_hfov_deg) {
        oak_enabled_ = false;
        oak_warp_    = false;
        if (render_width <= 0 || target_hfov_deg <= 0.0 || target_hfov_deg >= 180.0) {
            std::fprintf(stderr,
                "[oak_fisheye] bad args render_width=%d hfov=%.1f\n",
                render_width, target_hfov_deg);
            return false;
        }

        // ── Baked OAK-1 calibrated fisheye intrinsics ────────────────────────
        // OAK-1 calibration.
        // Kannala-Brandt 4-coeff (cv2.fisheye) model at 1280x800.
        constexpr double kW  = 1280.0, kH = 800.0;
        constexpr double kFx = 566.49554064, kFy = 566.78558951;
        constexpr double kCx = 642.80704971, kCy = 410.84365973;
        constexpr double kD0 = -0.01465799, kD1 = 0.0642358,
                         kD2 = -0.07535982, kD3 = 0.0304101;
        constexpr double kSensorWmm = 3.84;  // OV9782 sensor width (3.0µm × 1280)

        // ── Wide-pinhole render config ──
        // Widen the source pinhole to `target_hfov_deg` so the fisheye warp fills
        // the output with no black corners; keep the calibrated principal-point
        // offset. Vertical follows the fisheye aspect (square pixels).
        const int render_w = render_width;
        const int render_h = static_cast<int>(std::lround(render_width * kH / kW));
        const double f_mm    = (kSensorWmm / 2.0) /
                               std::tan(target_hfov_deg * M_PI / 180.0 / 2.0);
        const double px_size = kSensorWmm / render_w;
        const double fx_render = f_mm / px_size;
        const double fy_render = fx_render;  // square pixels
        // Principal offset in mm uses the FISHEYE width (kW).
        const double cx_off_mm = (kCx - kW / 2.0) * (kSensorWmm / kW);
        const double cy_off_mm = (kCy - kH / 2.0) * (kSensorWmm / kW);
        const double cx_render = render_w / 2.0 + cx_off_mm / px_size;
        const double cy_render = render_h / 2.0 + cy_off_mm / px_size;

        oak_src_w_     = render_w;
        oak_src_h_     = render_h;
        oak_fx_render_ = fx_render;
        oak_fy_render_ = fy_render;
        oak_cx_render_ = cx_render;
        oak_cy_render_ = cy_render;

        // ── Build the pinhole→fisheye remap ──
        // Output is the OAK native 1280x800. For each fisheye output pixel,
        // undistort via the calibrated K + K-B D (cv2.fisheye.undistortPoints
        // equivalent) to get the ideal-pinhole ray, then project onto the wide
        // pinhole source K. map_x/map_y index into the rendered source image.
        oak_out_w_ = static_cast<int>(kW);
        oak_out_h_ = static_cast<int>(kH);
        const auto n = static_cast<std::size_t>(oak_out_w_) *
                       static_cast<std::size_t>(oak_out_h_);
        oak_map_x_.resize(n);
        oak_map_y_.resize(n);
        for (int oy = 0; oy < oak_out_h_; ++oy) {
            for (int ox = 0; ox < oak_out_w_; ++ox) {
                const double xd = (ox - kCx) / kFx;   // distorted normalized
                const double yd = (oy - kCy) / kFy;
                const double theta_d = std::sqrt(xd * xd + yd * yd);
                // Invert theta_d = theta·(1 + D0·θ² + D1·θ⁴ + D2·θ⁶ + D3·θ⁸)
                // via Newton (cv2.fisheye uses 10 iterations).
                double theta = theta_d;
                for (int it = 0; it < 10; ++it) {
                    const double t2 = theta * theta, t4 = t2 * t2,
                                 t6 = t4 * t2, t8 = t4 * t4;
                    const double poly = 1.0 + kD0 * t2 + kD1 * t4 + kD2 * t6 + kD3 * t8;
                    const double f  = theta * poly - theta_d;
                    const double fp = 1.0 + 3.0 * kD0 * t2 + 5.0 * kD1 * t4
                                          + 7.0 * kD2 * t6 + 9.0 * kD3 * t8;
                    theta -= f / fp;
                }
                const double scale = (theta_d > 1e-9)
                                   ? std::tan(theta) / theta_d : 1.0;
                const double xn = xd * scale;  // ideal-pinhole normalized
                const double yn = yd * scale;
                const std::size_t oi =
                    static_cast<std::size_t>(oy) * static_cast<std::size_t>(oak_out_w_)
                    + static_cast<std::size_t>(ox);
                oak_map_x_[oi] = static_cast<float>(xn * fx_render + cx_render);
                oak_map_y_[oi] = static_cast<float>(yn * fy_render + cy_render);
            }
        }
        oak_rgb_out_.assign(n * 3, 0);
        oak_enabled_ = true;
        oak_warp_    = true;
        std::fprintf(stderr,
            "[oak_fisheye] baked intrinsics: wide-pinhole src %dx%d "
            "(hfov=%.1f°, f=%.4fmm, principal=%.1f,%.1f) -> fisheye %dx%d\n",
            oak_src_w_, oak_src_h_, target_hfov_deg, f_mm,
            cx_render, cy_render, oak_out_w_, oak_out_h_);
        return true;
    }

    [[nodiscard]] bool init_ego_renderer(int width, int height,
                                          const std::string& camera_name) override {
        if (ego_initialized_) return true;
        if (width <= 0 || height <= 0) {
            std::fprintf(stderr, "[ego_renderer] invalid size %dx%d\n", width, height);
            return false;
        }
        ego_cam_id_ = mj_name2id(m_, mjOBJ_CAMERA, camera_name.c_str());
        if (ego_cam_id_ < 0) {
            std::fprintf(stderr,
                "[ego_renderer] camera '%s' not found in MJCF; offscreen render disabled\n",
                camera_name.c_str());
            return false;
        }
        // Auto-enable the OAK fisheye when the bound camera is the OAK mount:
        // `--ego_camera_name oak1` then uses BOTH the XML mount/pose (via the
        // fixed camera, below) AND the baked calibrated intrinsics + K-B fisheye
        // warp (configure_oak_fisheye) — no separate flag needed. Any other
        // camera (e.g. the D435 `ego_camera`) renders a plain pinhole.
        if (!oak_enabled_ && camera_name_is_oak(camera_name)) {
            // 1600px wide-pinhole source, 140° HFOV.
            if (configure_oak_fisheye(1600, 140.0)) {
                std::fprintf(stderr,
                    "[oak_fisheye] auto-enabled for camera '%s'\n", camera_name.c_str());
            } else {
                std::fprintf(stderr,
                    "[oak_fisheye] auto-config failed; rendering '%s' as plain pinhole\n",
                    camera_name.c_str());
            }
        }
        // OAK path: render the wide pinhole at the baked source resolution and
        // install its intrinsics on the bound MJCF camera (mount/pose is left
        // untouched — it comes from the `oak1` camera in the XML). MuJoCo uses
        // the intrinsic model (off-center principal point, independent fx/fy)
        // whenever cam_sensorsize > 0, so this preserves the calibrated
        // principal-point offset that a plain fovy would drop.
        if (oak_enabled_) {
            width  = oak_src_w_;
            height = oak_src_h_;
            const double sensor_w = 3.84;  // OV9782 width (matches configure_oak_fisheye)
            const double sensor_h = sensor_w * oak_src_h_ / oak_src_w_;
            const int id = ego_cam_id_;
            m_->cam_sensorsize[2 * id]     = static_cast<float>(sensor_w);
            m_->cam_sensorsize[2 * id + 1] = static_cast<float>(sensor_h);
            m_->cam_resolution[2 * id]     = oak_src_w_;
            m_->cam_resolution[2 * id + 1] = oak_src_h_;
            // cam_intrinsic = (focal_x, focal_y, principal_x, principal_y) in
            // LENGTH units: length = pixels · sensor / resolution.
            m_->cam_intrinsic[4 * id + 0] =
                static_cast<float>(oak_fx_render_ * sensor_w / oak_src_w_);
            m_->cam_intrinsic[4 * id + 1] =
                static_cast<float>(oak_fy_render_ * sensor_h / oak_src_h_);
            m_->cam_intrinsic[4 * id + 2] = static_cast<float>(
                (oak_cx_render_ - oak_src_w_ / 2.0) * sensor_w / oak_src_w_);
            // NOTE the Y NEGATION: our intrinsics (oak_cy_render_) are in the
            // cv2 image convention (Y down, principal BELOW centre → +10.844,
            // matching oak_intrinsics.json / the warp's K). MuJoCo's sensor Y is
            // UP (this is exactly why the XML oak1 camera lists principalpixel
            // Y as -10.844 while X stays +2.807), and render_ego_image flips the
            // frame back to top-down before the warp — so the MuJoCo principal-Y
            // must carry the opposite sign for the rendered source principal to
            // land where the warp samples it.
            m_->cam_intrinsic[4 * id + 3] = static_cast<float>(
                -(oak_cy_render_ - oak_src_h_ / 2.0) * sensor_h / oak_src_h_);
            m_->cam_fovy[id] = 0.0;  // intrinsic model takes over
        }
        if (!ego_gl_create(width, height)) {
            return false;  // ego_gl_create logged the specific failure
        }

        ego_data_ = mj_makeData(m_);
        if (ego_data_ == nullptr) {
            std::fprintf(stderr, "[ego_renderer] mj_makeData failed; offscreen disabled\n");
            ego_gl_release();
            ego_gl_destroy();
            return false;
        }
#if defined(RT_DEPLOY_EGL_EGO)
        // EGL is surfaceless (no default framebuffer), so we render into the
        // mjrContext's OFFSCREEN buffer. Size the model's offscreen target to
        // our frame BEFORE mjr_makeContext allocates it (defaults are 640x480).
        if (m_->vis.global.offwidth  < width)  m_->vis.global.offwidth  = width;
        if (m_->vis.global.offheight < height) m_->vis.global.offheight = height;
#endif
        ego_scene_   = new mjvScene{};
        ego_camera_  = new mjvCamera{};
        ego_option_  = new mjvOption{};
        ego_context_ = new mjrContext{};
        mjv_defaultScene(ego_scene_);
        mjv_defaultCamera(ego_camera_);
        mjv_defaultOption(ego_option_);
        mjr_defaultContext(ego_context_);
        mjv_makeScene(m_, ego_scene_, /*maxgeom=*/4000);
        mjr_makeContext(m_, ego_context_, mjFONTSCALE_100);
#if defined(RT_DEPLOY_EGL_EGO)
        mjr_setBuffer(mjFB_OFFSCREEN, ego_context_);  // EGL has no window buffer
#endif
        ego_camera_->type       = mjCAMERA_FIXED;
        ego_camera_->fixedcamid = ego_cam_id_;

        ego_w_ = width;
        ego_h_ = height;
        ego_rgb_buffer_.assign(static_cast<std::size_t>(width * height * 3), 0);
        ego_rgb_flipped_.assign(static_cast<std::size_t>(width * height * 3), 0);

        // Ensure the qpos/qvel snapshot exists even when the viewer is off
        // (e.g. the headless system/cpp sim node): seed it once here so the first
        // render has a valid pose, and step() keeps it fresh (see step()).
        if (snapshot_qpos_.empty()) {
            const std::scoped_lock lock{snapshot_mu_};
            snapshot_qpos_.assign(static_cast<std::size_t>(m_->nq), 0.0);
            snapshot_qvel_.assign(static_cast<std::size_t>(m_->nv), 0.0);
            std::memcpy(snapshot_qpos_.data(), d_->qpos,
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nq));
            std::memcpy(snapshot_qvel_.data(), d_->qvel,
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nv));
            snapshot_time_ = d_->time;
        }

        // Release the context so the render thread can claim it on first call.
        ego_gl_release();
        ego_initialized_ = true;
        std::fprintf(stderr,
            "[ego_renderer] initialized (%s): camera='%s' (id=%d) %dx%d\n",
            ego_gl_backend_name(), camera_name.c_str(), ego_cam_id_, width, height);
        return true;
    }

    // Render the bound camera from the latest qpos snapshot into top-down RGB.
    // Thread-safe (serialized by ego_render_mu_); reads snapshot under
    // snapshot_mu_ so it never races the control thread's mj_step writes.
    [[nodiscard]] bool render_ego_image(std::vector<std::uint8_t>& rgb_out,
                                         int& w_out, int& h_out) noexcept override {
        if (!ego_initialized_) return false;
        const std::scoped_lock render_lock{ego_render_mu_};
        ego_gl_make_current();
        {
            const std::scoped_lock data_lock{snapshot_mu_};
            std::memcpy(ego_data_->qpos, snapshot_qpos_.data(),
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nq));
            std::memcpy(ego_data_->qvel, snapshot_qvel_.data(),
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nv));
            ego_data_->time = snapshot_time_;
        }
        mj_forward(m_, ego_data_);

        mjrRect viewport{0, 0, ego_w_, ego_h_};
        mjv_updateScene(m_, ego_data_, ego_option_, /*pert=*/nullptr,
                         ego_camera_, mjCAT_ALL, ego_scene_);
        mjr_render(viewport, ego_scene_, ego_context_);
        mjr_readPixels(ego_rgb_buffer_.data(), /*depth=*/nullptr, viewport, ego_context_);
        ego_gl_release();   // GL work done; the rest is CPU. See render_depth_frame_ (EGL sharing).

        // OpenGL returns rows bottom-up; flip to top-down (mujoco.Renderer
        // convention) so the bytes match the Python sim's ego frames.
        const int row_bytes = ego_w_ * 3;
        for (int row = 0; row < ego_h_; ++row) {
            const int src_off = (ego_h_ - 1 - row) * row_bytes;
            const int dst_off = row * row_bytes;
            std::memcpy(ego_rgb_flipped_.data() + dst_off,
                         ego_rgb_buffer_.data()  + src_off,
                         static_cast<std::size_t>(row_bytes));
        }
        rgb_out = ego_rgb_flipped_;
        w_out   = ego_w_;
        h_out   = ego_h_;

        // OAK fisheye: bilinear-remap the wide pinhole into the fisheye output.
        if (oak_warp_) {
            const int sw = ego_w_, sh = ego_h_;
            for (int oy = 0; oy < oak_out_h_; ++oy) {
                for (int ox = 0; ox < oak_out_w_; ++ox) {
                    const std::size_t oi =
                        static_cast<std::size_t>(oy) * static_cast<std::size_t>(oak_out_w_)
                        + static_cast<std::size_t>(ox);
                    const float fx = oak_map_x_[oi];
                    const float fy = oak_map_y_[oi];
                    std::uint8_t* dst = &oak_rgb_out_[oi * 3];
                    if (fx < 0.0F || fy < 0.0F ||
                        fx > static_cast<float>(sw - 1) ||
                        fy > static_cast<float>(sh - 1)) {
                        dst[0] = dst[1] = dst[2] = 0;  // black outside the frustum
                        continue;
                    }
                    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
                    const int x1 = std::min(x0 + 1, sw - 1);
                    const int y1 = std::min(y0 + 1, sh - 1);
                    const float ax = fx - static_cast<float>(x0);
                    const float ay = fy - static_cast<float>(y0);
                    const auto* src = ego_rgb_flipped_.data();
                    for (int c = 0; c < 3; ++c) {
                        const float v00 = src[(y0 * sw + x0) * 3 + c];
                        const float v01 = src[(y0 * sw + x1) * 3 + c];
                        const float v10 = src[(y1 * sw + x0) * 3 + c];
                        const float v11 = src[(y1 * sw + x1) * 3 + c];
                        const float top = v00 + ax * (v01 - v00);
                        const float bot = v10 + ax * (v11 - v10);
                        dst[c] = static_cast<std::uint8_t>(top + ay * (bot - top) + 0.5F);
                    }
                }
            }
            rgb_out = oak_rgb_out_;
            w_out   = oak_out_w_;
            h_out   = oak_out_h_;
        }
        return true;
    }

    // ── Depth-camera offscreen render (depth student, sim) ───────────────────
    // Set up the depth camera render. REUSES the ego offscreen context, so it
    // must be called AFTER a successful init_ego_renderer (same thread). Returns
    // false (no inbox: an image-group config is then refused at start) if the
    // ego context is absent, the model has no such camera, or the size is not
    // the fixed DepthFrame size.
    [[nodiscard]] bool init_depth_renderer(const std::string& camera_name,
                                           int width, int height,
                                           float clip_min, float clip_max,
                                           int obs_decim) override {
        if (!ego_initialized_) {
            std::fprintf(stderr, "[depth_renderer] needs the ego offscreen context "
                                 "(init_ego_renderer first); depth disabled.\n");
            return false;
        }
        // render_depth_frame_ writes a fixed-size DepthFrame slot, and the
        // viewport is drawn inside the ego framebuffer: reject anything else
        // rather than overrun either.
        if (width != g1::DepthFrame::kCols || height != g1::DepthFrame::kRows) {
            std::fprintf(stderr, "[depth_renderer] size %dx%d unsupported (DepthFrame is %dx%d); "
                                 "depth disabled.\n", width, height,
                         g1::DepthFrame::kCols, g1::DepthFrame::kRows);
            return false;
        }
        if (width > ego_w_ || height > ego_h_) {
            std::fprintf(stderr, "[depth_renderer] %dx%d does not fit the %dx%d ego framebuffer; "
                                 "depth disabled.\n", width, height, ego_w_, ego_h_);
            return false;
        }
        depth_cam_id_ = mj_name2id(m_, mjOBJ_CAMERA, camera_name.c_str());
        if (depth_cam_id_ < 0) {
            std::fprintf(stderr, "[depth_renderer] camera '%s' not in model; depth disabled.\n",
                         camera_name.c_str());
            return false;
        }
        depth_w_ = width;
        depth_h_ = height;
        depth_clip_min_ = clip_min;
        depth_clip_max_ = clip_max;
        // Hold each frame for obs_decim control steps by rendering only every
        // obs_decim-th step(); the inbox keeps serving the last published frame in
        // between, which IS the hold. Clamped rather than rejected: the config
        // parser has already range-checked it, and a backend is not the place to
        // fail a run over an observation-spec value.
        depth_decim_ = (obs_decim > 1) ? obs_decim : 1;
        // The first frame is rendered below, before the loop's first tick; the next
        // one depth_decim_ step()s later, so every frame is held for depth_decim_ ticks.
        depth_step_  = 0;
        // MuJoCo's readPixels depth is the OpenGL z-buffer (nonlinear [0,1]);
        // linearize to eye-space meters with the model's near/far (scaled by the
        // scene extent), matching mujoco.Renderer.enable_depth_rendering().
        const double extent = static_cast<double>(m_->stat.extent);
        depth_near_ = static_cast<double>(m_->vis.map.znear) * extent;
        depth_far_  = static_cast<double>(m_->vis.map.zfar)  * extent;
        mjv_defaultCamera(&depth_camera_);
        depth_camera_.type       = mjCAMERA_FIXED;
        depth_camera_.fixedcamid = depth_cam_id_;
        depth_gl_buf_.assign(g1::DepthFrame::kCount, 0.0F);
        depth_active_ = true;
        std::fprintf(stderr, "[depth_renderer] camera='%s' id=%d %dx%d clip[%.2f,%.2f]m "
                             "near=%.3f far=%.3f decim=%d\n", camera_name.c_str(), depth_cam_id_,
                     depth_w_, depth_h_, static_cast<double>(depth_clip_min_),
                     static_cast<double>(depth_clip_max_), depth_near_, depth_far_, depth_decim_);
        // Render the current pose now, so the observation of tick 0 (built before
        // the first step()) already has a real frame instead of the far plane, as
        // in the Python sim, which renders before its first observation.
        {
            const std::scoped_lock lock{snapshot_mu_};
            std::memcpy(snapshot_qpos_.data(), d_->qpos,
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nq));
            std::memcpy(snapshot_qvel_.data(), d_->qvel,
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nv));
            snapshot_time_ = d_->time;
        }
        render_depth_frame_();
        return true;
    }

    [[nodiscard]] core::TripleBuffer<g1::DepthFrame>* depth_inbox() noexcept override {
        return depth_active_ ? &depth_tb_ : nullptr;
    }

    void step() noexcept override {
        for (int s = 0; s < substeps_; ++s) {
            // PD law parity with Python pd_control + send_command.
            for (std::size_t i = 0; i < N; ++i) {
                const auto idx = static_cast<Eigen::Index>(i);
                const float q   = static_cast<float>(d_->qpos[7 + i]);
                const float dq  = static_cast<float>(d_->qvel[6 + i]);
                float       tau = kp_[idx] * (target_pos_[idx] - q)
                                + kd_[idx] * (target_vel_[idx] - dq)
                                + ff_torque_[idx];
                if (clip_torque_) {
                    const float lim = max_torque_[idx];
                    if (tau >  lim) tau =  lim;
                    if (tau < -lim) tau = -lim;
                }
                d_->ctrl[i] = static_cast<mjtNum>(tau);
            }

            // Apply mouse-driven perturbation (zeroes xfrc each step, so the
            // force is impulsive and stops the moment the user releases).
            if (viewer_enabled_) {
                const std::scoped_lock lock{perturb_mu_};
                mju_zero(d_->xfrc_applied, static_cast<int>(6 * m_->nbody));
                if (perturb_.active != 0 && perturb_.select > 0) {
                    mjv_applyPerturbForce(m_, d_, &perturb_);
                }
            }

            mj_step(m_, d_);
        }
        poll_scene_files_();
        // Publish the qpos/qvel snapshot for the viewer AND the ego renderer.
        // Needed even with the viewer off: a headless run can still render the
        // ego camera from this snapshot.
        if (viewer_enabled_ || ego_initialized_) {
            const std::scoped_lock lock{snapshot_mu_};
            std::memcpy(snapshot_qpos_.data(), d_->qpos,
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nq));
            std::memcpy(snapshot_qvel_.data(), d_->qvel,
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nv));
            snapshot_time_ = d_->time;
        }
        // Depth render at the control rate / obs_decim (reuses the ego offscreen
        // context; publishes to depth_tb_, drained by the control loop's
        // depth_inbox). depth_active_ implies ego_initialized_ (init_depth_renderer
        // requires it), so the snapshot above was just refreshed.
        if (depth_active_) {
            if (++depth_step_ >= depth_decim_) {
                depth_step_ = 0;
                render_depth_frame_();
            }
        }
    }

    void cleanup() noexcept override {
        if (d_ != nullptr) {
            std::memset(d_->ctrl, 0, static_cast<std::size_t>(m_->nu) * sizeof(mjtNum));
        }
    }

    [[nodiscard]] bool wait_for_state(std::int64_t /*timeout_ns*/) override { return true; }

    // Run the viewer on the calling thread (required on macOS). No-op if the
    // viewer was either disabled or already spawned on its own thread.
    void run_viewer_blocking() noexcept override {
        if (viewer_enabled_ && !viewer_thread_.joinable()) {
            viewer_loop();
        }
    }

    void request_viewer_stop() noexcept override {
        viewer_should_stop_.store(true, std::memory_order_release);
    }

    // Model camera lookup backing mujoco_camera_info(). m_ is immutable after
    // construction, so no lock is needed.
    [[nodiscard]] bool camera_info(const std::string& camera_name, MujocoCameraInfo& out) const {
        const int id = mj_name2id(m_, mjOBJ_CAMERA, camera_name.c_str());
        if (id < 0) return false;
        const char* parent = mj_id2name(m_, mjOBJ_BODY, m_->cam_bodyid[id]);
        out.parent_body = (parent != nullptr) ? parent : "";
        for (std::size_t i = 0; i < 3; ++i) out.pos[i] = m_->cam_pos[3 * id + static_cast<int>(i)];
        for (std::size_t i = 0; i < 4; ++i) out.quat_wxyz[i] = m_->cam_quat[4 * id + static_cast<int>(i)];
        out.fovy = m_->cam_fovy[id];
        return true;
    }
    // Backing mujoco_offscreen_samples(): the MSAA sample count the offscreen
    // (ego / depth) buffer is created with.
    [[nodiscard]] int offscreen_samples() const noexcept { return m_->vis.quality.offsamples; }

private:
    // Scene objects driven through files the system side writes and reads (sim only; each one
    // is off unless its HALO_*_FILE env var is set). Files, not a comm channel: RobotState has
    // no field for object poses. Runs on the physics thread, which owns m_/d_.
    void poll_scene_files_() noexcept {
        // LIVE BODY MOVE (HALO_MOVE_BODY + HALO_MOVE_FILE): one line "x y yaw_deg", an absolute
        // pose, polled at ~10 Hz. The scripted operator's `lead` step writes it to walk the human.
        // Writes body_pos/body_quat of a static body and refreshes with mj_kinematics only.
        if (move_file_bid_ == -2) {                     // resolve once, lazily
            move_file_bid_ = -1;
            const char* mb = std::getenv("HALO_MOVE_BODY");
            const char* mf = std::getenv("HALO_MOVE_FILE");
            if (mb != nullptr && mb[0] != '\0' && mf != nullptr && mf[0] != '\0') {
                const int bid = mj_name2id(m_, mjOBJ_BODY, mb);
                if (bid > 0) {
                    move_file_bid_ = bid;
                    move_file_path_ = mf;
                    move_file_z_ = m_->body_pos[3 * bid + 2];
                    // Seed the file from the scene: a file left by an earlier run would otherwise
                    // teleport the body to an old pose on the first poll.
                    const double* q = &m_->body_quat[4 * bid];
                    const double yaw0 = std::atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                                   1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]));
                    std::FILE* sf = std::fopen(mf, "w");
                    if (sf != nullptr) {
                        std::fprintf(sf, "%.4f %.4f %.2f\n", m_->body_pos[3 * bid + 0],
                                     m_->body_pos[3 * bid + 1], yaw0 * 180.0 / M_PI);
                        std::fclose(sf);
                    }
                    std::fprintf(stderr, "[move_body] live control of '%s' (body %d) via %s"
                                 "  (seeded from the scene: %.3f %.3f %.1f deg)\n", mb, bid, mf,
                                 m_->body_pos[3 * bid + 0], m_->body_pos[3 * bid + 1],
                                 yaw0 * 180.0 / M_PI);
                } else {
                    std::fprintf(stderr, "[move_body] HALO_MOVE_FILE set but body '%s' not in scene\n", mb);
                }
            }
        }
        if (move_file_bid_ > 0 && ++move_file_tick_ >= 5) {   // ~10 Hz at a 50 Hz control rate
            move_file_tick_ = 0;
            std::FILE* fp = std::fopen(move_file_path_.c_str(), "r");
            if (fp != nullptr) {
                double mx = 0.0, my = 0.0, mdeg = 0.0;
                if (std::fscanf(fp, "%lf %lf %lf", &mx, &my, &mdeg) == 3) {
                    const double h = (mdeg * M_PI / 180.0) * 0.5;
                    m_->body_pos[3 * move_file_bid_ + 0]  = mx;
                    m_->body_pos[3 * move_file_bid_ + 1]  = my;
                    m_->body_pos[3 * move_file_bid_ + 2]  = move_file_z_;
                    m_->body_quat[4 * move_file_bid_ + 0] = std::cos(h);
                    m_->body_quat[4 * move_file_bid_ + 1] = 0.0;
                    m_->body_quat[4 * move_file_bid_ + 2] = 0.0;
                    m_->body_quat[4 * move_file_bid_ + 3] = std::sin(h);
                    mj_kinematics(m_, d_);
                }
                std::fclose(fp);
            }
        }
        // LIVE OBJECT POSES (HALO_OBJ_FILE). The studio's 3D view reconstructs the scene from the
        // XML and poses the ROBOT from g1/state, so anything the physics moves -- a pushed box --
        // and anything a sender moves -- the human -- is invisible to it and sits frozen at its
        // keyframe. This publishes those poses so the viewer can show them.
        //
        // Cheap by construction: one small text file, rewritten at ~10 Hz on the physics thread
        // (which already reads a file at that rate just above), and skipped entirely when the env
        // var is unset -- so a run that nobody is watching pays one integer compare per step.
        if (live_bid_state_ == -2) {                      // resolve once, lazily
            live_bid_state_ = -1;
            const char* of = std::getenv("HALO_OBJ_FILE");
            if (of != nullptr && of[0] != '\0') {
                live_obj_path_ = of;
                const int pel = mj_name2id(m_, mjOBJ_BODY, "pelvis");
                const int rootb = (pel >= 0) ? m_->body_rootid[pel] : -1;
                for (int j = 0; j < m_->njnt; ++j) {      // movable objects: the box, etc.
                    if (m_->jnt_type[j] != mjJNT_FREE) continue;
                    const int b = m_->jnt_bodyid[j];
                    if (b == rootb) continue;             // the robot travels on g1/state already
                    live_adr_.push_back(m_->jnt_qposadr[j]);
                    live_free_bid_.push_back(b);
                    const char* nm = mj_id2name(m_, mjOBJ_BODY, b);
                    live_names_.emplace_back(nm != nullptr ? nm : "body");
                }
                // The human has no freejoint (it is posed through body_pos), so add it by id.
                if (move_file_bid_ > 0) {
                    live_static_.push_back(move_file_bid_);
                    const char* nm = mj_id2name(m_, mjOBJ_BODY, move_file_bid_);
                    live_names_.emplace_back(nm != nullptr ? nm : "body");
                }
                // HALO_POSE_BODIES: extra JOINTLESS bodies to publish (table, chair, door leaf, ...).
                // Placement already works for any body -- set_object writes body_pos for anything
                // without a freejoint -- but the studio builds its object list from THIS feed, so a
                // body that is never published can never be selected there. Comma-separated names.
                if (const char* pb = std::getenv("HALO_POSE_BODIES")) {
                    std::string spec(pb), one;
                    std::stringstream ss(spec);
                    while (std::getline(ss, one, ',')) {
                        // trim, so "a, b" works as well as "a,b"
                        const auto b0 = one.find_first_not_of(" \t");
                        const auto b1 = one.find_last_not_of(" \t");
                        if (b0 == std::string::npos) continue;
                        one = one.substr(b0, b1 - b0 + 1);
                        const int b = mj_name2id(m_, mjOBJ_BODY, one.c_str());
                        if (b <= 0) {
                            std::fprintf(stderr, "[objects] HALO_POSE_BODIES: no body '%s'\n",
                                         one.c_str());
                            continue;
                        }
                        // A freejoint body is already covered by the scan above, and adding it to
                        // live_static_ would publish its MODEL pose, which physics does not update.
                        bool has_free = false;
                        for (int k = 0; k < m_->body_jntnum[b]; ++k) {
                            if (m_->jnt_type[m_->body_jntadr[b] + k] == mjJNT_FREE) has_free = true;
                        }
                        if (has_free) continue;
                        if (std::find(live_static_.begin(), live_static_.end(), b) != live_static_.end())
                            continue;
                        live_static_.push_back(b);
                        live_names_.emplace_back(one);
                    }
                }
                if (live_adr_.empty() && live_static_.empty()) {
                    live_bid_state_ = -1;
                    std::fprintf(stderr, "[objects] HALO_OBJ_FILE set but the scene has no movable "
                                 "objects and no posed body; not publishing\n");
                } else {
                    live_bid_state_ = 1;
                    std::fprintf(stderr, "[objects] live poses -> %s (", live_obj_path_.c_str());
                    for (size_t i = 0; i < live_names_.size(); ++i)
                        std::fprintf(stderr, "%s%s", i ? ", " : "", live_names_[i].c_str());
                    std::fprintf(stderr, ")\n");
                }
            }
        }
        if (live_bid_state_ == 1 && ++live_tick_ >= 5) {   // ~10 Hz at a 50 Hz control rate
            live_tick_ = 0;
            // temp + rename: the reader must never see a half-written file, and it polls freely.
            const std::string tmp = live_obj_path_ + ".tmp";
            std::FILE* fp = std::fopen(tmp.c_str(), "w");
            if (fp != nullptr) {
                size_t n = 0;
                for (size_t i = 0; i < live_adr_.size(); ++i) {
                    const int a0 = live_adr_[i];
                    std::fprintf(fp, "%s %.5f %.5f %.5f %.6f %.6f %.6f %.6f %d\n",
                                 live_names_[n++].c_str(), d_->qpos[a0 + 0], d_->qpos[a0 + 1],
                                 d_->qpos[a0 + 2], d_->qpos[a0 + 3], d_->qpos[a0 + 4],
                                 d_->qpos[a0 + 5], d_->qpos[a0 + 6],
                                 body_is_hidden_(live_free_bid_[i]) ? 1 : 0);
                }
                for (const int b : live_static_) {
                    std::fprintf(fp, "%s %.5f %.5f %.5f %.6f %.6f %.6f %.6f %d\n",
                                 live_names_[n++].c_str(), m_->body_pos[3 * b + 0],
                                 m_->body_pos[3 * b + 1], m_->body_pos[3 * b + 2],
                                 m_->body_quat[4 * b + 0], m_->body_quat[4 * b + 1],
                                 m_->body_quat[4 * b + 2], m_->body_quat[4 * b + 3],
                                 body_is_hidden_(b) ? 1 : 0);
                }
                std::fclose(fp);
                std::rename(tmp.c_str(), live_obj_path_.c_str());
            }
        }
        // SIM RESET (HALO_RESET_FILE). Holds one number; when it CHANGES the scene goes back to its
        // keyframe -- robot at the start pose, box back where it began, human home. Done on the
        // physics thread because that is what owns m_/d_.
        //
        // The value at startup is READ AND IGNORED: a token left over from an earlier run would
        // otherwise fire a reset on the first tick of every new run.
        if (reset_state_ == -2) {
            reset_state_ = -1;
            const char* rf = std::getenv("HALO_RESET_FILE");
            if (rf != nullptr && rf[0] != '\0') {
                reset_path_ = rf; reset_state_ = 1;
                // Armed on the first step, and the move-file poll does not fire until 5 ticks in,
                // so body_pos/body_quat here are still the scene's own values.
                home_body_pos_.assign(m_->body_pos, m_->body_pos + 3 * m_->nbody);
                home_body_quat_.assign(m_->body_quat, m_->body_quat + 4 * m_->nbody);
                std::FILE* fp = std::fopen(rf, "r");
                if (fp != nullptr) {                      // adopt, do not act on, the current token
                    double tok = -1.0;
                    if (std::fscanf(fp, "%lf", &tok) == 1) reset_seen_ = tok;
                    std::fclose(fp);
                }
                std::fprintf(stderr, "[reset] watching %s\n", reset_path_.c_str());
            }
        }
        if (reset_state_ == 1 && ++reset_tick_ >= 5) {     // ~10 Hz
            reset_tick_ = 0;
            std::FILE* fp = std::fopen(reset_path_.c_str(), "r");
            if (fp != nullptr) {
                double tok = -1.0;
                const bool got = (std::fscanf(fp, "%lf", &tok) == 1);
                std::fclose(fp);
                if (got && tok != reset_seen_) {
                    reset_seen_ = tok;
                    // UNHIDE everything first. Reset means "back to the scene as authored", and an
                    // object hidden before the reset would otherwise stay invisible AND non-colliding
                    // for the rest of the session -- the robot walking through a box that is really
                    // still there in the XML.
                    for (const auto& [g, orig] : hidden_geoms_) {
                        m_->geom_rgba[4 * g + 3] = static_cast<float>(orig[0]);
                        m_->geom_contype[g] = static_cast<int>(orig[1]);
                        m_->geom_conaffinity[g] = static_cast<int>(orig[2]);
                    }
                    const size_t n_unhidden = hidden_geoms_.size();
                    hidden_geoms_.clear();
                    hidden_frozen_.clear();       // the keyframe pose supersedes any frozen one
                    if (m_->nkey > 0) mj_resetDataKeyframe(m_, d_, 0);
                    else              mj_resetData(m_, d_);
                    // qpos is back (robot, box); now put the MODEL-posed bodies back too.
                    if (home_body_pos_.size() == static_cast<size_t>(3 * m_->nbody)) {
                        std::copy(home_body_pos_.begin(), home_body_pos_.end(), m_->body_pos);
                        std::copy(home_body_quat_.begin(), home_body_quat_.end(), m_->body_quat);
                    }
                    mj_forward(m_, d_);
                    // Re-seed the move file from the scene, or the next poll (~100 ms later) would
                    // teleport the human straight back to where it was before the reset -- the
                    // sender still holds the old pose and the file is absolute.
                    if (move_file_bid_ > 0 && !move_file_path_.empty()) {
                        const double* q = &m_->body_quat[4 * move_file_bid_];
                        const double yaw0 = std::atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                                       1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]));
                        std::FILE* sf = std::fopen(move_file_path_.c_str(), "w");
                        if (sf != nullptr) {
                            std::fprintf(sf, "%.4f %.4f %.2f\n",
                                         m_->body_pos[3 * move_file_bid_ + 0],
                                         m_->body_pos[3 * move_file_bid_ + 1],
                                         yaw0 * 180.0 / M_PI);
                            std::fclose(sf);
                        }
                    }
                    std::fprintf(stderr, "[reset] scene restored to its keyframe (token %.3f"
                                 "%s)\n", tok,
                                 n_unhidden ? ", hidden objects restored" : "");
                }
            }
        }
        // OPERATOR OBJECT PLACEMENT (HALO_SET_FILE). Lets the studio drag the box (or anything else)
        // mid-run. Format, one line per body, absolute like every other file here:
        //     <name> <x> <y> <z> <yaw_deg>
        // plus a first line holding a token, so a repeated poll of an unchanged file does nothing --
        // without that, re-applying the same pose every 100 ms would fight physics and pin the box.
        //
        // A FREEJOINT body is placed through qpos (and its velocity zeroed, or it keeps the momentum
        // it had before being picked up and slides away the moment you let go). A jointless body is
        // placed through body_pos/body_quat, the same way the human is.
        if (set_state_ == -2) {
            set_state_ = -1;
            const char* sf = std::getenv("HALO_SET_FILE");
            if (sf != nullptr && sf[0] != '\0') {
                set_path_ = sf; set_state_ = 1;
                // ADOPT the current token WITHOUT acting on it, exactly as the reset watcher does.
                // The file survives between runs, so a placement made in the LAST trial would
                // otherwise be re-applied on the first tick of the NEXT one -- the human, table and
                // box would silently start wherever the previous session left them instead of at
                // their scene pose. The scene XML is the default, and nothing from a past run
                // should override it.
                std::FILE* fp0 = std::fopen(sf, "r");
                if (fp0 != nullptr) {
                    double tok0 = -1.0;
                    if (std::fscanf(fp0, "%lf", &tok0) == 1) set_seen_ = tok0;
                    std::fclose(fp0);
                }
                std::fprintf(stderr, "[set_object] watching %s (ignoring any pose left by a previous "
                             "run; the scene XML is the default)\n", set_path_.c_str());
            }
        }
        if (set_state_ == 1 && ++set_tick_ >= 5) {          // ~10 Hz
            set_tick_ = 0;
            std::FILE* fp = std::fopen(set_path_.c_str(), "r");
            if (fp != nullptr) {
                double tok = -1.0;
                if (std::fscanf(fp, "%lf", &tok) == 1 && tok != set_seen_) {
                    set_seen_ = tok;
                    char nm[128];
                    double x = 0.0, y = 0.0, z = 0.0, ydeg = 0.0;
                    int applied = 0;
                    // Line-based, because a line is either a POSE or a hide/show directive and
                    // fscanf's field-at-a-time parsing cannot tell them apart.
                    char line[256];
                    // finish the token line (a checked call: glibc marks fgets warn_unused_result)
                    const bool more = std::fgets(line, sizeof(line), fp) != nullptr;
                    while (more && std::fgets(line, sizeof(line), fp) != nullptr) {
                        // `hide <name>` / `show <name>`: make an object VANISH from the sim --
                        // invisible AND non-colliding, so the robot cannot bump what it cannot see.
                        // Not a teleport: moving it far away leaves it in the scene's bounds and, for
                        // the move-file body, the poll would drag it straight back.
                        char what[16];
                        if (std::sscanf(line, "%15s %127s", what, nm) == 2 &&
                            (std::strcmp(what, "hide") == 0 || std::strcmp(what, "show") == 0)) {
                            const int b = mj_name2id(m_, mjOBJ_BODY, nm);
                            if (b <= 0) continue;
                            const bool hide = (std::strcmp(what, "hide") == 0);
                            int hfj = -1;
                            for (int k = 0; k < m_->body_jntnum[b]; ++k) {
                                const int j = m_->body_jntadr[b] + k;
                                if (m_->jnt_type[j] == mjJNT_FREE) { hfj = j; break; }
                            }
                            if (hide && hfj >= 0) {
                                const int a0 = m_->jnt_qposadr[hfj];
                                std::array<double, 8> fr{};
                                fr[0] = static_cast<double>(a0);
                                for (int k = 0; k < 7; ++k) fr[static_cast<size_t>(k) + 1] = d_->qpos[a0 + k];
                                hidden_frozen_[b] = fr;
                            } else if (!hide) {
                                hidden_frozen_.erase(b);
                            }
                            for (int gi = 0; gi < m_->body_geomnum[b]; ++gi) {
                                const int g = m_->body_geomadr[b] + gi;
                                if (hide) {
                                    if (hidden_geoms_.find(g) == hidden_geoms_.end()) {
                                        hidden_geoms_[g] = {static_cast<double>(m_->geom_rgba[4 * g + 3]),
                                                            static_cast<double>(m_->geom_contype[g]),
                                                            static_cast<double>(m_->geom_conaffinity[g])};
                                    }
                                    m_->geom_rgba[4 * g + 3] = 0.0F;
                                    m_->geom_contype[g] = 0;
                                    m_->geom_conaffinity[g] = 0;
                                } else {
                                    const auto it = hidden_geoms_.find(g);
                                    if (it == hidden_geoms_.end()) continue;
                                    m_->geom_rgba[4 * g + 3] = static_cast<float>(it->second[0]);
                                    m_->geom_contype[g] = static_cast<int>(it->second[1]);
                                    m_->geom_conaffinity[g] = static_cast<int>(it->second[2]);
                                    hidden_geoms_.erase(it);
                                }
                            }
                            std::fprintf(stderr, "[set_object] %s '%s' (%d geom(s), first alpha "
                                         "now %.2f contype %d, saved %zu)\n", what, nm,
                                         m_->body_geomnum[b],
                                         m_->body_geomnum[b] > 0
                                             ? static_cast<double>(m_->geom_rgba[4 * m_->body_geomadr[b] + 3])
                                             : -1.0,
                                         m_->body_geomnum[b] > 0
                                             ? m_->geom_contype[m_->body_geomadr[b]] : -1,
                                         hidden_geoms_.size());
                            ++applied;
                            continue;
                        }
                        // `name x y z yaw_deg [mode]`. mode 0 (default) = MOVE ONLY, leave the
                        // orientation untouched; mode 1 = also turn by yaw_deg as a DELTA about Z.
                        //
                        // Both halves of that matter. Writing a pure-Z quaternion (what this used to
                        // do) DESTROYS a body whose scene orientation is not a plain yaw: the table's
                        // quat is -0.342 -0.342 0.619 0.619, a 90 deg X rotation that stands its Y-up
                        // mesh upright, so replacing it laid the table over. And a DELTA composed onto
                        // the current quat needs no absolute yaw, which cannot even be extracted
                        // meaningfully from such a quaternion.
                        int mode = 0;
                        const int got = std::sscanf(line, "%127s %lf %lf %lf %lf %d",
                                                    nm, &x, &y, &z, &ydeg, &mode);
                        if (got < 5) continue;
                        if (got < 6) mode = 0;
                        {
                        const int b = mj_name2id(m_, mjOBJ_BODY, nm);
                        if (b <= 0) continue;
                        if (!rotate_only_init_) {
                            rotate_only_init_ = true;
                            if (const char* ro = std::getenv("HALO_POSE_ROTATE_ONLY")) {
                                std::string spec(ro), one;
                                std::stringstream ss(spec);
                                while (std::getline(ss, one, ',')) {
                                    const auto c0 = one.find_first_not_of(" \t");
                                    const auto c1 = one.find_last_not_of(" \t");
                                    if (c0 == std::string::npos) continue;
                                    one = one.substr(c0, c1 - c0 + 1);
                                    const int rb = mj_name2id(m_, mjOBJ_BODY, one.c_str());
                                    if (rb > 0) rotate_only_.push_back(rb);
                                }
                                if (!rotate_only_.empty())
                                    std::fprintf(stderr, "[set_object] rotate-only: %s\n", ro);
                            }
                        }
                        const bool turn_only = std::find(rotate_only_.begin(), rotate_only_.end(), b)
                                               != rotate_only_.end();
                        const double h = (ydeg * M_PI / 180.0) * 0.5;
                        int fj = -1;
                        for (int k = 0; k < m_->body_jntnum[b]; ++k) {
                            const int j = m_->body_jntadr[b] + k;
                            if (m_->jnt_type[j] == mjJNT_FREE) { fj = j; break; }
                        }
                        if (fj >= 0) {
                            const int a0 = m_->jnt_qposadr[fj];
                            d_->qpos[a0 + 0] = x; d_->qpos[a0 + 1] = y; d_->qpos[a0 + 2] = z;
                            if (mode == 1) {           // compose onto the CURRENT pose, not replace
                                const double dq[4] = {std::cos(h), 0.0, 0.0, std::sin(h)};
                                const double q[4] = {d_->qpos[a0 + 3], d_->qpos[a0 + 4],
                                                     d_->qpos[a0 + 5], d_->qpos[a0 + 6]};
                                d_->qpos[a0 + 3] = dq[0]*q[0] - dq[1]*q[1] - dq[2]*q[2] - dq[3]*q[3];
                                d_->qpos[a0 + 4] = dq[0]*q[1] + dq[1]*q[0] + dq[2]*q[3] - dq[3]*q[2];
                                d_->qpos[a0 + 5] = dq[0]*q[2] - dq[1]*q[3] + dq[2]*q[0] + dq[3]*q[1];
                                d_->qpos[a0 + 6] = dq[0]*q[3] + dq[1]*q[2] - dq[2]*q[1] + dq[3]*q[0];
                            }
                            const int v0 = m_->jnt_dofadr[fj];
                            for (int k = 0; k < 6; ++k) d_->qvel[v0 + k] = 0.0;
                        } else {
                            if (!turn_only) {          // a door keeps its hinge where it is
                                m_->body_pos[3 * b + 0]  = x;
                                m_->body_pos[3 * b + 1]  = y;
                                m_->body_pos[3 * b + 2]  = z;
                            }
                            if (mode == 1) {           // compose, never replace
                                const double dq[4] = {std::cos(h), 0.0, 0.0, std::sin(h)};
                                const double* q = &m_->body_quat[4 * b];
                                const double nq[4] = {
                                    dq[0]*q[0] - dq[1]*q[1] - dq[2]*q[2] - dq[3]*q[3],
                                    dq[0]*q[1] + dq[1]*q[0] + dq[2]*q[3] - dq[3]*q[2],
                                    dq[0]*q[2] - dq[1]*q[3] + dq[2]*q[0] + dq[3]*q[1],
                                    dq[0]*q[3] + dq[1]*q[2] - dq[2]*q[1] + dq[3]*q[0]};
                                for (int k = 0; k < 4; ++k) m_->body_quat[4 * b + k] = nq[k];
                            }
                        }
                        // If this body is ALSO the one driven by HALO_MOVE_FILE, that file holds an
                        // absolute pose and is polled every ~100 ms -- so without rewriting it the
                        // poll would drag the body straight back and the placement would look like
                        // it silently failed. Keep the sender's file in step with reality.
                        if (b == move_file_bid_ && !move_file_path_.empty()) {
                            // That file is absolute "x y yaw_deg", so read the yaw back OUT of the
                            // body's quaternion after the edit rather than echoing the request: with
                            // mode 0 the orientation did not change at all, and with mode 1 `ydeg`
                            // was a delta, not the resulting angle.
                            const double* q = &m_->body_quat[4 * b];
                            const double yaw_now = std::atan2(2.0 * (q[0]*q[3] + q[1]*q[2]),
                                                              1.0 - 2.0 * (q[2]*q[2] + q[3]*q[3]));
                            std::FILE* mfp = std::fopen(move_file_path_.c_str(), "w");
                            if (mfp != nullptr) {
                                std::fprintf(mfp, "%.4f %.4f %.2f\n",
                                             m_->body_pos[3 * b + 0], m_->body_pos[3 * b + 1],
                                             yaw_now * 180.0 / M_PI);
                                std::fclose(mfp);
                            }
                        }
                        ++applied;
                        }
                    }
                    if (applied > 0) {
                        mj_forward(m_, d_);
                        std::fprintf(stderr, "[set_object] placed %d body(ies)\n", applied);
                    }
                }
                std::fclose(fp);
            }
        }
        // HOLD every hidden freejoint body. Without this it free-falls (no collision while hidden) and
        // `show` then re-enables contacts on a body buried in the floor, which MuJoCo resolves by
        // launching it. Cheap: the map is empty unless something is actually hidden.
        for (const auto& [hb, fr] : hidden_frozen_) {
            (void)hb;
            const int a0 = static_cast<int>(fr[0]);
            for (int k = 0; k < 7; ++k) d_->qpos[a0 + k] = fr[static_cast<size_t>(k) + 1];
            const int v0 = m_->jnt_dofadr[m_->body_jntadr[hb]];
            for (int k = 0; k < 6; ++k) d_->qvel[v0 + k] = 0.0;
        }
    }

    // ── headless live body move (HALO_MOVE_BODY + HALO_MOVE_FILE; see step()) ──
    int         move_file_bid_{-2};      // -2 = not resolved yet, -1 = disabled, >0 = body id
    std::string move_file_path_{};
    double      move_file_z_{0.0};
    int         move_file_tick_{0};

    // ── LIVE object poses + sim RESET (sim only; HALO_OBJ_FILE / HALO_RESET_FILE) ──
    // Files, not a new ZMQ channel, for two reasons: RobotState has no field for object poses and
    // regenerating the protos needs protoc 29 + network (not available on every machine), and the
    // backend ALREADY polls a file every step for HALO_MOVE_FILE, so this is the established
    // mechanism here rather than a new one.
    int                      live_bid_state_{-2};   // -2 = unresolved, -1 = disabled
    std::string              live_obj_path_{};
    std::vector<int>         live_adr_{};           // freejoint qpos addresses (movable objects)
    std::vector<int>         live_free_bid_{};      // body id per live_adr_ entry (for the hidden flag)
    std::vector<int>         live_static_{};        // body ids posed via body_pos (human/lead)
    std::vector<std::string> live_names_{};         // parallel to live_adr_ then live_static_
    int                      live_tick_{0};
    int                      reset_state_{-2};      // -2 = unresolved, -1 = disabled, 1 = armed
    std::string              reset_path_{};
    double                   reset_seen_{-1.0};
    int                      reset_tick_{0};
    // The scene's ORIGINAL body_pos/body_quat. Needed because the human is posed by mutating the
    // MODEL (body_pos), not qpos -- so mj_resetDataKeyframe, which only restores DATA, leaves it
    // exactly where the sender last put it. Captured before anything can have moved it.
    std::vector<mjtNum>      home_body_pos_{};
    std::vector<mjtNum>      home_body_quat_{};
    // Operator-commanded object poses (HALO_SET_FILE). HALO_MOVE_FILE drives exactly ONE body and only
    // through body_pos, so it cannot place the box -- the box is a FREEJOINT, whose pose lives in
    // qpos. This handles both kinds, for any body, by name.
    // Bodies the operator may TURN but not MOVE (HALO_POSE_ROTATE_ONLY) -- a door swings on its hinge,
    // it does not slide across the room. Convenient here: `door_leaf`'s body origin IS the hinge, so
    // rewriting body_quat about Z swings the leaf correctly and its position must simply be left alone.
    std::vector<int>         rotate_only_{};
    bool                     rotate_only_init_{false};
    int                      set_state_{-2};
    std::string              set_path_{};
    double                   set_seen_{-1.0};
    int                      set_tick_{0};
    // Original rgba alpha + contype/conaffinity per geom, so `show` can put back exactly what `hide`
    // took away. Captured lazily the first time a body is hidden -- reading it at load would mean
    // carrying three arrays for every scene whether or not anything is ever hidden.
    std::map<int, std::array<double, 3>> hidden_geoms_{};
    // FROZEN POSE of each hidden freejoint body: {qpos adr, x,y,z, qw,qx,qy,qz}. A hidden body has
    // contype/conaffinity 0, so it stops colliding and FREE-FALLS -- and re-enabling collision once
    // it is embedded in the floor makes MuJoCo eject it violently (a box was found 265 m up). While
    // hidden, its pose is held and its velocity zeroed, so `show` puts it back exactly where it was.
    std::map<int, std::array<double, 8>> hidden_frozen_{};

    // Is any geom of this body currently hidden? Reported on the live pose feed so the studio, which
    // draws from the scene XML and knows nothing about runtime visibility, can match the sim after a
    // page reload instead of drawing an object the robot can no longer touch.
    [[nodiscard]] bool body_is_hidden_(int b) const noexcept {
        if (b <= 0 || m_ == nullptr) return false;
        for (int gi = 0; gi < m_->body_geomnum[b]; ++gi) {
            if (hidden_geoms_.count(m_->body_geomadr[b] + gi) != 0U) return true;
        }
        return false;
    }

    // ── scene object pose log state (sim only; see init_object_log) ──

    // Render the depth camera into depth_tb_ (metric meters, top-down 48x84).
    // Serialized with render_ego_image by ego_render_mu_ (they share the ego
    // context+scene+data); reads the qpos/qvel snapshot under snapshot_mu_ so it
    // never races the control thread's mj_step writes. Called from step().
    void render_depth_frame_() noexcept {
        const std::scoped_lock render_lock{ego_render_mu_};
        ego_gl_make_current();
        {
            const std::scoped_lock data_lock{snapshot_mu_};
            std::memcpy(ego_data_->qpos, snapshot_qpos_.data(),
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nq));
            std::memcpy(ego_data_->qvel, snapshot_qvel_.data(),
                         sizeof(mjtNum) * static_cast<std::size_t>(m_->nv));
            ego_data_->time = snapshot_time_;
        }
        mj_forward(m_, ego_data_);

        mjrRect viewport{0, 0, depth_w_, depth_h_};
        mjv_updateScene(m_, ego_data_, ego_option_, /*pert=*/nullptr,
                         &depth_camera_, mjCAT_ALL, ego_scene_);
        mjr_render(viewport, ego_scene_, ego_context_);
        mjr_readPixels(/*rgb=*/nullptr, depth_gl_buf_.data(), viewport, ego_context_);
        // Release before ego_render_mu_ drops: two threads share this context (control loop,
        // server-thread ego render) and EGL refuses a context still current on another thread
        // (EGL_BAD_ACCESS). GLFW tolerates it, so macOS does not show the problem.
        ego_gl_release();

        // Linearize OpenGL depth -> metric meters + flip rows bottom-up (GL) to
        // top-down (mujoco.Renderer / the Python sim), so the frame matches the
        // policy's trained student_depth. d==1 (no geometry) -> far (the encoder
        // clamps to clip_max downstream).
        auto& slot = depth_tb_.producer_slot();
        gl_depth_to_metric(depth_gl_buf_.data(), depth_w_, depth_h_,
                           depth_near_, depth_far_, slot.data.data());
        slot.seq = ++depth_seq_;
        slot.stamp_ns = core::MonotonicClock::now().count();  // render time (same clock as the real producers)
        depth_tb_.publish();
    }

    void viewer_loop() noexcept {
        if (glfwInit() == 0) {
            std::fprintf(stderr, "[mujoco viewer] glfwInit failed; viewer disabled\n");
            return;
        }
        GLFWwindow* window = glfwCreateWindow(viewer_w_, viewer_h_,
                                                "deploy_mujoco", nullptr, nullptr);
        if (window == nullptr) {
            std::fprintf(stderr, "[mujoco viewer] glfwCreateWindow failed; viewer disabled\n");
            return;
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        mjData* viewer_data = mj_makeData(m_);
        if (viewer_data == nullptr) {
            std::fprintf(stderr, "[mujoco viewer] mj_makeData failed; viewer disabled\n");
            glfwDestroyWindow(window);
            return;
        }

        // Ghost data is a separate mjData used only to run forward kinematics
        // on the motion playback pose. Never stepped, never collides — its
        // geom_xpos/geom_xmat are appended to the scene for rendering only.
        // The pose comes from ghost_motion_ + ghost_remap_, indexed by viewer
        // time. Ghost is N==29-only; for other instantiations the block
        // compiles out and no ghost mjData is allocated.
        const bool ghost_enabled = (N == 29)
            && ghost_motion_ != nullptr && ghost_motion_->num_frames() > 0
            && ghost_remap_ != nullptr;
        mjData* ghost_data = nullptr;
        if (ghost_enabled) {
            ghost_data = mj_makeData(m_);
            if (ghost_data == nullptr) {
                std::fprintf(stderr,
                             "[mujoco viewer] ghost mj_makeData failed; ghost disabled\n");
            }
        }

        // Heap-allocate the big MuJoCo structs. mjvScene + mjrContext together
        // are several MB — too large for the 512 KB default stack that macOS
        // gives non-main pthreads, which causes EXC_BAD_ACCESS in the thread
        // entry. On Linux the main thread has 8 MB so stack-allocating would
        // also work, but one path is simpler to reason about.
        mjvScene*   scene   = new mjvScene{};
        mjvCamera*  camera  = new mjvCamera{};
        mjvOption*  option  = new mjvOption{};
        mjrContext* context = new mjrContext{};
        mjv_defaultScene(scene);
        // Match mujoco.viewer.launch_passive: bare MjvCamera defaults
        // (distance=2, azim=90, elev=-45, lookat=origin). launch_passive
        // does NOT call mjv_defaultFreeCamera — overriding to extent-based
        // distance pushes the camera ~10x too far on large environment scenes.
        // We then nudge `distance` a bit farther (default 2 is too tight on
        // the G1 scene); the Python sim viewers do the same nudge after
        // launch_passive.
        mjv_defaultCamera(camera);
        camera->distance = 4.5;
        mjv_defaultOption(option);
        mjr_defaultContext(context);
        mjv_makeScene(m_, scene, /*maxgeom=*/4000);
        mjr_makeContext(m_, context, mjFONTSCALE_100);

        ViewerCtx ctx{};
        ctx.model       = m_;
        ctx.camera      = camera;
        ctx.scene       = scene;
        ctx.option      = option;
        ctx.viewer_data = viewer_data;
        ctx.perturb     = &perturb_;
        ctx.perturb_mu  = &perturb_mu_;
        glfwSetWindowUserPointer(window, &ctx);
        glfwSetMouseButtonCallback(window, on_mouse_button);
        glfwSetCursorPosCallback(window, on_cursor_pos);
        glfwSetScrollCallback(window, on_scroll);
        glfwSetKeyCallback(window, on_key);

        std::fprintf(stderr,
            "[mujoco viewer] Mouse: drag to orbit, right-drag to pan, scroll to zoom.\n"
            "                Ctrl+drag a body to apply force/torque.\n");

        static core::RateProbe probe{"mujoco_viewer"};

        while (!viewer_should_stop_.load(std::memory_order_acquire)
               && glfwWindowShouldClose(window) == 0) {
            probe.tick();
            auto probe_scope = probe.scope();
            {
                const std::scoped_lock lock{snapshot_mu_};
                std::memcpy(viewer_data->qpos, snapshot_qpos_.data(),
                             sizeof(mjtNum) * static_cast<std::size_t>(m_->nq));
                std::memcpy(viewer_data->qvel, snapshot_qvel_.data(),
                             sizeof(mjtNum) * static_cast<std::size_t>(m_->nv));
                viewer_data->time = snapshot_time_;
            }
            mj_forward(m_, viewer_data);

            mjrRect viewport{0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

            // Pass the perturbation into the scene so the rubber-band gets drawn.
            {
                const std::scoped_lock lock{perturb_mu_};
                mjv_updateScene(m_, viewer_data, option, &perturb_, camera,
                                 mjCAT_ALL, scene);
            }

            // Append ghost geoms — N == 29 only. For other instantiations the
            // block compiles out: ghost_data stays null, the body is skipped.
            if constexpr (N == 29) {
                if (ghost_data != nullptr) {
                    set_ghost_qpos(ghost_data, viewer_data->time);
                    mj_kinematics(m_, ghost_data);
                    const int ngeom_before = scene->ngeom;
                    mjv_addGeoms(m_, ghost_data, option, /*pert=*/nullptr,
                                  mjCAT_DYNAMIC, scene);
                    for (int i = ngeom_before; i < scene->ngeom; ++i) {
                        mjvGeom* g  = &scene->geoms[i];
                        g->rgba[0]  = ghost_rgba_[0];
                        g->rgba[1]  = ghost_rgba_[1];
                        g->rgba[2]  = ghost_rgba_[2];
                        g->rgba[3]  = ghost_rgba_[3];
                        g->emission    = 0.15F;
                        g->specular    = 0.0F;
                        g->reflectance = 0.0F;
                    }
                }
            }

            mjr_render(viewport, scene, context);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        viewer_should_stop_.store(true, std::memory_order_release);

        mjv_freeScene(scene);
        mjr_freeContext(context);
        delete scene;
        delete camera;
        delete option;
        delete context;
        mj_deleteData(viewer_data);
        if (ghost_data != nullptr) {
            mj_deleteData(ghost_data);
        }
        glfwDestroyWindow(window);
        // NOTE: no glfwTerminate() here — GLFW is process-global and the ego
        // renderer may still be alive. The single teardown lives in the dtor.
    }

    // Write base pose + joint positions for the ghost pose at simulation-time
    // `t` into `gd->qpos`. Frame index is clamped to the motion's valid range.
    // dof_poss are stored in training order; the remap converts back to
    // deployment order (MuJoCo qpos layout). N == 29 only.
    void set_ghost_qpos(mjData* gd, mjtNum t) noexcept {
        if constexpr (N == 29) {
            if (ghost_motion_ == nullptr || ghost_remap_ == nullptr) return;
            const auto& motion = *ghost_motion_;
            const auto& remap  = *ghost_remap_;
            const std::int32_t max_idx = motion.max_index();
            std::int32_t idx = static_cast<std::int32_t>(
                t * static_cast<mjtNum>(motion.output_fps));
            if (idx < 0)        idx = 0;
            if (idx > max_idx)  idx = max_idx;

            // Base free-joint qpos: [x, y, z, qw, qx, qy, qz].
            gd->qpos[0] = static_cast<mjtNum>(motion.base_poss(idx, 0));
            gd->qpos[1] = static_cast<mjtNum>(motion.base_poss(idx, 1));
            gd->qpos[2] = static_cast<mjtNum>(motion.base_poss(idx, 2));
            gd->qpos[3] = static_cast<mjtNum>(motion.base_rots(idx, 0));
            gd->qpos[4] = static_cast<mjtNum>(motion.base_rots(idx, 1));
            gd->qpos[5] = static_cast<mjtNum>(motion.base_rots(idx, 2));
            gd->qpos[6] = static_cast<mjtNum>(motion.base_rots(idx, 3));

            // Joints: training → deployment order via remap.train_to_deploy.
            for (std::size_t i = 0; i < 29; ++i) {
                const std::int32_t src = remap.train_to_deploy[i];
                gd->qpos[7 + static_cast<std::ptrdiff_t>(i)]
                    = static_cast<mjtNum>(
                        motion.dof_poss(idx, static_cast<Eigen::Index>(src)));
            }
        } else {
            (void)gd;
            (void)t;
        }
    }

    mjModel*               m_{nullptr};
    mjData*                d_{nullptr};
    int                    substeps_{1};

    JointVec target_pos_{JointVec::Zero()};
    JointVec target_vel_{JointVec::Zero()};
    JointVec kp_{JointVec::Zero()};
    JointVec kd_{JointVec::Zero()};
    JointVec ff_torque_{JointVec::Zero()};
    JointVec max_torque_{JointVec::Zero()};
    bool     clip_torque_{false};
    JointVec kp_scale_{JointVec::Ones()};
    JointVec kd_scale_{JointVec::Ones()};

    // Viewer plumbing.
    bool                  viewer_enabled_{false};
    int                   viewer_w_{1200};
    int                   viewer_h_{900};
    std::atomic<bool>     viewer_should_stop_{false};
    std::thread           viewer_thread_;

    mutable std::mutex    snapshot_mu_;
    std::vector<mjtNum>   snapshot_qpos_;
    std::vector<mjtNum>   snapshot_qvel_;
    mjtNum                snapshot_time_{0.0};

    // Depth renderer (depth student, sim). REUSES the ego offscreen context/scene
    // (ego_context_/ego_scene_/ego_data_/ego_option_) under ego_render_mu_, so
    // render_depth_frame_() never races render_ego_image. Rendered every
    // depth_decim_-th step(); published as metric-meter DepthFrames into
    // depth_tb_, which the control loop drains via depth_inbox().
    bool                               depth_active_{false};
    int                                depth_cam_id_{-1};
    int                                depth_w_{g1::DepthFrame::kCols};
    int                                depth_h_{g1::DepthFrame::kRows};
    float                              depth_clip_min_{0.0F};  // logged only; the obs group encodes
    float                              depth_clip_max_{0.0F};
    // Render every depth_decim_-th step() (see init_depth_renderer's obs_decim).
    int                                depth_decim_{1};
    int                                depth_step_{0};
    double                             depth_near_{0.0};
    double                             depth_far_{0.0};
    mjvCamera                          depth_camera_{};
    std::vector<float>                 depth_gl_buf_{};      // raw GL depth [w*h], bottom-up
    core::TripleBuffer<g1::DepthFrame> depth_tb_{};
    std::int64_t                       depth_seq_{0};

    // ── Ego GL-context backend (EGL on Linux, hidden GLFW window on macOS) ────
    // Abstracts headless GL-context create / make-current / release / destroy so
    // init_ego_renderer(), render_ego_image() and the dtor stay backend-neutral.
    [[nodiscard]] static const char* ego_gl_backend_name() noexcept {
#if defined(RT_DEPLOY_EGL_EGO)
        return "EGL";
#else
        return "GLFW";
#endif
    }

#if defined(RT_DEPLOY_EGL_EGO)
    // Acquire and initialize a headless EGL display. eglGetDisplay(
    // EGL_DEFAULT_DISPLAY) maps to the X11 display and fails on a headless box,
    // so enumerate GPU devices via the EGL device extension and create a
    // platform display from one — the path mujoco.Renderer / dm_control use
    // headless. Falls back to the default display when a server IS present.
    // Returns an already-eglInitialize'd display, or EGL_NO_DISPLAY.
    [[nodiscard]] static EGLDisplay egl_acquire_headless_display() noexcept {
        auto query_devices = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
            eglGetProcAddress("eglQueryDevicesEXT"));
        auto get_platform_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));
        if (query_devices != nullptr && get_platform_display != nullptr) {
            constexpr EGLint kMaxDevices = 16;
            EGLDeviceEXT     devices[kMaxDevices];
            EGLint           n_devices = 0;
            if (query_devices(kMaxDevices, devices, &n_devices) == EGL_TRUE) {
                for (EGLint i = 0; i < n_devices; ++i) {
                    EGLDisplay d = get_platform_display(
                        EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
                    if (d == EGL_NO_DISPLAY) continue;
                    EGLint mj = 0, mn = 0;
                    if (eglInitialize(d, &mj, &mn) == EGL_TRUE) {
                        std::fprintf(stderr,
                            "[ego_renderer] EGL device %d initialized (EGL %d.%d)\n",
                            static_cast<int>(i), static_cast<int>(mj),
                            static_cast<int>(mn));
                        return d;
                    }
                }
            }
        }
        // Fallback: default display (works when a display server IS present).
        EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (d != EGL_NO_DISPLAY) {
            EGLint mj = 0, mn = 0;
            if (eglInitialize(d, &mj, &mn) == EGL_TRUE) return d;
        }
        return EGL_NO_DISPLAY;
    }
#endif

    // Create the headless GL context and make it current on the calling thread.
    [[nodiscard]] bool ego_gl_create(int width, int height) noexcept {
#if defined(RT_DEPLOY_EGL_EGO)
        (void)width; (void)height;  // surfaceless: size lives in the offscreen FBO
        ego_egl_display_ = egl_acquire_headless_display();  // returns an INITIALIZED display
        if (ego_egl_display_ == EGL_NO_DISPLAY) {
            std::fprintf(stderr,
                "[ego_renderer] no usable EGL display (tried device + default); offscreen disabled\n");
            return false;
        }
        const EGLint cfg_attribs[] = {
            EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
            EGL_RED_SIZE,        8,
            EGL_GREEN_SIZE,      8,
            EGL_BLUE_SIZE,       8,
            EGL_DEPTH_SIZE,      24,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_NONE};
        EGLConfig cfg = nullptr;
        EGLint    n_cfg = 0;
        if (eglChooseConfig(ego_egl_display_, cfg_attribs, &cfg, 1, &n_cfg) == EGL_FALSE
            || n_cfg < 1) {
            std::fprintf(stderr, "[ego_renderer] eglChooseConfig failed; offscreen disabled\n");
            eglTerminate(ego_egl_display_); ego_egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        if (eglBindAPI(EGL_OPENGL_API) == EGL_FALSE) {
            std::fprintf(stderr, "[ego_renderer] eglBindAPI(OpenGL) failed; offscreen disabled\n");
            eglTerminate(ego_egl_display_); ego_egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        ego_egl_context_ = eglCreateContext(ego_egl_display_, cfg, EGL_NO_CONTEXT, nullptr);
        if (ego_egl_context_ == EGL_NO_CONTEXT) {
            std::fprintf(stderr, "[ego_renderer] eglCreateContext failed; offscreen disabled\n");
            eglTerminate(ego_egl_display_); ego_egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        if (eglMakeCurrent(ego_egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE,
                           ego_egl_context_) == EGL_FALSE) {
            std::fprintf(stderr, "[ego_renderer] eglMakeCurrent failed; offscreen disabled\n");
            eglDestroyContext(ego_egl_display_, ego_egl_context_); ego_egl_context_ = EGL_NO_CONTEXT;
            eglTerminate(ego_egl_display_); ego_egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        return true;
#else
        if (glfwInit() == 0) {
            std::fprintf(stderr, "[ego_renderer] glfwInit failed; offscreen disabled\n");
            return false;
        }
        glfwWindowHint(GLFW_VISIBLE,   GLFW_FALSE);
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        ego_window_ = glfwCreateWindow(width, height, "ego_offscreen", nullptr, nullptr);
        // Reset hints so a later viewer glfwCreateWindow doesn't inherit
        // VISIBLE=FALSE / DECORATED=FALSE (GLFW hints are global, sticky state).
        glfwDefaultWindowHints();
        if (ego_window_ == nullptr) {
            std::fprintf(stderr, "[ego_renderer] glfwCreateWindow failed; offscreen disabled\n");
            return false;
        }
        glfwMakeContextCurrent(ego_window_);
        return true;
#endif
    }

    void ego_gl_make_current() noexcept {
#if defined(RT_DEPLOY_EGL_EGO)
        eglMakeCurrent(ego_egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, ego_egl_context_);
#else
        glfwMakeContextCurrent(ego_window_);
#endif
    }

    void ego_gl_release() noexcept {
#if defined(RT_DEPLOY_EGL_EGO)
        eglMakeCurrent(ego_egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
#else
        glfwMakeContextCurrent(nullptr);
#endif
    }

    // Tear down the GL context. Call after the mjr/mjv resources are freed.
    void ego_gl_destroy() noexcept {
#if defined(RT_DEPLOY_EGL_EGO)
        if (ego_egl_display_ != EGL_NO_DISPLAY) {
            eglMakeCurrent(ego_egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (ego_egl_context_ != EGL_NO_CONTEXT) {
                eglDestroyContext(ego_egl_display_, ego_egl_context_);
                ego_egl_context_ = EGL_NO_CONTEXT;
            }
            eglTerminate(ego_egl_display_);
            ego_egl_display_ = EGL_NO_DISPLAY;
        }
#else
        if (ego_window_ != nullptr) { glfwDestroyWindow(ego_window_); ego_window_ = nullptr; }
#endif
    }

    // Ego-camera offscreen renderer (optional; created on demand by
    // init_ego_renderer()). The render path snapshots qpos/qvel under
    // snapshot_mu_, runs mj_forward on a private mjData, and reads pixels back
    // via mjr_readPixels (top-down RGB). ego_render_mu_ serializes concurrent
    // render_ego_image() calls (the server thread). Inert until initialized.
    bool                  ego_initialized_{false};
    int                   ego_cam_id_{-1};
    int                   ego_w_{0};
    int                   ego_h_{0};
    GLFWwindow*           ego_window_{nullptr};
#if defined(RT_DEPLOY_EGL_EGO)
    EGLDisplay            ego_egl_display_{EGL_NO_DISPLAY};
    EGLContext            ego_egl_context_{EGL_NO_CONTEXT};
#endif
    mjvScene*             ego_scene_{nullptr};
    mjvCamera*            ego_camera_{nullptr};
    mjvOption*            ego_option_{nullptr};
    mjrContext*           ego_context_{nullptr};
    mjData*               ego_data_{nullptr};
    std::mutex            ego_render_mu_;
    std::vector<std::uint8_t> ego_rgb_buffer_;
    std::vector<std::uint8_t> ego_rgb_flipped_;
    // OAK-1 fisheye FPV (optional): render a wide pinhole (calibrated intrinsics
    // baked in configure_oak_fisheye) at oak_src_w_ x oak_src_h_, then warp it
    // into the calibrated fisheye (oak_map_x_/y_ index into the source render;
    // output is oak_out_w_ x oak_out_h_ = 1280x800). Enabled when the bound
    // camera name contains "oak". The camera MOUNT/pose is NOT here — it comes from the bound MJCF camera (`oak1`).
    bool                      oak_enabled_{false};
    bool                      oak_warp_{false};
    int                       oak_out_w_{0};
    int                       oak_out_h_{0};
    int                       oak_src_w_{0};
    int                       oak_src_h_{0};
    double                    oak_fx_render_{0.0};
    double                    oak_fy_render_{0.0};
    double                    oak_cx_render_{0.0};
    double                    oak_cy_render_{0.0};
    std::vector<float>        oak_map_x_;
    std::vector<float>        oak_map_y_;
    std::vector<std::uint8_t> oak_rgb_out_;

    // Mouse perturbation: viewer thread updates refpos/refquat; control thread
    // calls mjv_applyPerturbForce inside step() under the same mutex.
    std::mutex            perturb_mu_;
    mjvPerturb            perturb_{};

    // Ghost visualization — owned by the viewer thread after construction,
    // so no synchronization needed on these. Set once from the config.
    // Inert for N != 29 (the if-constexpr branch in viewer_loop compiles out).
    const policy::MotionData*         ghost_motion_{nullptr};
    const policy::JointRemap<29>*     ghost_remap_{nullptr};
    std::array<float, 4>              ghost_rgba_{0.4F, 0.8F, 1.0F, 0.45F};
};

}  // namespace

template <std::size_t N>
core::Result<std::unique_ptr<IRobotBackend<N>>>
make_mujoco_backend(const MujocoBackendConfig<N>& cfg) {
    using ResT = core::Result<std::unique_ptr<IRobotBackend<N>>>;

    const std::string& xml = cfg.xml_override.empty() ? cfg.xml_path : cfg.xml_override;
    std::array<char, 1024> err_buf{};
    mjModel*               m = nullptr;
    if (cfg.depth_render_enabled) {
        // Inject the depth student's camera at load via MjSpec — the shared
        // scene usually lacks it. Mirrors the Python sim's runtime MjSpec
        // inject; the pose is already MuJoCo-body-frame. If the parent body is
        // missing the model still loads, without the camera, and
        // init_depth_renderer then fails (no inbox).
        mjSpec* spec = mj_parseXML(xml.c_str(), nullptr,
                                    err_buf.data(), static_cast<int>(err_buf.size()));
        if (spec == nullptr) {
            return typename ResT::Error{std::string{"mj_parseXML failed: "} + err_buf.data()};
        }
        mjsBody* body = mjs_findBody(spec, cfg.depth_cam_parent_body.c_str());
        if (body != nullptr) {
            mjsCamera* cam = mjs_addCamera(body, nullptr);
#if mjVERSION_HEADER >= 334   // mjs_setName exists from MuJoCo 3.3.4; earlier specs carry an mjString* name
            mjs_setName(cam->element, cfg.depth_camera_name.c_str());
#else
            mjs_setString(cam->name, cfg.depth_camera_name.c_str());
#endif
            for (int i = 0; i < 3; ++i) {
                cam->pos[i] = static_cast<double>(cfg.depth_cam_pos[static_cast<std::size_t>(i)]);
            }
            for (int i = 0; i < 4; ++i) {
                cam->quat[i] = static_cast<double>(cfg.depth_cam_quat_wxyz[static_cast<std::size_t>(i)]);
            }
            cam->fovy = static_cast<double>(cfg.depth_cam_fovy);
            // mode defaults to fixed for a camera with no target — no need to set.
        } else {
            std::fprintf(stderr, "[make_mujoco_backend] depth parent body '%s' not found; "
                                 "depth camera not injected.\n",
                         cfg.depth_cam_parent_body.c_str());
        }
        m = mj_compile(spec, nullptr);
        if (m == nullptr) {
            const std::string why = mjs_getError(spec);
            mj_deleteSpec(spec);
            return typename ResT::Error{"mj_compile (depth-injected model) failed: " + why};
        }
        mj_deleteSpec(spec);
        // No multisampling on the offscreen buffer the depth camera reads (all
        // GL contexts are made later, from the backend, so this lands before
        // any mjr_makeContext). Some Linux/EGL drivers resolve a multisampled
        // DEPTH buffer by averaging samples, which blends foreground and
        // background along every silhouette edge: up to 1.2 m of error was
        // reported there, i.e. obstacles that do not exist. Harmless on macOS.
        m->vis.quality.offsamples = 0;
    } else {
        m = mj_loadXML(xml.c_str(), nullptr,
                        err_buf.data(), static_cast<int>(err_buf.size()));
        if (m == nullptr) {
            return typename ResT::Error{std::string{"mj_loadXML failed: "} + err_buf.data()};
        }
    }
    if (m->nu < static_cast<int>(N)) {
        mj_deleteModel(m);
        return typename ResT::Error{std::string{"MuJoCo model has fewer than "}
                            + std::to_string(N) + " actuators"};
    }
    // The whole backend hard-codes the floating-base generalized-coordinate
    // layout: base pos at qpos[0..2], base quat at qpos[3..6], the N joints at
    // qpos[7..7+N), base ang-vel at qvel[3..5], joint vels at qvel[6..6+N).
    // Reject any model that is not a free-base model with room for N joints,
    // so read_state/step/the initial-pose writes below cannot index past the
    // ends of MuJoCo's qpos/qvel arrays (silent OOB read in the hot path and
    // OOB write during construction otherwise).
    if (m->nq < 7 + static_cast<int>(N) || m->nv < 6 + static_cast<int>(N)) {
        mj_deleteModel(m);
        return typename ResT::Error{
            std::string{"MuJoCo model is not a floating-base model with at least 7+"}
            + std::to_string(N) + " qpos and 6+" + std::to_string(N) + " qvel"};
    }

    mjData* d = mj_makeData(m);
    if (d == nullptr) {
        mj_deleteModel(m);
        return typename ResT::Error{"mj_makeData failed"};
    }

    if (cfg.physics_timestep) {
        m->opt.timestep = static_cast<mjtNum>(*cfg.physics_timestep);
    }

    if (cfg.initial_base_pos) {
        for (int i = 0; i < 3; ++i) {
            d->qpos[i] = static_cast<mjtNum>((*cfg.initial_base_pos)[i]);
        }
    }
    if (cfg.initial_base_quat_wxyz) {
        for (int i = 0; i < 4; ++i) {
            d->qpos[3 + i] = static_cast<mjtNum>((*cfg.initial_base_quat_wxyz)[i]);
        }
    }
    if (cfg.initial_joint_pos) {
        for (std::size_t i = 0; i < N; ++i) {
            d->qpos[7 + i] = static_cast<mjtNum>(
                (*cfg.initial_joint_pos)[static_cast<Eigen::Index>(i)]);
        }
    }

    return std::unique_ptr<IRobotBackend<N>>(
        new MujocoBackendImpl<N>{m, d, cfg.substeps, cfg.enable_viewer,
                                  cfg.spawn_viewer_thread,
                                  cfg.viewer_width, cfg.viewer_height,
                                  cfg.max_torque, cfg.kp_scale, cfg.kd_scale,
                                  cfg.ghost_motion, cfg.ghost_remap,
                                  cfg.ghost_rgba});
}

// Explicit instantiations — keep in sync with the `extern template` decls
// in mujoco_backend.hpp.
template core::Result<std::unique_ptr<IRobotBackend<29>>>
make_mujoco_backend(const MujocoBackendConfig<29>& cfg);

// Downcasts to the concrete MujocoBackendImpl (complete in this TU).
template <std::size_t N>
bool mujoco_camera_info(IRobotBackend<N>& backend, const std::string& camera_name,
                        MujocoCameraInfo& out) {
    auto* impl = dynamic_cast<MujocoBackendImpl<N>*>(&backend);
    return impl != nullptr && impl->camera_info(camera_name, out);
}

template bool mujoco_camera_info(IRobotBackend<29>&, const std::string&, MujocoCameraInfo&);

template <std::size_t N>
int mujoco_offscreen_samples(IRobotBackend<N>& backend) {
    auto* impl = dynamic_cast<MujocoBackendImpl<N>*>(&backend);
    return impl != nullptr ? impl->offscreen_samples() : -1;
}

template int mujoco_offscreen_samples(IRobotBackend<29>&);

}  // namespace rt_deploy::robot
