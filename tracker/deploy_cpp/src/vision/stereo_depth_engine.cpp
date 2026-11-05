//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// StereoDepthEngine — TensorRT 10 implementation of IStereoDepthEngine.
//
// Verified against the same TRT 10 API surface as inference/trt_engine.cpp
// (getNbIOTensors / getIOTensorName / setInputTensorAddress / setTensorAddress /
// enqueueV3, std::unique_ptr default-delete). The differences from the policy
// engine are all here: two inputs, the pre/post-process kernels, and unified-
// memory-aware buffers.
//
// Buffer strategy (the Orin unified-memory point):
//   integrated GPU (Orin NX) → every IO buffer is cudaHostAllocMapped; the
//     kernels and enqueueV3 read/write the mapped device alias, so there is NO
//     cudaMemcpy across a bus. The only copies are the two small CPU memcpys
//     that move the IR bytes out of the (generic) TripleBuffer slot and the
//     float32 metric depth back into it — those are host↔host, same DRAM.
//   discrete GPU (x86 dev box) → pinned host staging + cudaMalloc device
//     buffers + async H2D/D2H, exactly like inference::TrtEngine. This is the
//     "PCIe copy" path; on Orin it is skipped entirely.
//
// Hot-path: infer() is noexcept and allocation-free. CUDA-graph capture (lazy,
// on the first infer once depth params are set) collapses the kernel + enqueue
// + copy chain into one launch, halving launch overhead and tightening jitter.

#include "rt_deploy/vision/stereo_depth_engine.hpp"

#include "rt_deploy/inference/cuda_compat.hpp"
#include "rt_deploy/vision/stereo_kernels.hpp"

#include <NvInfer.h>
#if defined(RT_DEPLOY_HAVE_NVINFER_PLUGIN)
#    include <NvInferPlugin.h>
#endif
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace rt_deploy::vision {

using inference::InferenceStatus;

namespace {

// ─── Logger (mirrors inference/trt_engine.cpp) ───────────────────────────────
class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > Severity::kINFO) {
            return;
        }
        const char* tag = nullptr;
        switch (severity) {
            case Severity::kINTERNAL_ERROR: tag = "[TRT:F]"; break;
            case Severity::kERROR:          tag = "[TRT:E]"; break;
            case Severity::kWARNING:        tag = "[TRT:W]"; break;
            case Severity::kINFO:           tag = "[TRT:I]"; break;
            case Severity::kVERBOSE:        tag = "[TRT:V]"; break;
            default:                         tag = "[TRT:?]"; break;
        }
        std::fprintf(stderr, "%s %s\n", tag, msg);
    }
};

TrtLogger& runtime_logger() noexcept {
    static TrtLogger logger;
    return logger;
}

core::Result<std::vector<std::uint8_t>> read_plan_bytes(const std::string& path) {
    using ResT = core::Result<std::vector<std::uint8_t>>;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return ResT::Error{"failed to open plan file: " + path};
    }
    const auto size = in.tellg();
    if (size <= 0) {
        return ResT::Error{"plan file is empty: " + path};
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!in) {
        return ResT::Error{"failed to read plan file: " + path};
    }
    return bytes;
}

// ─── Unified-memory-aware buffer ─────────────────────────────────────────────
//
// `host` is a CPU-writable pointer (null for device-only intermediates).
// `dev` is the address handed to kernels / TensorRT. On an integrated GPU with
// a host side, `dev` is the mapped alias of `host` (no separate allocation and
// no copy); h2d()/d2h() are then no-ops.
struct Buf {
    void*       host       = nullptr;
    void*       dev        = nullptr;
    std::size_t bytes      = 0;
    bool        integrated = false;
    bool        has_host   = false;

    cudaError_t alloc(std::size_t n, bool integrated_, bool need_host) noexcept {
        bytes      = n;
        integrated = integrated_;
        has_host   = need_host;
        if (integrated) {
            if (need_host) {
                if (const auto e = cudaHostAlloc(&host, n, cudaHostAllocMapped);
                    e != cudaSuccess) {
                    return e;
                }
                return cudaHostGetDevicePointer(&dev, host, 0);
            }
            return cudaMalloc(&dev, n);
        }
        if (need_host) {
            if (const auto e = cudaHostAlloc(&host, n, cudaHostAllocDefault);
                e != cudaSuccess) {  // pinned
                return e;
            }
        }
        return cudaMalloc(&dev, n);
    }

    void free() noexcept {
        if (host != nullptr) {
            (void)cudaFreeHost(host);
            host = nullptr;
        }
        // On integrated + has_host, `dev` aliases `host` (freed above).
        if (dev != nullptr && !(integrated && has_host)) {
            (void)cudaFree(dev);
        }
        dev = nullptr;
    }

    [[nodiscard]] cudaError_t h2d(cudaStream_t s) noexcept {
        if (integrated || !has_host) {
            return cudaSuccess;
        }
        return cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s);
    }
    [[nodiscard]] cudaError_t d2h(cudaStream_t s) noexcept {
        if (integrated || !has_host) {
            return cudaSuccess;
        }
        return cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, s);
    }
};

[[nodiscard]] bool is_float(const nvinfer1::ICudaEngine& e, const char* name) noexcept {
    return e.getTensorDataType(name) == nvinfer1::DataType::kFLOAT;
}

// Render a shape as "(1,3,240,424)" so a rejected plan says what it actually
// contains. TRT 10's Dims holds int64_t extents.
[[nodiscard]] std::string dims_to_string(const nvinfer1::Dims& d) {
    if (d.nbDims < 0) {
        return "(unknown rank)";
    }
    std::string s{"("};
    for (int i = 0; i < d.nbDims; ++i) {
        if (i > 0) {
            s += ',';
        }
        s += std::to_string(static_cast<long long>(d.d[i]));
    }
    s += ')';
    return s;
}

// Rank-4 NCHW with every extent fixed and >= 1. A dynamic axis (TRT spells it
// -1) is the dangerous case: getTensorShape() would hand us a -1 that the pad
// arithmetic below turns into a bogus byte count, and the engine would then
// read/write the wrong number of bytes on the hot path with no error anywhere.
// Load time is the only place this is cheap to catch.
[[nodiscard]] core::Result<void, std::string> check_nchw(const char* role, const std::string& name,
                                                         const nvinfer1::Dims& d) {
    using ResT = core::Result<void, std::string>;
    if (d.nbDims != 4) {
        return ResT::Error{std::string{role} + " '" + name + "' must be rank-4 NCHW, got "
                           + dims_to_string(d)};
    }
    for (int i = 0; i < 4; ++i) {
        if (d.d[i] < 1) {
            return ResT::Error{std::string{role} + " '" + name
                               + "' has a dynamic or non-positive extent " + dims_to_string(d)
                               + "; rebuild the plan with fixed shapes"};
        }
    }
    return ResT{};
}

// Log a CUDA error with its name + string. Returns true on success so it can be
// used inline in the hot path during bring-up.
[[nodiscard]] bool cuda_ok(const char* what, cudaError_t e) noexcept {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[stereo_engine] %s FAILED: %s (%s)\n", what, cudaGetErrorName(e),
                     cudaGetErrorString(e));
        return false;
    }
    return true;
}

// ─── Engine impl ─────────────────────────────────────────────────────────────
class StereoEngineImpl final : public IStereoDepthEngine {
public:
    StereoEngineImpl(std::unique_ptr<nvinfer1::IRuntime>          runtime,
                     std::unique_ptr<nvinfer1::ICudaEngine>       engine,
                     std::unique_ptr<nvinfer1::IExecutionContext> context, bool integrated,
                     int width, int height, int pad_w, int pad_h, int out_w, int out_h,
                     float fx_baseline, float clip_min, float clip_max, Reduce downsample,
                     bool use_graph) noexcept
        : runtime_{std::move(runtime)}, engine_{std::move(engine)},
          context_{std::move(context)}, integrated_{integrated}, width_{width}, height_{height},
          pad_w_{pad_w}, pad_h_{pad_h}, out_w_{out_w}, out_h_{out_h}, fx_baseline_{fx_baseline},
          clip_min_{clip_min}, clip_max_{clip_max}, downsample_{downsample},
          use_graph_{use_graph} {}

    ~StereoEngineImpl() override {
        if (graph_exec_ != nullptr) {
            (void)cudaGraphExecDestroy(graph_exec_);
        }
        if (graph_ != nullptr) {
            (void)cudaGraphDestroy(graph_);
        }
        if (stream_ != nullptr) {
            (void)cudaStreamSynchronize(stream_);
            (void)cudaStreamDestroy(stream_);
        }
        left_u8_.free();
        right_u8_.free();
        left_chw_.free();
        right_chw_.free();
        disp_.free();
        depth_f32_.free();
    }

    // Allocate stream + buffers and wire tensor addresses. On any failure the
    // dtor reclaims whatever was allocated, so partial construction can't leak.
    [[nodiscard]] core::Result<void, std::string>
    finish_init(const std::string& left_name, const std::string& right_name,
                const std::string& out_name) {
        using ResT = core::Result<void, std::string>;
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            return ResT::Error{"cudaStreamCreate failed"};
        }

        const std::size_t px = static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
        const std::size_t chw_elems =
            static_cast<std::size_t>(pad_w_) * static_cast<std::size_t>(pad_h_) * 3U;
        const std::size_t disp_elems =
            static_cast<std::size_t>(pad_w_) * static_cast<std::size_t>(pad_h_);
        const std::size_t out_elems =
            static_cast<std::size_t>(out_w_) * static_cast<std::size_t>(out_h_);

        struct Alloc {
            Buf&        b;
            std::size_t bytes;
            bool        host;
            const char* name;
        };
        const Alloc allocs[] = {
            {left_u8_, px, true, "left_u8"},
            {right_u8_, px, true, "right_u8"},
            {left_chw_, chw_elems * sizeof(float), false, "left_chw"},
            {right_chw_, chw_elems * sizeof(float), false, "right_chw"},
            {disp_, disp_elems * sizeof(float), false, "disp"},
            {depth_f32_, out_elems * sizeof(float), true, "depth_f32"},
        };
        for (const auto& a : allocs) {
            if (a.b.alloc(a.bytes, integrated_, a.host) != cudaSuccess) {
                return ResT::Error{std::string{"alloc("} + a.name + ") failed"};
            }
        }

        if (!context_->setInputTensorAddress(left_name.c_str(), left_chw_.dev)) {
            return ResT::Error{"setInputTensorAddress('" + left_name + "') failed"};
        }
        if (!context_->setInputTensorAddress(right_name.c_str(), right_chw_.dev)) {
            return ResT::Error{"setInputTensorAddress('" + right_name + "') failed"};
        }
        if (!context_->setTensorAddress(out_name.c_str(), disp_.dev)) {
            return ResT::Error{"setTensorAddress('" + out_name + "') failed"};
        }
        return ResT{};
    }

    [[nodiscard]] InferenceStatus infer(const StereoFrame& in, DepthFrame& out) noexcept override {
        if (in.width != width_ || in.height != height_) {
            return InferenceStatus::kSizeMismatch;
        }
        const std::size_t px = static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);

        // Move IR bytes into the (mapped/pinned) input staging. Host↔host.
        std::memcpy(left_u8_.host, in.left.data(), px);
        std::memcpy(right_u8_.host, in.right.data(), px);

        if (use_graph_ && graph_exec_ == nullptr) {
            // Lazy capture on first call — depth params are set by now. A failure
            // is LATCHED: capture is not something that becomes possible later
            // (an unsupported op stays unsupported), so retrying would re-run the
            // warm-up + capture attempt and spam stderr from the produce path on
            // every frame, turning a one-off degradation into a permanent cost.
            if (!try_enable_cuda_graph()) {
                use_graph_ = false;
                std::fprintf(stderr,
                             "[stereo_engine] CUDA graph capture failed; falling back to "
                             "stream-based submit for the life of this engine\n");
            }
        }

        if (graph_exec_ != nullptr) {
            if (cudaGraphLaunch(graph_exec_, stream_) != cudaSuccess
                || cudaStreamSynchronize(stream_) != cudaSuccess) {
                return InferenceStatus::kRuntimeError;
            }
        } else {
            if (const InferenceStatus s = submit_sequence(); s != InferenceStatus::kOk) {
                return s;
            }
            if (!cuda_ok("streamSynchronize", cudaStreamSynchronize(stream_))) {
                return InferenceStatus::kRuntimeError;
            }
        }

        const std::size_t out_elems =
            static_cast<std::size_t>(out_w_) * static_cast<std::size_t>(out_h_);
        std::memcpy(out.depth_m.data(), depth_f32_.host, out_elems * sizeof(float));
        out.width  = out_w_;
        out.height = out_h_;
        return InferenceStatus::kOk;
    }

    void set_output_params(float fx_baseline, float clip_min_m, float clip_max_m) noexcept override {
        fx_baseline_ = fx_baseline;
        clip_min_    = clip_min_m;
        clip_max_    = clip_max_m;
    }

    [[nodiscard]] std::int32_t pad_width() const noexcept override { return pad_w_; }
    [[nodiscard]] std::int32_t pad_height() const noexcept override { return pad_h_; }
    [[nodiscard]] bool zero_copy() const noexcept override { return integrated_; }
    [[nodiscard]] std::string_view backend_name() const noexcept override {
        return integrated_ ? "tensorrt-stereo(zero-copy)" : "tensorrt-stereo(discrete)";
    }

private:
    // The full device sequence, minus the trailing stream sync (so it is legal
    // inside stream capture). Used directly when graphs are off, and as the
    // body that gets captured.
    [[nodiscard]] InferenceStatus submit_sequence() noexcept {
        if (!cuda_ok("h2d(left)", left_u8_.h2d(stream_))
            || !cuda_ok("h2d(right)", right_u8_.h2d(stream_))) {
            return InferenceStatus::kRuntimeError;
        }
        if (!cuda_ok("ir_to_chw(left)",
                     launch_ir_to_chw(static_cast<const std::uint8_t*>(left_u8_.dev),
                                      static_cast<float*>(left_chw_.dev), width_, height_, pad_w_,
                                      pad_h_, stream_))) {
            return InferenceStatus::kRuntimeError;
        }
        if (!cuda_ok("ir_to_chw(right)",
                     launch_ir_to_chw(static_cast<const std::uint8_t*>(right_u8_.dev),
                                      static_cast<float*>(right_chw_.dev), width_, height_, pad_w_,
                                      pad_h_, stream_))) {
            return InferenceStatus::kRuntimeError;
        }
        if (!context_->enqueueV3(stream_)) {
            std::fprintf(stderr, "[stereo_engine] enqueueV3 returned false\n");
            return InferenceStatus::kRuntimeError;
        }
        if (!cuda_ok("disp_to_depth_reduce",
                     launch_disp_to_depth_reduce(
                         static_cast<const float*>(disp_.dev),
                         static_cast<float*>(depth_f32_.dev), width_, height_, pad_w_, out_w_,
                         out_h_, fx_baseline_, clip_min_, clip_max_, downsample_, stream_))) {
            return InferenceStatus::kRuntimeError;
        }
        if (!cuda_ok("d2h(depth)", depth_f32_.d2h(stream_))) {
            return InferenceStatus::kRuntimeError;
        }
        return InferenceStatus::kOk;
    }

    [[nodiscard]] bool try_enable_cuda_graph() noexcept {
        // Warm up so TRT JITs first-call kernels / allocates lazy scratch
        // outside the captured graph.
        if (submit_sequence() != InferenceStatus::kOk
            || !cuda_ok("warmup streamSynchronize", cudaStreamSynchronize(stream_))) {
            return false;
        }
        if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            return false;
        }
        if (submit_sequence() != InferenceStatus::kOk) {
            cudaGraph_t discard = nullptr;
            (void)cudaStreamEndCapture(stream_, &discard);
            if (discard != nullptr) {
                (void)cudaGraphDestroy(discard);
            }
            return false;
        }
        if (cudaStreamEndCapture(stream_, &graph_) != cudaSuccess) {
            graph_ = nullptr;
            return false;
        }
#if CUDART_VERSION >= 12000
        if (cudaGraphInstantiate(&graph_exec_, graph_, 0) != cudaSuccess) {
#else
        if (cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0) != cudaSuccess) {
#endif
            (void)cudaGraphDestroy(graph_);
            graph_      = nullptr;
            graph_exec_ = nullptr;
            return false;
        }
        std::fprintf(stderr, "[stereo_engine] CUDA graph enabled\n");
        return true;
    }

    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t                                  stream_{nullptr};

    Buf left_u8_;
    Buf right_u8_;
    Buf left_chw_;
    Buf right_chw_;
    Buf disp_;
    Buf depth_f32_;

    bool  integrated_{false};
    int   width_{0};
    int   height_{0};
    int   pad_w_{0};
    int   pad_h_{0};
    int   out_w_{0};
    int   out_h_{0};
    float fx_baseline_{0.0F};
    float clip_min_{0.2F};
    float clip_max_{2.5F};
    Reduce downsample_{Reduce::kMin};
    bool  use_graph_{true};

    cudaGraph_t     graph_{nullptr};
    cudaGraphExec_t graph_exec_{nullptr};
};

}  // namespace

// ─── Factory ──────────────────────────────────────────────────────────────────

core::Result<std::unique_ptr<IStereoDepthEngine>>
make_stereo_depth_engine(const StereoDepthEngineConfig& cfg) {
    using ResT = core::Result<std::unique_ptr<IStereoDepthEngine>>;

    if (cfg.width <= 0 || cfg.height <= 0 || cfg.width > kMaxWidth || cfg.height > kMaxHeight) {
        return ResT::Error{"invalid network resolution (must be 1..kMax)"};
    }
    if (cfg.out_width <= 0 || cfg.out_height <= 0 || cfg.out_width > kMaxOutWidth
        || cfg.out_height > kMaxOutHeight) {
        return ResT::Error{"invalid output resolution (must be 1..kMaxOut)"};
    }
    if (cudaSetDevice(cfg.device_id) != cudaSuccess) {
        return ResT::Error{"cudaSetDevice failed for device " + std::to_string(cfg.device_id)};
    }

    // Register TensorRT's built-in plugins (InstanceNormalization_TRT, etc.).
    // The model's InstanceNormalization is built as a plugin layer by the TRT
    // 8.x ONNX parser, so the plan embeds it; the runtime process must have the
    // creators in its registry or deserializeCudaEngine fails with
    // "IPluginCreator not found". Idempotent — registered once per process.
    //
    // Guarded by RT_DEPLOY_HAVE_NVINFER_PLUGIN, which CMake defines exactly when
    // it links libnvinfer_plugin: calling it without the library is a link error.
    // A build without the library says so at configure time and here, once.
#if defined(RT_DEPLOY_HAVE_NVINFER_PLUGIN)
    static const bool s_plugins_ready = initLibNvInferPlugins(&runtime_logger(), "");
    if (!s_plugins_ready) {
        std::fprintf(stderr, "[stereo_engine] warning: initLibNvInferPlugins returned false\n");
    }
#else
    static const bool s_plugins_warned = [] {
        std::fprintf(stderr,
                     "[stereo_engine] warning: built without libnvinfer_plugin; a plan containing "
                     "plugin layers (e.g. InstanceNormalization) will fail to deserialize\n");
        return true;
    }();
    (void)s_plugins_warned;
#endif

    cudaDeviceProp prop{};
    if (inference::cuda_compat::get_device_properties(prop, cfg.device_id) != cudaSuccess) {
        return ResT::Error{"cudaGetDeviceProperties failed"};
    }
    const bool integrated = prop.integrated != 0;  // Orin NX → true
    std::fprintf(stderr, "[stereo_engine] device='%s' integrated=%s (%s)\n", prop.name,
                 integrated ? "yes" : "no",
                 integrated ? "unified memory: zero-copy IO" : "discrete: pinned + async copies");

    auto plan_bytes = read_plan_bytes(cfg.plan_path);
    if (!plan_bytes) {
        return ResT::Error{plan_bytes.error()};
    }

    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(runtime_logger())};
    if (!runtime) {
        return ResT::Error{"createInferRuntime returned null"};
    }
    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(plan_bytes->data(), plan_bytes->size())};
    if (!engine) {
        return ResT::Error{"deserializeCudaEngine failed (check TRT version vs plan)"};
    }

    // Resolve the two inputs + one output. Prefer inputs named left/right; else
    // fall back to IO order (first input = left).
    std::vector<std::string> inputs;
    std::string              output;
    const int                nb = engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (name == nullptr) {
            continue;
        }
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputs.emplace_back(name);
        } else {
            output = name;
        }
    }
    if (inputs.size() != 2 || output.empty()) {
        return ResT::Error{"engine must have exactly 2 inputs + 1 output (got "
                           + std::to_string(inputs.size()) + " inputs)"};
    }
    std::string left_name  = inputs[0];
    std::string right_name = inputs[1];
    for (const auto& n : inputs) {
        if (n == "left") {
            left_name = n;
        } else if (n == "right") {
            right_name = n;
        }
    }
    if (left_name == right_name) {  // both matched the same fallback — undo
        left_name  = inputs[0];
        right_name = inputs[1];
    }

    if (!is_float(*engine, left_name.c_str()) || !is_float(*engine, right_name.c_str())
        || !is_float(*engine, output.c_str())) {
        return ResT::Error{"engine IO is not FP32; build the plan with FP32 IO "
                           "(onnx_to_trt --weakly-typed --fp16, or strongly-typed)"};
    }

    // Shapes: inputs (1,3,padH,padW); output (1,1,padH,padW). Every extent is
    // pinned here, not just the two the pad arithmetic reads: the buffers below
    // are sized from the LEFT input alone, so a plan with a dynamic batch axis or
    // a right input shaped differently from the left would be accepted and then
    // read/write the wrong number of bytes per frame. This runs once, at load.
    const nvinfer1::Dims dleft  = engine->getTensorShape(left_name.c_str());
    const nvinfer1::Dims dright = engine->getTensorShape(right_name.c_str());
    const nvinfer1::Dims dout   = engine->getTensorShape(output.c_str());
    if (auto r = check_nchw("left input", left_name, dleft); !r) {
        return ResT::Error{r.error()};
    }
    if (auto r = check_nchw("right input", right_name, dright); !r) {
        return ResT::Error{r.error()};
    }
    if (auto r = check_nchw("output", output, dout); !r) {
        return ResT::Error{r.error()};
    }
    if (dleft.d[0] != 1 || dright.d[0] != 1 || dout.d[0] != 1) {
        return ResT::Error{"engine batch must be 1: left '" + left_name + "' "
                           + dims_to_string(dleft) + ", right '" + right_name + "' "
                           + dims_to_string(dright) + ", output '" + output + "' "
                           + dims_to_string(dout)};
    }
    if (dleft.d[1] != 3) {
        return ResT::Error{"left input '" + left_name
                           + "' must have 3 channels (IR replicated to RGB), got "
                           + dims_to_string(dleft)};
    }
    for (int i = 0; i < 4; ++i) {
        if (dright.d[i] != dleft.d[i]) {
            return ResT::Error{"right input '" + right_name + "' " + dims_to_string(dright)
                               + " must have the same shape as left input '" + left_name + "' "
                               + dims_to_string(dleft)};
        }
    }
    if (dout.d[1] != 1) {
        return ResT::Error{"output '" + output + "' must have 1 channel (disparity), got "
                           + dims_to_string(dout)};
    }
    if (dout.d[2] != dleft.d[2] || dout.d[3] != dleft.d[3]) {
        return ResT::Error{"output '" + output + "' " + dims_to_string(dout)
                           + " must match the input H/W " + dims_to_string(dleft)};
    }
    const int pad_h = static_cast<int>(dleft.d[2]);
    const int pad_w = static_cast<int>(dleft.d[3]);
    if (cfg.width > pad_w || cfg.height > pad_h) {
        return ResT::Error{"configured resolution exceeds engine padded dims"};
    }

    std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
    if (!context) {
        return ResT::Error{"createExecutionContext failed"};
    }

    auto impl = std::make_unique<StereoEngineImpl>(
        std::move(runtime), std::move(engine), std::move(context), integrated, cfg.width,
        cfg.height, pad_w, pad_h, cfg.out_width, cfg.out_height, cfg.fx_baseline, cfg.clip_min_m,
        cfg.clip_max_m, cfg.downsample, cfg.use_cuda_graph);

    if (auto r = impl->finish_init(left_name, right_name, output); !r) {
        return ResT::Error{r.error()};
    }
    std::fprintf(stderr,
                 "[stereo_engine] %s  net=%dx%d  padded=%dx%d  out=%dx%d  inputs=[%s,%s]  "
                 "output=%s\n",
                 cfg.plan_path.c_str(), cfg.width, cfg.height, pad_w, pad_h, cfg.out_width,
                 cfg.out_height, left_name.c_str(), right_name.c_str(), output.c_str());

    return std::unique_ptr<IStereoDepthEngine>(std::move(impl));
}

}  // namespace rt_deploy::vision
