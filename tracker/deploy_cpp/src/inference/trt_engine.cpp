//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// TrtEngine — TensorRT 10 implementation of IInferenceEngine.
//
// Verified against upstream TensorRT 10.16.1:
//   TensorRT/include/NvInfer.h
//   TensorRT/include/NvInferRuntime.h
//   TensorRT/include/NvInferRuntimeBase.h
//   TensorRT/samples/sampleOnnxMNIST/sampleOnnxMNIST.cpp
//
// Key TRT-10 API idioms followed here:
//   - `std::unique_ptr` with default deleter — TRT 10 removed destroy() methods;
//     all interface objects (IRuntime, ICudaEngine, IExecutionContext, etc.)
//     are deleted via the standard `delete` operator.
//   - `getNbIOTensors()` + `getIOTensorName()` for introspection (replaces
//     getNbBindings / getBindingName from TRT 8).
//   - `setInputTensorAddress(name, void const*)` / `setOutputTensorAddress(name, void*)`
//     followed by `enqueueV3(stream)` — the TRT 10 standard, not executeV2.
//
// Hot-path contract (infer_multi()):
//   memcpy  input -> pinned host staging
//   async H2D memcpy
//   enqueueV3
//   async D2H memcpy
//   stream sync
//   memcpy  pinned host staging -> output
//
// All allocations happen at construction. Failures during construction are
// surfaced through Result<>. CUDA resources are managed by local RAII guards
// in the factory so partial failure cannot leak.

#include "rt_deploy/inference/trt_engine.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#if defined(RT_DEPLOY_HAVE_NVINFER_PLUGIN)
#    include <NvInferPlugin.h>
#endif
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace rt_deploy::inference {

namespace {

// ─── Logger ─────────────────────────────────────────────────────────────────
//
// Matches the severity-prefix convention used by TensorRT's own samples
// (see TensorRT/samples/common/logging.h). Must be thread-safe per the
// ILogger contract (NvInferRuntime.h:1627) — std::fprintf is MT-safe under
// glibc, and each `log()` call writes a single atomic record.
class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress verbose-level chatter at runtime; everything warning-or-worse
        // surfaces through stderr. The build-time onnx_to_trt tool uses its
        // own logger and handles all severities.
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

// ─── CUDA RAII guards ───────────────────────────────────────────────────────
//
// Release() transfers ownership back to the caller once construction is
// confirmed successful. If any step fails before release(), the destructor
// reclaims the underlying CUDA resource — no leaks, no nested if-cleanup.
struct DeviceBufferGuard {
    void* ptr = nullptr;
    DeviceBufferGuard() noexcept = default;
    DeviceBufferGuard(const DeviceBufferGuard&)            = delete;
    DeviceBufferGuard& operator=(const DeviceBufferGuard&) = delete;
    DeviceBufferGuard(DeviceBufferGuard&& o) noexcept : ptr{o.ptr} { o.ptr = nullptr; }
    DeviceBufferGuard& operator=(DeviceBufferGuard&& o) noexcept {
        if (this != &o) {
            if (ptr != nullptr) { (void)cudaFree(ptr); }
            ptr   = o.ptr;
            o.ptr = nullptr;
        }
        return *this;
    }
    ~DeviceBufferGuard() noexcept {
        if (ptr != nullptr) {
            (void)cudaFree(ptr);
        }
    }
    void* release() noexcept {
        void* p = ptr;
        ptr     = nullptr;
        return p;
    }
};

struct HostBufferGuard {
    void* ptr    = nullptr;
    bool  pinned = true;
    HostBufferGuard() noexcept = default;
    HostBufferGuard(const HostBufferGuard&)            = delete;
    HostBufferGuard& operator=(const HostBufferGuard&) = delete;
    HostBufferGuard(HostBufferGuard&& o) noexcept : ptr{o.ptr}, pinned{o.pinned} { o.ptr = nullptr; }
    HostBufferGuard& operator=(HostBufferGuard&& o) noexcept {
        if (this != &o) {
            if (ptr != nullptr) {
                if (pinned) { (void)cudaFreeHost(ptr); } else { std::free(ptr); }  // NOLINT
            }
            ptr    = o.ptr;
            pinned = o.pinned;
            o.ptr  = nullptr;
        }
        return *this;
    }
    ~HostBufferGuard() noexcept {
        if (ptr != nullptr) {
            if (pinned) {
                (void)cudaFreeHost(ptr);
            } else {
                std::free(ptr);  // NOLINT(cppcoreguidelines-owning-memory)
            }
        }
    }
    void* release() noexcept {
        void* p = ptr;
        ptr     = nullptr;
        return p;
    }
};

struct StreamGuard {
    cudaStream_t stream = nullptr;
    ~StreamGuard() noexcept {
        if (stream != nullptr) {
            (void)cudaStreamDestroy(stream);
        }
    }
    cudaStream_t release() noexcept {
        cudaStream_t s = stream;
        stream         = nullptr;
        return s;
    }
};

// ─── Shape helpers ──────────────────────────────────────────────────────────

// TRT 10 uses Dims64 where d[i] is int64_t; the MAX_DIMS bound is 8
// (NvInferRuntimeBase.h:224). We keep our TensorShape in int32 for ONNX
// interop; any dim > INT32_MAX would be absurd for a policy tensor.
TensorShape from_trt_dims(const nvinfer1::Dims& d) {
    TensorShape s;
    if (d.nbDims < 0) {
        // "Unknown rank" sentinel from TRT — caller should treat as invalid.
        return s;
    }
    s.dims.reserve(static_cast<std::size_t>(d.nbDims));
    for (int i = 0; i < d.nbDims; ++i) {
        s.dims.push_back(static_cast<std::int32_t>(d.d[i]));
    }
    return s;
}

// Dynamic dim detection: TRT represents runtime-selectable dims as d[i] == -1.
// Fixed-shape engines built by our onnx_to_trt tool never have them.
[[nodiscard]] bool has_dynamic_dim(const TensorShape& s) noexcept {
    for (auto v : s.dims) {
        if (v < 0) {
            return true;
        }
    }
    return false;
}

// ─── Plan file loader ───────────────────────────────────────────────────────

core::Result<std::vector<std::uint8_t>> read_plan_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return typename core::Result<std::vector<std::uint8_t>>::Error{
            "failed to open plan file: " + path};
    }
    const auto size = in.tellg();
    if (size <= 0) {
        return typename core::Result<std::vector<std::uint8_t>>::Error{
            "plan file is empty: " + path};
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!in) {
        return typename core::Result<std::vector<std::uint8_t>>::Error{
            "failed to read plan file: " + path};
    }
    return bytes;
}

// ─── Engine impl ────────────────────────────────────────────────────────────

class TrtEngineImpl final : public IInferenceEngine {
public:
    TrtEngineImpl(std::unique_ptr<nvinfer1::IRuntime>          runtime,
                  std::unique_ptr<nvinfer1::ICudaEngine>       engine,
                  std::unique_ptr<nvinfer1::IExecutionContext> context,
                  cudaStream_t                                  stream,
                  std::vector<void*>                            d_inputs,
                  void*                                         d_output,
                  std::vector<float*>                           h_inputs,
                  float*                                        h_output,
                  bool                                          pinned_host,
                  std::vector<std::string>                      input_names,
                  std::vector<TensorShape>                      in_shapes,
                  std::vector<std::size_t>                      in_elems,
                  std::size_t                                   out_elems) noexcept
        : runtime_{std::move(runtime)},
          engine_{std::move(engine)},
          context_{std::move(context)},
          stream_{stream},
          d_inputs_{std::move(d_inputs)},
          d_output_{d_output},
          h_inputs_{std::move(h_inputs)},
          h_output_{h_output},
          pinned_host_{pinned_host},
          input_names_{std::move(input_names)},
          input_shapes_{std::move(in_shapes)},
          in_elems_{std::move(in_elems)},
          out_elems_{out_elems} {}

    ~TrtEngineImpl() override {
        // Destroy CUDA objects in reverse dependency order. Stream goes first
        // so any pending async ops finish before we release their buffers.
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
        for (float* h : h_inputs_) {
            if (h != nullptr) {
                if (pinned_host_) {
                    (void)cudaFreeHost(h);
                } else {
                    std::free(h);  // NOLINT(cppcoreguidelines-owning-memory)
                }
            }
        }
        if (h_output_ != nullptr) {
            if (pinned_host_) {
                (void)cudaFreeHost(h_output_);
            } else {
                std::free(h_output_);  // NOLINT(cppcoreguidelines-owning-memory)
            }
        }
        for (void* d : d_inputs_) {
            if (d != nullptr) {
                (void)cudaFree(d);
            }
        }
        if (d_output_ != nullptr) {
            (void)cudaFree(d_output_);
        }
        // context_, engine_, runtime_ destroyed by std::unique_ptr in the
        // order they're declared (context → engine → runtime), which matches
        // TRT's documented dependency chain.
    }

    // Multi-input hot path: memcpy each caller span into its pinned host
    // staging buffer, then replay the captured graph (H2D×N + enqueueV3 + D2H)
    // or run the explicit submission path. No allocation.
    [[nodiscard]] InferenceStatus
    infer_multi(std::span<const std::span<const float>> inputs,
                std::span<float>                        output) noexcept override {
        if (inputs.size() != in_elems_.size() || output.size() != out_elems_) {
            return InferenceStatus::kSizeMismatch;
        }
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].size() != in_elems_[i]) {
                return InferenceStatus::kSizeMismatch;
            }
            std::memcpy(h_inputs_[i], inputs[i].data(), in_elems_[i] * sizeof(float));
        }

        if (graph_exec_ != nullptr) {
            // Fast path: replay the captured H2D + enqueueV3 + D2H sequence.
            if (cudaGraphLaunch(graph_exec_, stream_) != cudaSuccess) {
                return InferenceStatus::kRuntimeError;
            }
            if (cudaStreamSynchronize(stream_) != cudaSuccess) {
                return InferenceStatus::kRuntimeError;
            }
        } else {
            const InferenceStatus s = infer_impl_explicit();
            if (s != InferenceStatus::kOk) {
                return s;
            }
        }

        std::memcpy(output.data(), h_output_, out_elems_ * sizeof(float));
        return InferenceStatus::kOk;
    }

    [[nodiscard]] std::string_view backend_name() const noexcept override { return "tensorrt"; }

    [[nodiscard]] std::size_t input_count() const noexcept override {
        return in_elems_.size();
    }
    [[nodiscard]] std::size_t input_size_at(std::size_t idx) const noexcept override {
        return idx < in_elems_.size() ? in_elems_[idx] : 0U;
    }
    [[nodiscard]] TensorShape input_shape_at(std::size_t idx) const override {
        return idx < input_shapes_.size() ? input_shapes_[idx] : TensorShape{};
    }
    [[nodiscard]] std::string_view input_name_at(std::size_t idx) const noexcept override {
        return idx < input_names_.size() ? std::string_view{input_names_[idx]}
                                         : std::string_view{};
    }

    // Capture the H2D×N + enqueueV3 + D2H sequence into a CUDA graph at init.
    // Returns true on success; on failure, leaves graph_/graph_exec_ null and
    // the engine continues to use the explicit submission path.
    [[nodiscard]] bool try_enable_cuda_graphs() noexcept {
        // Warm up TRT so it allocates any lazy internal scratch and JITs any
        // first-call kernels — capturing those into the graph would either
        // fail or bake one-shot work into every replay.
        for (std::size_t i = 0; i < h_inputs_.size(); ++i) {
            std::memset(h_inputs_[i], 0, in_elems_[i] * sizeof(float));
        }
        if (infer_impl_explicit() != InferenceStatus::kOk) {
            return false;
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            return false;
        }

        // Capture. cudaStreamCaptureModeRelaxed lets TRT's enqueueV3 issue
        // synchronization primitives during capture without aborting it.
        if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            return false;
        }
        const InferenceStatus s = infer_impl_explicit_capture();
        if (s != InferenceStatus::kOk) {
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
        // CUDA < 12 (e.g. 11.4 on JetPack 5) uses the 5-arg form
        // (errorNode, logBuffer, bufferSize). Passing nulls + 0 disables the
        // error-reporting outputs, matching the flags=0 behaviour above.
        if (cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0)
            != cudaSuccess) {
#endif
            (void)cudaGraphDestroy(graph_);
            graph_      = nullptr;
            graph_exec_ = nullptr;
            return false;
        }
        return true;
    }

private:
    // Explicit per-call submission path: H2D (each input), enqueueV3, D2H, sync.
    [[nodiscard]] InferenceStatus infer_impl_explicit() noexcept {
        for (std::size_t i = 0; i < d_inputs_.size(); ++i) {
            if (cudaMemcpyAsync(d_inputs_[i], h_inputs_[i], in_elems_[i] * sizeof(float),
                                cudaMemcpyHostToDevice, stream_)
                != cudaSuccess) {
                return InferenceStatus::kRuntimeError;
            }
        }
        if (!context_->enqueueV3(stream_)) {
            return InferenceStatus::kRuntimeError;
        }
        if (cudaMemcpyAsync(h_output_, d_output_, out_elems_ * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_)
            != cudaSuccess) {
            return InferenceStatus::kRuntimeError;
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            return InferenceStatus::kRuntimeError;
        }
        return InferenceStatus::kOk;
    }

    // Same as infer_impl_explicit but without the trailing stream sync, which
    // is illegal during stream capture. Used only inside the capture window.
    [[nodiscard]] InferenceStatus infer_impl_explicit_capture() noexcept {
        for (std::size_t i = 0; i < d_inputs_.size(); ++i) {
            if (cudaMemcpyAsync(d_inputs_[i], h_inputs_[i], in_elems_[i] * sizeof(float),
                                cudaMemcpyHostToDevice, stream_)
                != cudaSuccess) {
                return InferenceStatus::kRuntimeError;
            }
        }
        if (!context_->enqueueV3(stream_)) {
            return InferenceStatus::kRuntimeError;
        }
        if (cudaMemcpyAsync(h_output_, d_output_, out_elems_ * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_)
            != cudaSuccess) {
            return InferenceStatus::kRuntimeError;
        }
        return InferenceStatus::kOk;
    }

    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t                                  stream_{nullptr};
    std::vector<void*>                            d_inputs_;
    void*                                         d_output_{nullptr};
    std::vector<float*>                           h_inputs_;
    float*                                        h_output_{nullptr};
    bool                                          pinned_host_{true};
    std::vector<std::string>                      input_names_;
    std::vector<TensorShape>                      input_shapes_;
    std::vector<std::size_t>                      in_elems_;
    std::size_t                                   out_elems_{0};
    cudaGraph_t                                   graph_{nullptr};
    cudaGraphExec_t                               graph_exec_{nullptr};
};

}  // namespace

// ─── Factory ────────────────────────────────────────────────────────────────

core::Result<std::unique_ptr<IInferenceEngine>>
make_trt_engine(const TrtEngineConfig& cfg) {
    using ResT = core::Result<std::unique_ptr<IInferenceEngine>>;

    if (cudaSetDevice(cfg.device_id) != cudaSuccess) {
        return ResT::Error{"cudaSetDevice failed for device " + std::to_string(cfg.device_id)};
    }

    // Register TensorRT's built-in plugin creators BEFORE deserializing.
    //
    // A plan embeds a plugin layer whenever the ONNX parser could not map an op
    // to a native layer and fell back to one — InstanceNormalization is the
    // common case on TRT 8.x. The student policy exports contain no
    // plugin op (the depth student's CNN is Conv + Elu); only the stereo depth
    // net (LiteAnyStereo, src/vision/) needs one. This is defence in depth, so a
    // future policy with a plugin layer loads too.
    // The creators live in libnvinfer_plugin and are NOT registered
    // automatically in a process that only links libnvinfer, so deserialization
    // fails with "Cannot deserialize plugin since corresponding IPluginCreator
    // not found in Plugin Registry" — a message that points at the plan rather
    // than at the loader. Idempotent and registered once per process.
    //
    // Guarded by RT_DEPLOY_HAVE_NVINFER_PLUGIN, which CMake defines exactly when
    // it links libnvinfer_plugin: calling it without the library is a link error.
    // A build without the library says so at configure time and here, once.
#if defined(RT_DEPLOY_HAVE_NVINFER_PLUGIN)
    static const bool s_plugins_ready = initLibNvInferPlugins(&runtime_logger(), "");
    if (!s_plugins_ready) {
        std::fprintf(stderr, "[trt_engine] warning: initLibNvInferPlugins returned false; "
                              "a plan containing plugin layers will fail to deserialize\n");
    }
#else
    static const bool s_plugins_warned = [] {
        std::fprintf(stderr,
                     "[trt_engine] warning: built without libnvinfer_plugin; a plan containing "
                     "plugin layers (e.g. InstanceNormalization) will fail to deserialize\n");
        return true;
    }();
    (void)s_plugins_warned;
#endif

    auto plan_bytes = read_plan_bytes(cfg.plan_path);
    if (!plan_bytes) {
        return ResT::Error{plan_bytes.error()};
    }

    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(runtime_logger())};
    if (!runtime) {
        return ResT::Error{"nvinfer1::createInferRuntime returned null"};
    }

    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(plan_bytes->data(), plan_bytes->size())};
    if (!engine) {
        return ResT::Error{"deserializeCudaEngine failed (check TRT version vs plan)"};
    }

    // Locate the input tensors (≥1; the student policies have two or three)
    // and exactly one output, in IO-iteration order. The input order here is
    // what input_name_at(i)/input_shape_at(i)/input_size_at(i) report; the
    // control loop maps its observation groups to slots by name.
    std::vector<std::string> input_names;
    std::string              output_name;
    int                      outputs_seen = 0;
    const int nb = engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (name == nullptr) {
            continue;
        }
        const auto mode = engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_names.emplace_back(name);
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_name = name;
            ++outputs_seen;
        }
    }
    if (input_names.empty() || outputs_seen != 1) {
        return ResT::Error{"engine has " + std::to_string(input_names.size()) + " inputs and "
                            + std::to_string(outputs_seen)
                            + " outputs; rt_deploy expects ≥1 input + exactly 1 output"};
    }

    TensorShape out_shape = from_trt_dims(engine->getTensorShape(output_name.c_str()));
    if (out_shape.dims.empty() || has_dynamic_dim(out_shape)) {
        return ResT::Error{"engine output has invalid or dynamic rank; rebuild with fixed shape"};
    }
    // IO element type must be FP32: every device/host buffer is sized as
    // elems * sizeof(float) and infer_multi() memcpys/cudaMemcpyAsyncs that many
    // bytes. A kHALF (FP16) IO tensor would make enqueueV3 read/write 2-byte
    // elements against our 4-byte buffers — silent action corruption.
    if (engine->getTensorDataType(output_name.c_str()) != nvinfer1::DataType::kFLOAT) {
        return ResT::Error{"engine output is not FP32; rebuild the plan with FP32 IO"};
    }

    std::vector<TensorShape> in_shapes;
    std::vector<std::size_t> in_elems_vec;
    in_shapes.reserve(input_names.size());
    in_elems_vec.reserve(input_names.size());
    for (const auto& name : input_names) {
        TensorShape s = from_trt_dims(engine->getTensorShape(name.c_str()));
        if (s.dims.empty() || has_dynamic_dim(s)) {
            return ResT::Error{"engine input '" + name + "' has invalid or dynamic rank"};
        }
        if (engine->getTensorDataType(name.c_str()) != nvinfer1::DataType::kFLOAT) {
            return ResT::Error{"engine input '" + name + "' is not FP32; rebuild with FP32 IO"};
        }
        const std::size_t elems = s.elements();
        if (elems == 0) {
            return ResT::Error{"engine input '" + name + "' has zero elements"};
        }
        in_shapes.push_back(std::move(s));
        in_elems_vec.push_back(elems);
    }

    const std::size_t out_elems = out_shape.elements();
    if (out_elems == 0) {
        return ResT::Error{"engine has zero-element output tensor"};
    }

    std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
    if (!context) {
        return ResT::Error{"createExecutionContext failed"};
    }

    // All subsequent CUDA resources use RAII guards so any early-return cleans up.
    std::vector<DeviceBufferGuard> d_input_g(input_names.size());
    std::vector<HostBufferGuard>   h_input_g(input_names.size());
    DeviceBufferGuard              d_output_g;
    HostBufferGuard                h_output_g;
    StreamGuard                    stream_g;

    for (std::size_t i = 0; i < input_names.size(); ++i) {
        if (cudaMalloc(&d_input_g[i].ptr, in_elems_vec[i] * sizeof(float)) != cudaSuccess) {
            return ResT::Error{"cudaMalloc(d_input) failed"};
        }
        h_input_g[i].pinned = cfg.use_pinned_host;
        if (cfg.use_pinned_host) {
            if (cudaMallocHost(&h_input_g[i].ptr, in_elems_vec[i] * sizeof(float)) != cudaSuccess) {
                return ResT::Error{"cudaMallocHost(h_input) failed"};
            }
        } else {
            h_input_g[i].ptr = std::malloc(in_elems_vec[i] * sizeof(float));  // NOLINT
            if (h_input_g[i].ptr == nullptr) {
                return ResT::Error{"malloc(h_input) failed"};
            }
        }
    }
    if (cudaMalloc(&d_output_g.ptr, out_elems * sizeof(float)) != cudaSuccess) {
        return ResT::Error{"cudaMalloc(d_output) failed"};
    }
    h_output_g.pinned = cfg.use_pinned_host;
    if (cfg.use_pinned_host) {
        if (cudaMallocHost(&h_output_g.ptr, out_elems * sizeof(float)) != cudaSuccess) {
            return ResT::Error{"cudaMallocHost(h_output) failed"};
        }
    } else {
        h_output_g.ptr = std::malloc(out_elems * sizeof(float));  // NOLINT
        if (h_output_g.ptr == nullptr) {
            return ResT::Error{"malloc(h_output) failed"};
        }
    }
    if (cudaStreamCreate(&stream_g.stream) != cudaSuccess) {
        return ResT::Error{"cudaStreamCreate failed"};
    }

    // Bind each input's device buffer, then the output. setInputTensorAddress
    // (const data) exists on TRT 8.5+; the generic setTensorAddress works for
    // outputs on every enqueueV3-capable TRT (8.5 through 10).
    for (std::size_t i = 0; i < input_names.size(); ++i) {
        if (!context->setInputTensorAddress(input_names[i].c_str(), d_input_g[i].ptr)) {
            return ResT::Error{"setInputTensorAddress('" + input_names[i] + "') failed"};
        }
    }
    if (!context->setTensorAddress(output_name.c_str(), d_output_g.ptr)) {
        return ResT::Error{"setTensorAddress('" + output_name + "') failed"};
    }

    // Commit: release guards into the impl, which takes over lifetime.
    std::vector<void*>  d_inputs;
    std::vector<float*> h_inputs;
    d_inputs.reserve(input_names.size());
    h_inputs.reserve(input_names.size());
    for (std::size_t i = 0; i < input_names.size(); ++i) {
        d_inputs.push_back(d_input_g[i].release());
        h_inputs.push_back(static_cast<float*>(h_input_g[i].release()));
    }

    auto impl = std::make_unique<TrtEngineImpl>(
        std::move(runtime), std::move(engine), std::move(context),
        stream_g.release(),
        std::move(d_inputs), d_output_g.release(),
        std::move(h_inputs), static_cast<float*>(h_output_g.release()),
        cfg.use_pinned_host,
        std::move(input_names), std::move(in_shapes),
        std::move(in_elems_vec), out_elems);

    if (cfg.use_cuda_graph) {
        if (impl->try_enable_cuda_graphs()) {
            std::fprintf(stderr, "[trt_engine] CUDA Graphs enabled\n");
        } else {
            std::fprintf(stderr,
                         "[trt_engine] CUDA Graphs capture failed; using explicit launch path\n");
        }
    }

    return std::unique_ptr<IInferenceEngine>(std::move(impl));
}

}  // namespace rt_deploy::inference
