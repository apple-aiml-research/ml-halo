//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// OrtEngine — ONNX Runtime backend.
//
// Build gating: compiled only when RT_DEPLOY_WITH_ORT=ON. We keep the hot
// path allocation-free by preallocating two Ort::Value tensors bound to fixed
// host buffers at construction and passing them to Session::Run (the
// name+Value overload) on every tick. infer_multi() only memcpys into/out of those
// fixed buffers — no per-call Ort::Value construction or heap traffic. (Note:
// with the CUDA EP, ORT still performs its own internal host<->device staging
// inside Run; on the CPU EP there is no extra copy.)

#include "rt_deploy/inference/ort_engine.hpp"

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

namespace rt_deploy::inference {

namespace {

class OrtEngineImpl final : public IInferenceEngine {
public:
    OrtEngineImpl(Ort::Env                       env,
                  Ort::SessionOptions            session_opts,
                  Ort::Session                   session,
                  Ort::MemoryInfo                mem_info,
                  std::vector<std::string>       input_names,
                  std::string                    output_name,
                  std::vector<TensorShape>       in_shapes,
                  TensorShape                    out_shape,
                  std::vector<std::vector<float>> h_inputs,
                  std::vector<float>             h_output) noexcept
        : env_{std::move(env)},
          session_opts_{std::move(session_opts)},
          session_{std::move(session)},
          mem_info_{std::move(mem_info)},
          input_names_{std::move(input_names)},
          output_name_{std::move(output_name)},
          input_shapes_{std::move(in_shapes)},
          output_shape_{std::move(out_shape)},
          h_inputs_{std::move(h_inputs)},
          h_output_{std::move(h_output)},
          out_elems_{h_output_.size()} {
        // Bind each input/output to its fixed host buffer once (pre-created
        // Ort::Value passed to Session::Run on the hot path) — no per-call
        // Ort::Value construction or arena reshuffling. Build the name pointers
        // and Ort::Values HERE (in the ctor body, after the members are moved
        // into place) so the char*/float* they capture stay valid: moving a
        // std::string with SSO relocates its buffer, and computing c_str()
        // before the move would dangle.
        const std::size_t n = input_names_.size();
        in_elems_.reserve(n);
        input_values_.reserve(n);
        in_name_ptrs_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            in_elems_.push_back(h_inputs_[i].size());
            const auto shape_i64 = to_i64(input_shapes_[i].dims);
            input_values_.push_back(Ort::Value::CreateTensor<float>(
                mem_info_, h_inputs_[i].data(), h_inputs_[i].size(),
                shape_i64.data(), shape_i64.size()));
            in_name_ptrs_.push_back(input_names_[i].c_str());
        }

        const auto out_shape_i64 = to_i64(output_shape_.dims);
        output_value_ = Ort::Value::CreateTensor<float>(mem_info_,
                                                         h_output_.data(),
                                                         h_output_.size(),
                                                         out_shape_i64.data(),
                                                         out_shape_i64.size());
        out_name_c_ = output_name_.c_str();
    }

    // Multi-input hot path: memcpy each caller span into its bound host buffer,
    // run once, memcpy the output out. No allocation, no exceptions escape.
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
            std::memcpy(h_inputs_[i].data(), inputs[i].data(),
                        in_elems_[i] * sizeof(float));
        }

        try {
            Ort::RunOptions run_opts{nullptr};
            session_.Run(run_opts, in_name_ptrs_.data(), input_values_.data(),
                         in_name_ptrs_.size(), &out_name_c_, &output_value_, 1);
        } catch (const Ort::Exception&) {
            return InferenceStatus::kRuntimeError;
        } catch (...) {
            return InferenceStatus::kRuntimeError;
        }

        std::memcpy(output.data(), h_output_.data(), out_elems_ * sizeof(float));
        return InferenceStatus::kOk;
    }

    [[nodiscard]] std::string_view backend_name() const noexcept override { return "onnxruntime"; }

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

private:
    static std::vector<std::int64_t> to_i64(const std::vector<std::int32_t>& d) {
        std::vector<std::int64_t> out;
        out.reserve(d.size());
        for (auto v : d) {
            out.push_back(static_cast<std::int64_t>(v));
        }
        return out;
    }

    Ort::Env                        env_;
    Ort::SessionOptions             session_opts_;
    Ort::Session                    session_;
    Ort::MemoryInfo                 mem_info_;
    std::vector<std::string>        input_names_;
    std::string                     output_name_;
    std::vector<TensorShape>        input_shapes_;
    TensorShape                     output_shape_;
    std::vector<std::vector<float>> h_inputs_;
    std::vector<float>              h_output_;
    std::vector<std::size_t>        in_elems_;
    std::size_t                     out_elems_{0};
    std::vector<Ort::Value>         input_values_;
    Ort::Value                      output_value_{nullptr};
    std::vector<const char*>        in_name_ptrs_;
    const char*                     out_name_c_{nullptr};
};

}  // namespace

core::Result<std::unique_ptr<IInferenceEngine>>
make_ort_engine(const OrtEngineConfig& cfg) {
    using ResT = core::Result<std::unique_ptr<IInferenceEngine>>;

    try {
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "rt_deploy"};

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(cfg.intra_op_threads);
        opts.SetInterOpNumThreads(cfg.inter_op_threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.provider == OrtProvider::kCuda) {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = cfg.device_id;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
        }

        Ort::Session session{env, cfg.onnx_path.c_str(), opts};

        // Introspect IO shapes / names. One or more float32 inputs (the student
        // policies have two or three); exactly one output.
        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t n_inputs = session.GetInputCount();
        if (n_inputs < 1 || session.GetOutputCount() != 1) {
            return ResT::Error{"ORT model must have ≥1 input and exactly 1 output"};
        }

        constexpr std::int64_t kMaxDim = std::numeric_limits<std::int32_t>::max();

        std::vector<std::string>        input_names;
        std::vector<TensorShape>        in_shapes;
        std::vector<std::vector<float>> h_inputs;
        input_names.reserve(n_inputs);
        in_shapes.reserve(n_inputs);
        h_inputs.reserve(n_inputs);

        for (std::size_t idx = 0; idx < n_inputs; ++idx) {
            auto name_alloc = session.GetInputNameAllocated(idx, allocator);
            input_names.emplace_back(name_alloc.get());

            const auto type_info   = session.GetInputTypeInfo(idx);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            // Element type must be float32: the engine binds CreateTensor<float>
            // and every host buffer / memcpy assumes 4-byte float elements.
            if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                return ResT::Error{"ORT input is not float32 — rebuild the model with FP32 IO"};
            }
            const auto dims_i64 = tensor_info.GetShape();
            // Reject dynamic/symbolic (-1) dims and any that wouldn't survive
            // the int32 narrowing, so a >INT32_MAX dim can't alias a small
            // positive value and produce an undersized host buffer.
            for (auto d : dims_i64) {
                if (d <= 0 || d > kMaxDim)
                    return ResT::Error{"ORT input has a dynamic or out-of-range dim — rebuild with fixed shape"};
            }
            std::vector<std::int32_t> dims(dims_i64.begin(), dims_i64.end());
            TensorShape shape{dims};
            h_inputs.emplace_back(shape.elements());
            in_shapes.push_back(std::move(shape));
        }

        auto out_name_alloc = session.GetOutputNameAllocated(0, allocator);
        std::string output_name{out_name_alloc.get()};

        const auto out_type_info   = session.GetOutputTypeInfo(0);
        const auto out_tensor_info = out_type_info.GetTensorTypeAndShapeInfo();
        if (out_tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return ResT::Error{"ORT output is not float32 — rebuild the model with FP32 IO"};
        }
        const auto out_dims_i64 = out_tensor_info.GetShape();
        for (auto d : out_dims_i64) {
            if (d <= 0 || d > kMaxDim)
                return ResT::Error{"ORT output has a dynamic or out-of-range dim — rebuild with fixed shape"};
        }
        std::vector<std::int32_t> out_dims(out_dims_i64.begin(), out_dims_i64.end());
        TensorShape out_shape{out_dims};

        std::vector<float> h_output(out_shape.elements());

        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

        return std::unique_ptr<IInferenceEngine>(new OrtEngineImpl{
            std::move(env), std::move(opts), std::move(session), std::move(mem_info),
            std::move(input_names), std::move(output_name),
            std::move(in_shapes), std::move(out_shape),
            std::move(h_inputs), std::move(h_output)});
    } catch (const Ort::Exception& e) {
        return ResT::Error{std::string{"ORT exception: "} + e.what()};
    } catch (const std::exception& e) {
        return ResT::Error{std::string{"ORT init error: "} + e.what()};
    }
}

}  // namespace rt_deploy::inference
