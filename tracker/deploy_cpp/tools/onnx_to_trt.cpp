//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// onnx_to_trt — offline TensorRT plan builder.
//
// Verified against upstream TensorRT 10.16.1:
//   TensorRT/include/NvInfer.h          (IBuilder, IBuilderConfig, createInferBuilder)
//   TensorRT/include/NvOnnxParser.h     (IParser, createParser, parseFromFile)
//   TensorRT/samples/sampleOnnxMNIST/sampleOnnxMNIST.cpp  (reference idioms)
//
// Key TRT 10 decisions reflected below:
//   - Default network flag is `kSTRONGLY_TYPED` — TRT 10 best practice per the
//     samples. In strongly-typed networks, precision (FP16/FP32) comes from
//     the ONNX model's own typing, so the --fp16 BuilderFlag is a no-op. Use
//     --weakly-typed + --fp16 if you need the legacy precision-control path.
//   - `platformHasFastFp16()` is deprecated (TRT 10.5); we query the CUDA
//     device compute capability directly instead.
//   - TRT 10 removed `destroy()`; all interface objects go through
//     `std::unique_ptr` with the default deleter.
//
// Usage:
//   onnx_to_trt --onnx model.onnx --plan model.plan
//               [--fp16] [--weakly-typed] [--workspace GB] [--device N]

#include <NvInfer.h>
#include <NvOnnxParser.h>
#if defined(RT_DEPLOY_HAVE_NVINFER_PLUGIN)
#    include <NvInferPlugin.h>
#endif
#include <cuda_runtime_api.h>

#include "rt_deploy/inference/cuda_compat.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>

// TensorRT capability gates (NV_TENSORRT_MAJOR/MINOR come from NvInferVersion.h,
// pulled in by NvInfer.h):
//   kSTRONGLY_TYPED — added in TRT 8.6 / 9.0. On older TRT (e.g. 8.5 on
//                     JetPack 5 / L4T r35) it doesn't exist; build a weakly-typed
//                     network and control precision via BuilderFlag::kFP16.
//   kEXPLICIT_BATCH — required by the ONNX parser on TRT 8.x (the flags==0
//                     default is implicit batch, which the parser rejects).
//                     Removed from the enum in TRT 10, where explicit batch is
//                     the only mode.
#define RT_DEPLOY_TRT_HAS_STRONGLY_TYPED \
    (NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 6))
#define RT_DEPLOY_TRT_NEEDS_EXPLICIT_BATCH (NV_TENSORRT_MAJOR < 10)

namespace {

class StdoutLogger final : public nvinfer1::ILogger {
public:
    explicit StdoutLogger(Severity min_severity = Severity::kINFO) noexcept
        : min_severity_{min_severity} {}

    void log(Severity severity, const char* msg) noexcept override {
        if (severity > min_severity_) {
            return;
        }
        const char* tag = nullptr;
        switch (severity) {
            case Severity::kINTERNAL_ERROR: tag = "[F]"; break;
            case Severity::kERROR:          tag = "[E]"; break;
            case Severity::kWARNING:        tag = "[W]"; break;
            case Severity::kINFO:           tag = "[I]"; break;
            case Severity::kVERBOSE:        tag = "[V]"; break;
            default:                         tag = "[?]"; break;
        }
        std::fprintf(stderr, "%s %s\n", tag, msg);
    }

private:
    Severity min_severity_;
};

struct Args {
    std::string onnx_path;
    std::string plan_path;
    bool        fp16           = false;
    bool        strongly_typed = true;   // TRT 10 best practice
    int         workspace_gb   = 2;
    int         device_id      = 0;
    bool        verbose        = false;
};

void print_usage() {
    std::fprintf(
        stderr,
        "usage: onnx_to_trt --onnx <path> --plan <path>\n"
        "  --onnx PATH            input ONNX model\n"
        "  --plan PATH            output serialized engine plan\n"
        "  --fp16                 enable FP16 (legacy; ignored with strongly-typed)\n"
        "  --weakly-typed         build a legacy weakly-typed network (precision\n"
        "                         controlled by BuilderFlag::kFP16)\n"
        "  --workspace N          workspace pool limit in GiB (default 2)\n"
        "  --device N             CUDA device id (default 0)\n"
        "  --verbose              log at kVERBOSE severity\n");
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string_view a{argv[i]};
        auto             need = [&](int& j) -> const char* {
            if (j + 1 >= argc) {
                std::fprintf(stderr, "missing value for %.*s\n",
                             static_cast<int>(a.size()), a.data());
                return nullptr;
            }
            return argv[++j];
        };
        if (a == "--onnx") {
            const char* v = need(i); if (v == nullptr) return false; args.onnx_path = v;
        } else if (a == "--plan") {
            const char* v = need(i); if (v == nullptr) return false; args.plan_path = v;
        } else if (a == "--fp16") {
            args.fp16 = true;
        } else if (a == "--weakly-typed") {
            args.strongly_typed = false;
        } else if (a == "--workspace") {
            const char* v = need(i); if (v == nullptr) return false;
            args.workspace_gb = std::atoi(v);
        } else if (a == "--device") {
            const char* v = need(i); if (v == nullptr) return false;
            args.device_id = std::atoi(v);
        } else if (a == "--verbose") {
            args.verbose = true;
        } else if (a == "-h" || a == "--help") {
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %.*s\n",
                         static_cast<int>(a.size()), a.data());
            return false;
        }
    }
    if (args.onnx_path.empty() || args.plan_path.empty()) {
        std::fprintf(stderr, "error: --onnx and --plan are required\n");
        return false;
    }
    return true;
}

// Query the CUDA device's compute capability to decide whether FP16 is likely
// to be fast. Replaces the deprecated IBuilder::platformHasFastFp16 (TRT 10.5).
// The CUDA-version-guarded property lookup lives in cuda_compat.hpp so the
// same code compiles on CUDA 12 (Orin) and CUDA 13 (Thor).
[[nodiscard]] bool cuda_has_fast_fp16(int device_id) noexcept {
    return rt_deploy::inference::cuda_compat::device_has_fast_fp16(device_id);
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage();
        return 2;
    }

    std::fprintf(stderr, "[onnx_to_trt] TensorRT %d.%d.%d, ONNX parser\n",
                 NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

    if (cudaSetDevice(args.device_id) != cudaSuccess) {
        std::fprintf(stderr, "[F] cudaSetDevice(%d) failed\n", args.device_id);
        return 1;
    }

    StdoutLogger logger{args.verbose ? nvinfer1::ILogger::Severity::kVERBOSE
                                      : nvinfer1::ILogger::Severity::kINFO};

    // Register the built-in plugin creators before parsing. The ONNX parser falls
    // back to a plugin layer for ops it cannot map natively (InstanceNormalization
    // on TRT 8.x), and it can only do so if the creators are in the registry. The
    // LOADER needs them too — see the matching call in src/inference/trt_engine.cpp.
#if defined(RT_DEPLOY_HAVE_NVINFER_PLUGIN)
    if (!initLibNvInferPlugins(&logger, "")) {
        std::fprintf(stderr, "[onnx_to_trt] warning: initLibNvInferPlugins returned false\n");
    }
#else
    std::fprintf(stderr,
                 "[onnx_to_trt] warning: built without libnvinfer_plugin; ops that need a "
                 "plugin layer (e.g. InstanceNormalization) will fail to parse\n");
#endif

    std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    if (!builder) {
        std::fprintf(stderr, "[F] createInferBuilder returned null\n");
        return 1;
    }

    // TRT 10: strongly-typed networks are the recommended default; the ONNX
    // model carries its own type info and precision control comes from there.
    // We still honour --weakly-typed for users with legacy control flows.
    // On TRT 8.x the ONNX parser needs an explicit-batch network and there is
    // no strongly-typed mode, so we add kEXPLICIT_BATCH and fall back to the
    // weakly-typed (--fp16) path.
    nvinfer1::NetworkDefinitionCreationFlags network_flags = 0;
#if RT_DEPLOY_TRT_NEEDS_EXPLICIT_BATCH
    network_flags
        |= 1U << static_cast<std::uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
#if RT_DEPLOY_TRT_HAS_STRONGLY_TYPED
    if (args.strongly_typed) {
        network_flags
            |= 1U << static_cast<std::uint32_t>(
                    nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    }
#else
    if (args.strongly_typed) {
        std::fprintf(stderr,
                     "[W] strongly-typed networks need TensorRT >= 8.6; this build "
                     "is %d.%d — building a weakly-typed network. Pass --fp16 to "
                     "enable FP16 precision.\n",
                     NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
        args.strongly_typed = false;  // engages the BuilderFlag::kFP16 path below
    }
#endif
    std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(network_flags)};
    if (!network) {
        std::fprintf(stderr, "[F] createNetworkV2 failed\n");
        return 1;
    }

    std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
    if (!parser) {
        std::fprintf(stderr, "[F] nvonnxparser::createParser returned null\n");
        return 1;
    }

    const int parser_verbosity = static_cast<int>(
        args.verbose ? nvinfer1::ILogger::Severity::kVERBOSE
                      : nvinfer1::ILogger::Severity::kWARNING);
    if (!parser->parseFromFile(args.onnx_path.c_str(), parser_verbosity)) {
        std::fprintf(stderr, "[F] parseFromFile failed: %s\n", args.onnx_path.c_str());
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            const auto* err = parser->getError(i);
            if (err != nullptr) {
                std::fprintf(stderr, "  parser[%d]: %s\n", i, err->desc());
            }
        }
        return 1;
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config) {
        std::fprintf(stderr, "[F] createBuilderConfig failed\n");
        return 1;
    }

    const std::size_t ws_bytes = static_cast<std::size_t>(args.workspace_gb) * (1ULL << 30);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, ws_bytes);
    std::fprintf(stderr, "[I] workspace limit: %d GiB\n", args.workspace_gb);

    if (args.fp16) {
        if (args.strongly_typed) {
            std::fprintf(stderr,
                         "[W] --fp16 is a no-op with a strongly-typed network; "
                         "precision comes from the ONNX model. Use --weakly-typed "
                         "to regain legacy BuilderFlag::kFP16 control.\n");
        } else if (cuda_has_fast_fp16(args.device_id)) {
            // BuilderFlag::kFP16 is deprecated in TRT 10 in favor of strongly-
            // typed networks, but it's still the only knob in the weakly-typed
            // legacy path that this branch handles.
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic pop
#endif
            std::fprintf(stderr, "[I] BuilderFlag::kFP16 enabled\n");
        } else {
            std::fprintf(stderr,
                         "[W] --fp16 requested but device %d does not report fast FP16\n",
                         args.device_id);
        }
    }

    std::unique_ptr<nvinfer1::IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        std::fprintf(stderr, "[F] buildSerializedNetwork failed\n");
        return 1;
    }

    std::ofstream out(args.plan_path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "[F] failed to open plan output: %s\n", args.plan_path.c_str());
        return 1;
    }
    out.write(static_cast<const char*>(plan->data()),
               static_cast<std::streamsize>(plan->size()));
    if (!out) {
        std::fprintf(stderr, "[F] failed to write plan\n");
        return 1;
    }

    std::fprintf(stderr, "[I] wrote %zu bytes to %s\n",
                 static_cast<std::size_t>(plan->size()), args.plan_path.c_str());
    return 0;
}
