#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# FindTensorRT.cmake — locate TensorRT (verified against TRT 10.x layout).
#
# TensorRT does NOT ship its own find_package() config — upstream's own
# CMakeLists uses manual find_library() calls (TensorRT/CMakeLists.txt:160-167).
# We follow the same pattern.
#
# Install layouts we search:
#   - JetPack 5 / 6 on Jetson Orin (aarch64):
#     /usr/include/aarch64-linux-gnu/{NvInfer.h,NvInferRuntime.h,...}
#     /usr/lib/aarch64-linux-gnu/libnvinfer.so
#   - x86_64 systems with TRT tarball installed to /usr/local/tensorrt:
#     /usr/local/tensorrt/include/NvInfer.h
#     /usr/local/tensorrt/lib/libnvinfer.so
#   - Any custom location via TENSORRT_ROOT environment variable.
#
# Exports imported targets:
#   TensorRT::nvinfer          — core runtime
#   TensorRT::nvinfer_plugin   — built-in plugins (optional; symbol may be
#                                absent if the user only installed `nvinfer`)
#   TensorRT::nvonnxparser     — ONNX parser; transitively links nvinfer
#
# Sets variables:
#   TensorRT_FOUND
#   TensorRT_VERSION (x.y.z, parsed from NvInferVersion.h when available)

include(FindPackageHandleStandardArgs)

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        $ENV{TENSORRT_ROOT}/include
        /usr/include/aarch64-linux-gnu
        /usr/include/x86_64-linux-gnu
        /usr/include
        /usr/local/include
        /usr/local/tensorrt/include
    DOC "TensorRT include directory")

find_library(TensorRT_nvinfer_LIBRARY
    NAMES nvinfer
    HINTS
        $ENV{TENSORRT_ROOT}/lib
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
        /usr/local/tensorrt/lib
        /usr/local/lib)

find_library(TensorRT_nvinfer_plugin_LIBRARY
    NAMES nvinfer_plugin
    HINTS
        $ENV{TENSORRT_ROOT}/lib
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
        /usr/local/tensorrt/lib
        /usr/local/lib)

find_library(TensorRT_nvonnxparser_LIBRARY
    NAMES nvonnxparser
    HINTS
        $ENV{TENSORRT_ROOT}/lib
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
        /usr/local/tensorrt/lib
        /usr/local/lib)

# Parse NV_TENSORRT_MAJOR/MINOR/PATCH from NvInferVersion.h (TRT 8.5+).
if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver_text)
    string(REGEX MATCH "#define NV_TENSORRT_MAJOR +([0-9]+)" _m "${_trt_ver_text}")
    set(TensorRT_VERSION_MAJOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "#define NV_TENSORRT_MINOR +([0-9]+)" _m "${_trt_ver_text}")
    set(TensorRT_VERSION_MINOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "#define NV_TENSORRT_PATCH +([0-9]+)" _m "${_trt_ver_text}")
    set(TensorRT_VERSION_PATCH "${CMAKE_MATCH_1}")
    if(TensorRT_VERSION_MAJOR)
        set(TensorRT_VERSION
            "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
    endif()
endif()

find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR
                  TensorRT_nvinfer_LIBRARY
                  TensorRT_nvonnxparser_LIBRARY
    VERSION_VAR   TensorRT_VERSION)

if(TensorRT_FOUND)
    if(NOT TARGET TensorRT::nvinfer)
        add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
    endif()

    if(TensorRT_nvinfer_plugin_LIBRARY AND NOT TARGET TensorRT::nvinfer_plugin)
        add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvinfer_plugin_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES      "TensorRT::nvinfer")
    endif()

    if(NOT TARGET TensorRT::nvonnxparser)
        add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES      "TensorRT::nvinfer")
    endif()
endif()

mark_as_advanced(
    TensorRT_INCLUDE_DIR
    TensorRT_nvinfer_LIBRARY
    TensorRT_nvinfer_plugin_LIBRARY
    TensorRT_nvonnxparser_LIBRARY)
