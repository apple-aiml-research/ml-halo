#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# FindONNXRuntime.cmake — locate ONNX Runtime 1.15+ (CPU or GPU build).
#
# Exports target: ONNXRuntime::ONNXRuntime
#
# Search order, in order of preference:
#   1. -DONNXRUNTIME_ROOT=<path>   CMake variable
#   2. $ONNXRUNTIME_ROOT           environment variable
#   3. system include / lib paths (/usr, /usr/local, /opt/onnxruntime)
#
# Air-gapped builds can stage the SDK under deploy_cpp/third_party/onnxruntime/ and pass
# `-DONNXRUNTIME_ROOT=$PWD/third_party/onnxruntime` at configure time.

include(FindPackageHandleStandardArgs)

set(_ort_hint_paths "")
if(ONNXRUNTIME_ROOT)
    list(APPEND _ort_hint_paths "${ONNXRUNTIME_ROOT}/include"
                                  "${ONNXRUNTIME_ROOT}/lib")
endif()
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    list(APPEND _ort_hint_paths "$ENV{ONNXRUNTIME_ROOT}/include"
                                  "$ENV{ONNXRUNTIME_ROOT}/lib")
endif()

find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ${_ort_hint_paths}
        /usr/include/onnxruntime
        /usr/local/include/onnxruntime
        /opt/onnxruntime/include
    PATH_SUFFIXES onnxruntime core/session)

find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    HINTS
        ${_ort_hint_paths}
        /usr/lib
        /usr/local/lib
        /opt/onnxruntime/lib)

find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR)

if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
    add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
endif()

mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
