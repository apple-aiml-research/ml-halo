#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

cmake_minimum_required(VERSION 3.22)
project(rt_deploy
    VERSION 0.1.0
    DESCRIPTION "Hard real-time robot deployment pipeline (C++20, TensorRT, Unitree SDK2)"
    LANGUAGES CXX)

# ─── Policy ─────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Custom Release-RT build type: CMake doesn't populate CMAKE_<LANG>_FLAGS_RELEASE-RT,
# so any target not linking rt_deploy_rt_flags (e.g. FetchContent deps such as
# yaml-cpp) would otherwise compile with no optimization.
set(CMAKE_C_FLAGS_RELEASE-RT   "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE-RT "-O3 -DNDEBUG" CACHE STRING "" FORCE)
# Same reason for CUDA: enable_language(CUDA) only seeds the standard configs,
# so the vision kernels would otherwise be nvcc-compiled with no optimisation
# and no NDEBUG under the release-rt-orin preset. Set unconditionally — the
# variable is simply unused when CUDA is never enabled.
set(CMAKE_CUDA_FLAGS_RELEASE-RT "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELEASE-RT    "" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE-RT "" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELEASE-RT "" CACHE STRING "" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELEASE-RT "" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CheckCXXCompilerFlag)

# ─── Options ────────────────────────────────────────────────────────────────
option(RT_DEPLOY_BUILD_TESTS       "Build the unit tests in tests/ (if present)" OFF)
option(RT_DEPLOY_WITH_TRT          "Build TensorRT inference"       OFF)
option(RT_DEPLOY_WITH_UNITREE      "Build Unitree SDK2 backend"     OFF)
option(RT_DEPLOY_WITH_MUJOCO       "Build MuJoCo backend"           OFF)
option(RT_DEPLOY_WITH_ORT          "Build ONNX Runtime fallback"    OFF)
option(RT_DEPLOY_WITH_REALSENSE    "Build D435i depth sources for student_depth" OFF)
option(RT_DEPLOY_ENABLE_SANITIZERS "Enable asan/ubsan in Debug"     OFF)
option(RT_DEPLOY_ENABLE_WERROR     "Treat warnings as errors"       ON)

# ─── Warnings + flags ───────────────────────────────────────────────────────
add_library(rt_deploy_warnings INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(rt_deploy_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion
        -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
        -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion
        -Wformat=2)
    if(RT_DEPLOY_ENABLE_WERROR)
        target_compile_options(rt_deploy_warnings INTERFACE -Werror)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # On aarch64, GCC's -Wnull-dereference and -Warray-bounds fire false
    # positives inside Eigen's inlined NEON intrinsics (vld1_f32 / vst1q_f32)
    # under -O3 — the diagnostic is reported in the inlining context, i.e. in
    # our translation unit, even with Eigen marked -isystem. The volume of
    # false positives is large enough to drown out anything useful, so disable
    # them entirely rather than just downgrading.
    # -Wno-psabi silences GCC's "parameter passing ABI changed in 10.1" notes
    # that appear for any function taking a small Eigen vector by value.
    # -Wno-interference-size silences GCC's ABI-stability warning about
    # std::hardware_destructive_interference_size — used in spsc_ring.hpp purely
    # to align an internal cache-line padding member, never as part of a public
    # ABI surface.
    target_compile_options(rt_deploy_warnings INTERFACE
        -Wno-null-dereference
        -Wno-array-bounds
        -Wno-psabi
        -Wno-interference-size)
endif()

# ─── Target platform selection ──────────────────────────────────────────────
# Choose the microarch tuning flags per Jetson platform. Supported values:
#   auto  — detect from /proc/device-tree/model on native aarch64 builds;
#           fall back to generic aarch64 if unknown.
#   orin  — Cortex-A78AE (ARMv8.2-A + crypto). JetPack 5 or 6 (CUDA 11.4/12,
#           TRT 8.5/10; see README.md, "Build on the G1").
#   thor  — Neoverse-V3AE (ARMv9.2-A).         JetPack 7, CUDA 13, TRT 10.
#   none  — non-Jetson target (desktop / x86_64 dev host). Uses -march=native.
set(RT_DEPLOY_JETSON_PLATFORM "auto" CACHE STRING
    "Target Jetson platform: auto | orin | thor | none")
set_property(CACHE RT_DEPLOY_JETSON_PLATFORM PROPERTY STRINGS auto orin thor none)

set(_rt_jetson_resolved "${RT_DEPLOY_JETSON_PLATFORM}")
if(_rt_jetson_resolved STREQUAL "auto")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64"
       AND EXISTS "/proc/device-tree/model")
        file(READ "/proc/device-tree/model" _rt_jetson_model LIMIT 256)
        string(TOLOWER "${_rt_jetson_model}" _rt_jetson_model_lc)
        if(_rt_jetson_model_lc MATCHES "thor|t5000")
            set(_rt_jetson_resolved "thor")
        elseif(_rt_jetson_model_lc MATCHES "orin")
            set(_rt_jetson_resolved "orin")
        else()
            set(_rt_jetson_resolved "none")
        endif()
    else()
        set(_rt_jetson_resolved "none")
    endif()
endif()

add_library(rt_deploy_rt_flags INTERFACE)
if(CMAKE_BUILD_TYPE STREQUAL "Release-RT")
    target_compile_options(rt_deploy_rt_flags INTERFACE
        -O3 -ffast-math -fno-math-errno -fno-trapping-math -DNDEBUG)
    if(_rt_jetson_resolved STREQUAL "orin")
        # ARMv8.2-A + crypto extensions (Cortex-A78AE). The -march ISA flag is
        # supported by gcc >= 6, but -mtune=cortex-a78 needs gcc >= 11 (the
        # JetPack 6 default). On gcc-10 (JetPack 5 / L4T r35) cortex-a78 is an
        # unknown tune value, so probe it and fall back to the closest core gcc-10
        # knows. -mtune only steers instruction scheduling, never the ISA, so the
        # fallback never changes correctness — just leaves a little perf on aging
        # toolchains.
        target_compile_options(rt_deploy_rt_flags INTERFACE -march=armv8.2-a+crypto)
        check_cxx_compiler_flag("-mtune=cortex-a78" RT_DEPLOY_HAS_MTUNE_A78)
        if(RT_DEPLOY_HAS_MTUNE_A78)
            target_compile_options(rt_deploy_rt_flags INTERFACE -mtune=cortex-a78)
        else()
            target_compile_options(rt_deploy_rt_flags INTERFACE -mtune=cortex-a76)
        endif()
    elseif(_rt_jetson_resolved STREQUAL "thor")
        # ARMv9-A with SVE2 (Neoverse-V3AE). gcc 11.4 on JetPack 7 supports
        # -march=armv9-a.
        target_compile_options(rt_deploy_rt_flags INTERFACE
            -march=armv9-a+sve2)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        # Generic aarch64 when auto-detect didn't find a known Jetson.
        target_compile_options(rt_deploy_rt_flags INTERFACE -march=armv8-a)
    else()
        target_compile_options(rt_deploy_rt_flags INTERFACE -march=native)
    endif()
endif()

# NaN-safety opt-out, to be linked AFTER rt_deploy_rt_flags by any target whose
# correctness depends on NaN comparisons behaving.
#
# Release-RT's -ffast-math implies -ffinite-math-only, which licenses the
# compiler to assume no operand is ever NaN and to fold NaN-aware guards written
# as !(x > lo) / !(x < hi) — and std::isfinite() outright — into forms that are
# FALSE for NaN. The depth path depends on exactly those guards: a NaN sample
# that slips through poisons a whole 84x48 reduction cell (min-pool never
# compares true against NaN; area-pool sums it), turning a cell that contains a
# real close obstacle into the far plane. Hiding an obstacle is the one direction
# this pipeline must never fail in.
#
# This has to be an INTERFACE library rather than a target_compile_options(...
# PRIVATE) on each consumer: CMake emits a target's OWN options BEFORE those
# inherited from linked libraries, so a PRIVATE -fno-finite-math-only lands to
# the left of -ffast-math and is silently re-enabled by it. Options from linked
# INTERFACE libraries are emitted in LINK ORDER, so listing this one after
# rt_deploy_rt_flags puts the flag where it actually takes effect. Verified by
# inspecting compile_commands.json — do not "simplify" this back to PRIVATE.
add_library(rt_deploy_nan_safe INTERFACE)
if(CMAKE_BUILD_TYPE STREQUAL "Release-RT")
    target_compile_options(rt_deploy_nan_safe INTERFACE -fno-finite-math-only)
endif()

if(RT_DEPLOY_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE MATCHES "Debug")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# ─── Third-party deps ───────────────────────────────────────────────────────
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# Eigen (header-only). Prefer system install; FetchContent fallback.
find_package(Eigen3 5.0 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "[rt_deploy] Eigen3 not found system-wide; fetching 5.0.0")
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        5.0.0
        GIT_SHALLOW    TRUE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING  OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()

# Treat Eigen headers as SYSTEM includes so its internal old-style casts and
# other style violations don't trip our -Werror -Wold-style-cast -Wshadow flags.
if(TARGET Eigen3::Eigen)
    get_target_property(_eigen_aliased Eigen3::Eigen ALIASED_TARGET)
    if(_eigen_aliased)
        set(_eigen_real ${_eigen_aliased})
    else()
        set(_eigen_real Eigen3::Eigen)
    endif()
    get_target_property(_eigen_includes ${_eigen_real} INTERFACE_INCLUDE_DIRECTORIES)
    if(_eigen_includes)
        set_target_properties(${_eigen_real} PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_eigen_includes}")
    endif()
endif()

# yaml-cpp
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "[rt_deploy] yaml-cpp not found system-wide; fetching 0.8.0")
    FetchContent_Declare(yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG        0.8.0
        GIT_SHALLOW    TRUE)
    set(YAML_CPP_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB  OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(yaml-cpp)
endif()

# Older yaml-cpp packages (e.g. libyaml-cpp-dev on Ubuntu 20.04 / yaml-cpp 0.6)
# only export the unnamespaced target `yaml-cpp`. The 0.8 release added the
# modern `yaml-cpp::yaml-cpp` alias. Backfill it when missing so the rest of
# the project can use the namespaced name unconditionally.
if(NOT TARGET yaml-cpp::yaml-cpp AND TARGET yaml-cpp)
    add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
endif()

# ─── Core library ───────────────────────────────────────────────────────────
# Robot-agnostic primitives (clocks, RT thread plumbing, jitter histogram,
# loop trace, quaternion math). The G1 YAML config lives in rt_deploy::g1.
add_library(rt_deploy_core STATIC
    src/core/clock.cpp
    src/core/rt_thread.cpp
    src/core/jitter_histogram.cpp
    src/core/loop_trace.cpp
    src/math/quat.cpp)
add_library(rt_deploy::core ALIAS rt_deploy_core)

target_include_directories(rt_deploy_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(rt_deploy_core
    PUBLIC  Eigen3::Eigen
    PRIVATE yaml-cpp::yaml-cpp rt_deploy_warnings rt_deploy_rt_flags)

if(UNIX AND NOT APPLE)
    target_link_libraries(rt_deploy_core PUBLIC pthread rt)
endif()

# ─── Inference library ──────────────────────────────────────────────────────
# The interface (IInferenceEngine) is header-only; backends compile only when
# their respective deps are present. rt_deploy_inference links whatever
# backends were enabled.
add_library(rt_deploy_inference INTERFACE)
add_library(rt_deploy::inference ALIAS rt_deploy_inference)
target_include_directories(rt_deploy_inference INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(rt_deploy_inference INTERFACE rt_deploy::core)

if(RT_DEPLOY_WITH_TRT)
    find_package(CUDAToolkit REQUIRED)
    find_package(TensorRT REQUIRED)

    add_library(rt_deploy_trt_engine STATIC src/inference/trt_engine.cpp)
    target_link_libraries(rt_deploy_trt_engine
        PUBLIC  rt_deploy::core
                TensorRT::nvinfer
                TensorRT::nvonnxparser
                CUDA::cudart
        PRIVATE rt_deploy_warnings rt_deploy_rt_flags)
    # libnvinfer_plugin supplies the IPluginCreator registry that
    # initLibNvInferPlugins() populates. Required at DESERIALIZE time by any plan
    # containing a plugin layer. The student policies need none (the depth
    # student's CNN is Conv + Elu); only the LiteAnyStereo stereo net does, through
    # rt_deploy_vision. Registering here is defence in depth for future policies.
    # As for rt_deploy_vision below, the link and the call are tied to ONE condition
    # through RT_DEPLOY_HAVE_NVINFER_PLUGIN (trt_engine.cpp and onnx_to_trt.cpp
    # only call it when defined), so a missing library is a warning, not a link
    # error.
    if(TARGET TensorRT::nvinfer_plugin)
        target_link_libraries(rt_deploy_trt_engine PUBLIC TensorRT::nvinfer_plugin)
        target_compile_definitions(rt_deploy_trt_engine PRIVATE RT_DEPLOY_HAVE_NVINFER_PLUGIN=1)
    else()
        message(WARNING
            "[rt_deploy] TensorRT::nvinfer_plugin not found: building rt_deploy_trt_engine "
            "and onnx_to_trt without plugin registration. The student policy plans do not "
            "contain plugin layers, but any that does (e.g. InstanceNormalization) will "
            "fail to deserialize at runtime. Install libnvinfer-plugin-dev.")
    endif()
    target_include_directories(rt_deploy_trt_engine PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_link_libraries(rt_deploy_inference INTERFACE rt_deploy_trt_engine)

    add_executable(onnx_to_trt tools/onnx_to_trt.cpp)
    target_link_libraries(onnx_to_trt
        PRIVATE rt_deploy::core
                TensorRT::nvinfer
                TensorRT::nvonnxparser
                CUDA::cudart
                rt_deploy_warnings)
    # Needed for initLibNvInferPlugins: the parser can only fall back to a plugin
    # layer if the creators are registered. Same dependency as the loader.
    if(TARGET TensorRT::nvinfer_plugin)
        target_link_libraries(onnx_to_trt PRIVATE TensorRT::nvinfer_plugin)
        target_compile_definitions(onnx_to_trt PRIVATE RT_DEPLOY_HAVE_NVINFER_PLUGIN=1)
    endif()
endif()

if(RT_DEPLOY_WITH_ORT)
    find_package(ONNXRuntime REQUIRED)

    add_library(rt_deploy_ort_engine STATIC src/inference/ort_engine.cpp)
    target_link_libraries(rt_deploy_ort_engine
        PUBLIC  rt_deploy::core ONNXRuntime::ONNXRuntime
        PRIVATE rt_deploy_warnings rt_deploy_rt_flags)
    target_include_directories(rt_deploy_ort_engine PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_link_libraries(rt_deploy_inference INTERFACE rt_deploy_ort_engine)
endif()

# ─── Policy library ─────────────────────────────────────────────────────────
# Policy-side motion primitives:
#   - joint_remap (header-only)
#   - motion_loader (CSV loading, resampling, trimming; 29-DOF)
#   - motion_processing (plan -> MotionData for runtime-pushed plans)
# G1-specific pieces (action_decoder, observation_builder, motion_switcher)
# live in rt_deploy::g1.
add_library(rt_deploy_policy STATIC
    src/policy/motion_loader.cpp
    src/policy/motion_processing.cpp)
add_library(rt_deploy::policy ALIAS rt_deploy_policy)

target_include_directories(rt_deploy_policy PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(rt_deploy_policy
    PUBLIC  rt_deploy::core
    PRIVATE rt_deploy_warnings rt_deploy_rt_flags)

# ─── Robot library ──────────────────────────────────────────────────────────
# Backend abstractions: IRobotBackend<N>, RemoteController and SafetyMonitor
# are header-only. The MuJoCo backend is optional. The Unitree
# HG (humanoid) DDS backend is G1-specific and lives in rt_deploy_g1_unitree.
add_library(rt_deploy_robot INTERFACE)
add_library(rt_deploy::robot ALIAS rt_deploy_robot)
target_include_directories(rt_deploy_robot INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(rt_deploy_robot INTERFACE rt_deploy::core)

if(RT_DEPLOY_WITH_MUJOCO)
    find_package(mujoco REQUIRED)
    # Viewer needs GLFW for the window + GL context.
    find_package(glfw3 REQUIRED)
    add_library(rt_deploy_mujoco_backend STATIC src/robot/mujoco_backend.cpp)
    target_link_libraries(rt_deploy_mujoco_backend
        PUBLIC  rt_deploy::core rt_deploy::policy mujoco::mujoco
        PRIVATE glfw rt_deploy_warnings rt_deploy_rt_flags)
    # Headless ego-camera offscreen render uses EGL on Linux (GPU, no display
    # server). macOS keeps the hidden-GLFW path and needs no EGL.
    if(UNIX AND NOT APPLE)
        find_package(OpenGL REQUIRED COMPONENTS EGL)
        target_link_libraries(rt_deploy_mujoco_backend PRIVATE OpenGL::EGL)
    endif()
    target_include_directories(rt_deploy_mujoco_backend PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_link_libraries(rt_deploy_robot INTERFACE rt_deploy_mujoco_backend)
endif()

# ─── G1 library ─────────────────────────────────────────────────────────────
# Everything that targets the Unitree G1 specifically: the YAML config loader,
# observation builder, action decoder, state machine, control loop, motion
# switcher. Headers under include/rt_deploy/g1/, sources under src/g1/.
add_library(rt_deploy_g1 STATIC
    src/g1/config.cpp
    src/g1/control_loop.cpp
    src/g1/depth_clip_registry.cpp
    src/g1/motion_switcher.cpp
    src/g1/observation_builder.cpp
    src/g1/send_ref_motion.cpp)
add_library(rt_deploy::g1 ALIAS rt_deploy_g1)

target_include_directories(rt_deploy_g1 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(rt_deploy_g1
    PUBLIC  rt_deploy::core rt_deploy::policy rt_deploy::inference rt_deploy::robot
    # rt_deploy_nan_safe MUST follow rt_deploy_rt_flags: the observation builder's
    # image group ends with `if (!std::isfinite(v) || v <= 1e-6F) v = hi;`, the
    # LAST barrier before depth reaches the policy. -ffinite-math-only would
    # delete that isfinite() call outright.
    PRIVATE yaml-cpp::yaml-cpp rt_deploy_warnings rt_deploy_rt_flags rt_deploy_nan_safe)

# G1's Unitree DDS backend lives in its own STATIC library so that the
# UnitreeSDK2 dependency is only pulled in by binaries that actually need the
# real-robot path. The MuJoCo-only build (deploy_g1_mujoco) links rt_deploy::g1
# without ever touching the Unitree SDK.
if(RT_DEPLOY_WITH_UNITREE)
    find_package(UnitreeSDK2 REQUIRED)
    add_library(rt_deploy_g1_unitree STATIC src/g1/unitree_backend.cpp)
    target_link_libraries(rt_deploy_g1_unitree
        PUBLIC  rt_deploy::g1 UnitreeSDK2::UnitreeSDK2
        PRIVATE rt_deploy_warnings rt_deploy_rt_flags)
    target_include_directories(rt_deploy_g1_unitree PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
endif()

# Runtime-pushed motion plans (ZMQ/protobuf transport) are served by the
# tracker nodes in system/cpp, which add_subdirectory() this project, link
# rt_deploy::g1, and drive the control loop through the public TripleBuffer /
# RawMotionPlan / ActivePlanSnapshot / trim-atomic seam. This project has no
# transport dependency.

# ─── Vision library (D435i depth for the depth student) ─────────────────────
# Optional, OFF by default, so a macOS/dev configure does no CUDA probe and no
# camera SDK lookup.
#
# Two interchangeable producers publish into the SAME wait-free sink
# (core::TripleBuffer<g1::DepthFrame>) that the g1 control loop drains:
#   kRaw     — the D435i's own Z16 depth -> metres -> 84x48. RealSense only.
#   kCleaned — IR1/IR2 (emitter off) -> LiteAnyStereo (TensorRT) -> fx*B/disp.
#
# TRT is therefore deliberately NOT a hard requirement: RealSense ON / TRT OFF
# is a valid raw-only build that needs no GPU, no plan file and no CUDA
# toolkit, and simply does not compile the cleaned sources. enable_language(CUDA)
# lives inside that inner branch for the same reason — an unconditional enable
# would add an nvcc probe to every configure on every host.
#
# The kernels live in their OWN target (rt_deploy_vision_cuda) because nvcc
# rejects both the host warning flags (-Wshadow / -Wconversion /
# -Wold-style-cast) and the Release-RT arch flags (-march=... / -ffast-math)
# carried by rt_deploy_warnings + rt_deploy_rt_flags. That target must never
# link either interface library.
if(RT_DEPLOY_WITH_REALSENSE)
    find_package(RealSense2 REQUIRED)

    # Host sources always present: config parsing, the source factory, the raw
    # Z16 producer, and the librealsense capture wrapper.
    set(_rt_vision_sources
        src/vision/config.cpp
        src/vision/depth_source.cpp
        src/vision/raw_depth_source.cpp
        src/vision/realsense_stereo.cpp)

    if(RT_DEPLOY_WITH_TRT)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        find_package(TensorRT REQUIRED)

        # Default to Orin NX (sm_87 — the deployment target). Override for a
        # discrete dev box, e.g. -DRT_DEPLOY_CUDA_ARCH=120 for an RTX 5090, or
        # =110 for Thor. Accepts any CUDA_ARCHITECTURES value (a ';'-separated
        # list works too), written without the `sm_` prefix.
        set(RT_DEPLOY_CUDA_ARCH "87" CACHE STRING
            "CUDA architecture(s) for the vision kernels, sm_XX without the prefix")

        add_library(rt_deploy_vision_cuda STATIC src/vision/stereo_kernels.cu)
        add_library(rt_deploy::vision_cuda ALIAS rt_deploy_vision_cuda)
        target_include_directories(rt_deploy_vision_cuda PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
        # CUDA::cudart and nothing else — see the flag-isolation note above.
        target_link_libraries(rt_deploy_vision_cuda PUBLIC CUDA::cudart)
        set_target_properties(rt_deploy_vision_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "${RT_DEPLOY_CUDA_ARCH}"
            POSITION_INDEPENDENT_CODE ON)

        list(APPEND _rt_vision_sources
            src/vision/depth_pipeline.cpp
            src/vision/stereo_depth_engine.cpp)
    endif()

    add_library(rt_deploy_vision STATIC ${_rt_vision_sources})
    add_library(rt_deploy::vision ALIAS rt_deploy_vision)
    unset(_rt_vision_sources)

    target_include_directories(rt_deploy_vision PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

    # PUBLIC, and required: depth_source.cpp checks it to decide which sources
    # are constructible, and raw_depth_source.cpp refuses to compile without it.
    # PUBLIC also keeps the #if defined(RT_DEPLOY_WITH_REALSENSE) blocks in
    # consumers (apps/g1/depth_setup.hpp) in agreement with the library.
    target_compile_definitions(rt_deploy_vision PUBLIC RT_DEPLOY_WITH_REALSENSE=1)

    # rt_deploy::g1 is PUBLIC because g1::DepthFrame — the sink element type —
    # appears in every depth source's constructor signature. RealSense is
    # PRIVATE: librealsense never leaks past the vision headers, so binaries
    # linking rt_deploy::vision don't inherit its include dir.
    target_link_libraries(rt_deploy_vision
        PUBLIC  rt_deploy::core rt_deploy::g1
        PRIVATE RealSense2::RealSense2
                yaml-cpp::yaml-cpp
                rt_deploy_warnings
                rt_deploy_rt_flags
                # MUST come after rt_deploy_rt_flags — see the target's definition.
                rt_deploy_nan_safe)

    if(RT_DEPLOY_WITH_TRT)
        # PUBLIC so the factory's declaration in depth_source.hpp and its
        # definition agree on whether kCleaned is constructible.
        target_compile_definitions(rt_deploy_vision PUBLIC RT_DEPLOY_WITH_TRT=1)
        target_link_libraries(rt_deploy_vision
            PUBLIC  rt_deploy::inference
                    rt_deploy::vision_cuda
                    TensorRT::nvinfer
                    CUDA::cudart)

        # LiteAnyStereo's InstanceNormalization is lowered by the ONNX parser to
        # the InstanceNormalization_TRT plugin, which lives in libnvinfer_plugin
        # and must be registered (initLibNvInferPlugins) before DESERIALIZE.
        #
        # The registration call and the link are tied to ONE condition through
        # RT_DEPLOY_HAVE_NVINFER_PLUGIN: stereo_depth_engine.cpp only calls
        # initLibNvInferPlugins when it is defined. Linking conditionally while
        # calling unconditionally would turn a missing plugin library into an
        # undefined-symbol LINK error, contradicting the warning below. Without
        # the library the build succeeds and a plan that contains a plugin layer
        # fails at load with TensorRT's "IPluginCreator not found" — which is what
        # the warning says.
        if(TARGET TensorRT::nvinfer_plugin)
            target_link_libraries(rt_deploy_vision PUBLIC TensorRT::nvinfer_plugin)
            target_compile_definitions(rt_deploy_vision PRIVATE RT_DEPLOY_HAVE_NVINFER_PLUGIN=1)
        else()
            message(WARNING
                "[rt_deploy] TensorRT::nvinfer_plugin not found: building rt_deploy_vision "
                "without plugin registration. Stereo plans that contain plugin layers "
                "(e.g. InstanceNormalization in LiteAnyStereo) will build and link but fail "
                "to deserialize at runtime. Install libnvinfer-plugin-dev.")
        endif()
    endif()
endif()

# ─── Binaries ───────────────────────────────────────────────────────────────
# Each binary requires its specific backend + at least one inference engine.
# Sources live under apps/g1/ — directory placement signals robot target.

if(RT_DEPLOY_WITH_MUJOCO AND (RT_DEPLOY_WITH_TRT OR RT_DEPLOY_WITH_ORT))
    add_executable(deploy_g1_mujoco apps/g1/deploy_mujoco.cpp)
    target_include_directories(deploy_g1_mujoco PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(deploy_g1_mujoco PRIVATE rt_deploy::g1 rt_deploy_warnings)
    if(RT_DEPLOY_WITH_TRT)
        target_compile_definitions(deploy_g1_mujoco PRIVATE RT_DEPLOY_WITH_TRT=1)
    endif()
    if(RT_DEPLOY_WITH_ORT)
        target_compile_definitions(deploy_g1_mujoco PRIVATE RT_DEPLOY_WITH_ORT=1)
    endif()
endif()

if(RT_DEPLOY_WITH_UNITREE AND (RT_DEPLOY_WITH_TRT OR RT_DEPLOY_WITH_ORT))
    add_executable(deploy_g1_real apps/g1/deploy_real.cpp)
    target_include_directories(deploy_g1_real PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(deploy_g1_real PRIVATE rt_deploy_g1_unitree rt_deploy_warnings)
    if(RT_DEPLOY_WITH_TRT)
        target_compile_definitions(deploy_g1_real PRIVATE RT_DEPLOY_WITH_TRT=1)
    endif()
    if(RT_DEPLOY_WITH_ORT)
        target_compile_definitions(deploy_g1_real PRIVATE RT_DEPLOY_WITH_ORT=1)
    endif()
    # Depth is opt-in at runtime (--depth_config); linking it only adds the
    # producer threads to the binary, it never changes the default control path.
    if(RT_DEPLOY_WITH_REALSENSE)
        target_compile_definitions(deploy_g1_real PRIVATE RT_DEPLOY_WITH_REALSENSE=1)
        target_link_libraries(deploy_g1_real PRIVATE rt_deploy::vision)
    endif()
endif()

# Read-only Unitree bring-up tool. No inference, no control loop, no commands:
# subscribes to rt/lowstate and prints a live dashboard. Use before enabling
# deploy_g1_real to validate the DDS path.
if(RT_DEPLOY_WITH_UNITREE)
    add_executable(g1_state_monitor apps/g1/state_monitor.cpp)
    target_include_directories(g1_state_monitor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(g1_state_monitor
        PRIVATE rt_deploy_g1_unitree rt_deploy_warnings)
endif()

# ─── Tests ──────────────────────────────────────────────────────────────────
if(RT_DEPLOY_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST          OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)
endif()

# ─── Status line ────────────────────────────────────────────────────────────
message(STATUS "[rt_deploy] build=${CMAKE_BUILD_TYPE}  "
               "trt=${RT_DEPLOY_WITH_TRT}  ort=${RT_DEPLOY_WITH_ORT}  "
               "unitree=${RT_DEPLOY_WITH_UNITREE}  "
               "mujoco=${RT_DEPLOY_WITH_MUJOCO}  "
               "realsense=${RT_DEPLOY_WITH_REALSENSE}  "
               "jetson=${_rt_jetson_resolved}")
