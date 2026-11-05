#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

cmake_minimum_required(VERSION 3.22)
project(halo_system LANGUAGES CXX)

# system C++ — the orchestration layer. It does NOT define libraries of
# its own beyond the node executables; it pulls in tracker (pure control) and
# halo_comm (transport) via add_subdirectory and links the tracker nodes against
# both. Dependency direction: halo_system -> {tracker, halo_comm};
# halo_comm is transport-only (no tracker dep); the rt_deploy <-> protobuf glue
# lives here, in transport/tracker_server.hpp.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sibling folders inside halo.
get_filename_component(_ovm_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(TRACKER_CPP_DIR "${_ovm_root}/tracker/deploy_cpp"
    CACHE PATH "tracker deploy_cpp directory")
set(HALO_COMM_DIR        "${_ovm_root}/system/comm"
    CACHE PATH "halo_comm directory")

# tracker FIRST (halo_comm is independent, but the nodes need rt_deploy::* and
# tracker exports its targets here). Building tracker's own apps/tests is
# harmless; disable them if you only want the libraries + nodes.
add_subdirectory("${TRACKER_CPP_DIR}" tracker_build)
add_subdirectory("${HALO_COMM_DIR}"        halo_comm_build)

# ── motiontrack_mujoco_zmq — MuJoCo sim tracker node ───────────────────────────
if(RT_DEPLOY_WITH_MUJOCO AND RT_DEPLOY_WITH_ORT)
    add_executable(motiontrack_mujoco_zmq nodes/motiontrack_mujoco_zmq.cpp)
    target_include_directories(motiontrack_mujoco_zmq PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"   # transport/tracker_server.hpp
        "${TRACKER_CPP_DIR}")          # apps/g1/common.hpp + rt_deploy/* headers
    target_link_libraries(motiontrack_mujoco_zmq PRIVATE rt_deploy::g1 halo_comm::halo_comm)
    target_compile_definitions(motiontrack_mujoco_zmq PRIVATE RT_DEPLOY_WITH_ORT=1)
endif()

# ── motiontrack_real_zmq — real-robot (Unitree DDS) tracker node ───────────────
# Needs the Unitree SDK (Linux/robot host); gated off on dev boxes without it.
if(RT_DEPLOY_WITH_UNITREE AND (RT_DEPLOY_WITH_TRT OR RT_DEPLOY_WITH_ORT))
    add_executable(motiontrack_real_zmq nodes/motiontrack_real_zmq.cpp)
    target_include_directories(motiontrack_real_zmq PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${TRACKER_CPP_DIR}")
    target_link_libraries(motiontrack_real_zmq PRIVATE rt_deploy_g1_unitree halo_comm::halo_comm)
    if(RT_DEPLOY_WITH_TRT)
        target_compile_definitions(motiontrack_real_zmq PRIVATE RT_DEPLOY_WITH_TRT=1)
    endif()
    if(RT_DEPLOY_WITH_ORT)
        target_compile_definitions(motiontrack_real_zmq PRIVATE RT_DEPLOY_WITH_ORT=1)
    endif()
    # D435i depth for the depth student (--depth_config), as in deploy_g1_real.
    if(RT_DEPLOY_WITH_REALSENSE)
        target_compile_definitions(motiontrack_real_zmq PRIVATE RT_DEPLOY_WITH_REALSENSE=1)
        target_link_libraries(motiontrack_real_zmq PRIVATE rt_deploy::vision)
    endif()
endif()
