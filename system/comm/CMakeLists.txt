#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

cmake_minimum_required(VERSION 3.22)
project(halo_comm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Custom Find modules live in cmake/ (e.g. FindZeroMQ.cmake).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ── protobuf ──────────────────────────────────────────────────────────────────
# Prefer the vendor CONFIG package (Homebrew / source installs ship
# protobuf-config.cmake + protobuf_generate). System/apt protobuf — common on
# Jetson / Ubuntu — usually ships NO config package, so fall back to CMake's
# bundled module-mode FindProtobuf, which (>= CMake 3.13; we require 3.22) also
# provides protobuf::libprotobuf, protobuf::protoc, and protobuf_generate().
# Either way protoc and libprotobuf come from the same install, so the gencode
# matches the runtime.
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
    message(STATUS "protobuf: no CONFIG package; using module-mode FindProtobuf "
                   "(apt: install protobuf-compiler + libprotobuf-dev)")
    find_package(Protobuf MODULE REQUIRED)
endif()

# ── ZeroMQ (libzmq C API; header-only Node includes <zmq.h> so this is PUBLIC) ─
find_package(ZeroMQ REQUIRED)

# ── generate C++ message sources from msgs/*.proto ───────────────────────────
# Invoke protoc explicitly (CMake 3.22's module-mode protobuf_generate cannot set the output dir we need).
file(GLOB HALO_COMM_PROTOS "${CMAKE_CURRENT_SOURCE_DIR}/msgs/*.proto")
set(_halo_comm_gen "${CMAKE_CURRENT_BINARY_DIR}/gen")
file(MAKE_DIRECTORY "${_halo_comm_gen}")        # protoc won't create its output dir
set(_halo_comm_pb_srcs "")
foreach(_proto ${HALO_COMM_PROTOS})
    get_filename_component(_pname "${_proto}" NAME_WE)
    add_custom_command(
        OUTPUT  "${_halo_comm_gen}/${_pname}.pb.cc" "${_halo_comm_gen}/${_pname}.pb.h"
        COMMAND protobuf::protoc
                -I "${CMAKE_CURRENT_SOURCE_DIR}/msgs"
                --cpp_out "${_halo_comm_gen}" "${_proto}"
        DEPENDS "${_proto}" protobuf::protoc
        COMMENT "protoc cpp ${_pname}.proto"
        VERBATIM)
    list(APPEND _halo_comm_pb_srcs "${_halo_comm_gen}/${_pname}.pb.cc")
endforeach()
add_library(halo_comm STATIC ${_halo_comm_pb_srcs})
add_library(halo_comm::halo_comm ALIAS halo_comm)

target_include_directories(halo_comm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   # halo_comm/node.hpp
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/gen>)      # state.pb.h, motion.pb.h
# ZeroMQ::ZeroMQ propagates zmq.h's include dir to consumers (node.hpp needs it).
target_link_libraries(halo_comm PUBLIC protobuf::libprotobuf ZeroMQ::ZeroMQ)

# NOTE: tracker is NOT required to build the pure middleware (it's transport-
# only). The halo_system tracker NODE that wires tracker's control loop to
# these services links rt_deploy::g1 separately (see system/cpp).
