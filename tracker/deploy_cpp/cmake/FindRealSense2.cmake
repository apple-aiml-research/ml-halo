#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# FindRealSense2.cmake — locate Intel librealsense2.
#
# librealsense ships its own realsense2Config.cmake on most installs, but the
# JetPack/apt package layout and tarball installs vary, so we follow the same
# manual find_path + find_library pattern as FindTensorRT.cmake for robustness.
# If the upstream config package is present, prefer it.
#
# Install layouts searched:
#   - apt (librealsense2-dev) on Jetson/Ubuntu:
#       /usr/include/librealsense2/rs.hpp
#       /usr/lib/aarch64-linux-gnu/librealsense2.so
#   - x86_64 dev host:
#       /usr/lib/x86_64-linux-gnu/librealsense2.so
#   - source/tarball install: /usr/local/{include,lib}
#   - custom: REALSENSE2_ROOT environment variable.
#
# Exports:
#   RealSense2::RealSense2  — imported target
#   RealSense2_FOUND, RealSense2_VERSION

# Prefer the upstream-provided config package when available.
find_package(realsense2 CONFIG QUIET)
if(realsense2_FOUND)
    if(NOT TARGET RealSense2::RealSense2)
        add_library(RealSense2::RealSense2 INTERFACE IMPORTED)
        target_link_libraries(RealSense2::RealSense2 INTERFACE realsense2::realsense2)
    endif()
    set(RealSense2_FOUND TRUE)
    set(RealSense2_VERSION "${realsense2_VERSION}")
    return()
endif()

include(FindPackageHandleStandardArgs)

find_path(RealSense2_INCLUDE_DIR
    NAMES librealsense2/rs.hpp
    HINTS
        $ENV{REALSENSE2_ROOT}/include
        /usr/include
        /usr/local/include
    DOC "librealsense2 include directory")

find_library(RealSense2_LIBRARY
    NAMES realsense2
    HINTS
        $ENV{REALSENSE2_ROOT}/lib
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib)

# Parse RS2_API_{MAJOR,MINOR,PATCH}_VERSION from rs.h when present.
if(RealSense2_INCLUDE_DIR AND EXISTS "${RealSense2_INCLUDE_DIR}/librealsense2/rs.h")
    file(READ "${RealSense2_INCLUDE_DIR}/librealsense2/rs.h" _rs_ver_text)
    string(REGEX MATCH "RS2_API_MAJOR_VERSION +([0-9]+)" _m "${_rs_ver_text}")
    set(_rs_major "${CMAKE_MATCH_1}")
    string(REGEX MATCH "RS2_API_MINOR_VERSION +([0-9]+)" _m "${_rs_ver_text}")
    set(_rs_minor "${CMAKE_MATCH_1}")
    string(REGEX MATCH "RS2_API_PATCH_VERSION +([0-9]+)" _m "${_rs_ver_text}")
    set(_rs_patch "${CMAKE_MATCH_1}")
    if(_rs_major)
        set(RealSense2_VERSION "${_rs_major}.${_rs_minor}.${_rs_patch}")
    endif()
endif()

find_package_handle_standard_args(RealSense2
    REQUIRED_VARS RealSense2_INCLUDE_DIR RealSense2_LIBRARY
    VERSION_VAR   RealSense2_VERSION)

if(RealSense2_FOUND AND NOT TARGET RealSense2::RealSense2)
    add_library(RealSense2::RealSense2 UNKNOWN IMPORTED)
    set_target_properties(RealSense2::RealSense2 PROPERTIES
        IMPORTED_LOCATION "${RealSense2_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${RealSense2_INCLUDE_DIR}")
endif()

mark_as_advanced(RealSense2_INCLUDE_DIR RealSense2_LIBRARY)
