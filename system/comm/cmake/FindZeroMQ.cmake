#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# FindZeroMQ.cmake — locate the ZeroMQ (libzmq) C library.
#
# halo_comm's Node is header-only and includes <zmq.h>, so libzmq is a PUBLIC
# dependency: consumers need both the header and the link library.
#
# Result variables:
#   ZeroMQ_FOUND          - TRUE if libzmq (header + library) was found
#   ZeroMQ_INCLUDE_DIRS   - directory containing zmq.h
#   ZeroMQ_LIBRARIES      - the libzmq link library
#   ZeroMQ_VERSION        - version parsed from zmq.h (if available)
#
# Imported target (preferred — propagates includes transitively):
#   ZeroMQ::ZeroMQ
#
# Hints honored: ZeroMQ_ROOT (CMake var or env), plus Homebrew / system prefixes.

# pkg-config is optional — used only for extra search hints when present.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_ZeroMQ QUIET libzmq)
endif()

find_path(ZeroMQ_INCLUDE_DIR
    NAMES zmq.h
    HINTS
        ${ZeroMQ_ROOT} ENV ZeroMQ_ROOT
        ${PC_ZeroMQ_INCLUDE_DIRS}
    PATHS
        /opt/homebrew/include /opt/homebrew/opt/zeromq/include
        /usr/local/include /usr/include
    PATH_SUFFIXES include)

find_library(ZeroMQ_LIBRARY
    NAMES zmq libzmq
    HINTS
        ${ZeroMQ_ROOT} ENV ZeroMQ_ROOT
        ${PC_ZeroMQ_LIBRARY_DIRS}
    PATHS
        /opt/homebrew/lib /opt/homebrew/opt/zeromq/lib
        /usr/local/lib /usr/lib
    PATH_SUFFIXES lib)

# Parse the version from zmq.h (ZMQ_VERSION_{MAJOR,MINOR,PATCH}).
if(ZeroMQ_INCLUDE_DIR AND EXISTS "${ZeroMQ_INCLUDE_DIR}/zmq.h")
    file(STRINGS "${ZeroMQ_INCLUDE_DIR}/zmq.h" _zmq_ver_lines
         REGEX "^#define[ \t]+ZMQ_VERSION_(MAJOR|MINOR|PATCH)[ \t]+[0-9]+")
    string(REGEX REPLACE ".*ZMQ_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" _zmq_major "${_zmq_ver_lines}")
    string(REGEX REPLACE ".*ZMQ_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" _zmq_minor "${_zmq_ver_lines}")
    string(REGEX REPLACE ".*ZMQ_VERSION_PATCH[ \t]+([0-9]+).*" "\\1" _zmq_patch "${_zmq_ver_lines}")
    if(_zmq_major MATCHES "^[0-9]+$")
        set(ZeroMQ_VERSION "${_zmq_major}.${_zmq_minor}.${_zmq_patch}")
    endif()
    unset(_zmq_ver_lines)
    unset(_zmq_major)
    unset(_zmq_minor)
    unset(_zmq_patch)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZeroMQ
    REQUIRED_VARS ZeroMQ_LIBRARY ZeroMQ_INCLUDE_DIR
    VERSION_VAR ZeroMQ_VERSION)

if(ZeroMQ_FOUND)
    set(ZeroMQ_LIBRARIES "${ZeroMQ_LIBRARY}")
    set(ZeroMQ_INCLUDE_DIRS "${ZeroMQ_INCLUDE_DIR}")
    if(NOT TARGET ZeroMQ::ZeroMQ)
        add_library(ZeroMQ::ZeroMQ UNKNOWN IMPORTED)
        set_target_properties(ZeroMQ::ZeroMQ PROPERTIES
            IMPORTED_LOCATION "${ZeroMQ_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ZeroMQ_INCLUDE_DIR}")
    endif()
endif()

mark_as_advanced(ZeroMQ_INCLUDE_DIR ZeroMQ_LIBRARY)
