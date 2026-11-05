#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# FindUnitreeSDK2.cmake — locate the Unitree SDK2 C++ install.
#
# Upstream (unitree_sdk2) ships a proper CMake package config at:
#   ${install_prefix}/lib/cmake/unitree_sdk2/unitree_sdk2Config.cmake
# which exports the `unitree_sdk2` target with ddsc / ddscxx / Threads::Threads
# pulled in transitively via INTERFACE linkage.
#
# We prefer that package config when present and fall back to a manual lookup
# only if it's missing (e.g. a user who copied headers + a raw library).
#
# ─── On the bundled "service" hook in the prebuilt archive ─────────────────
# The prebuilt /usr/local/lib/libunitree_sdk2.a ships with
#   service_application.cpp.o  (private Unitree onboard-service entry)
#   service_main.cpp.o
# whose `__attribute__((constructor))` parses `argv` before main() runs and
# calls exit(0) when it doesn't like the shape. That ctor is harmless on
# pure library users — except the linker pulls the .o in as part of the
# static archive, and its ctor fires ahead of ours.
#
# We sidestep this by copying the archive into the build tree and running
# `ar d` to delete the two offending .o files. The result is a patched
# libunitree_sdk2_client.a that we link against instead. Upstream .a is
# left untouched.
#
# Exports a single imported target for our code to depend on:
#   UnitreeSDK2::UnitreeSDK2

include(FindPackageHandleStandardArgs)

option(RT_DEPLOY_STRIP_UNITREE_SERVICE_HOOK
    "Remove service_application.cpp.o / service_main.cpp.o from libunitree_sdk2.a.\
 Safe to leave on: ar d is a no-op if the .o's aren't in the archive."
    ON)

# Helper: copy SRC_AR into ${CMAKE_BINARY_DIR}/third_party/unitree_sdk2/, then
# run `ar d` to strip the service-hook .o files. Sets <OUT_VAR> to the patched
# path. If RT_DEPLOY_STRIP_UNITREE_SERVICE_HOOK is OFF or the copy/patch fails,
# <OUT_VAR> is set to SRC_AR (unmodified original).
function(_unitree_sdk2_strip_service_hook SRC_AR OUT_VAR)
    if(NOT RT_DEPLOY_STRIP_UNITREE_SERVICE_HOOK)
        set(${OUT_VAR} "${SRC_AR}" PARENT_SCOPE)
        return()
    endif()

    get_filename_component(_ar_name "${SRC_AR}" NAME)
    # Rename on copy to make it obvious in `ldd`-style listings and log output
    # that the binary linked against a patched archive.
    string(REPLACE ".a" "_client_patched.a" _patched_name "${_ar_name}")
    set(_out_dir "${CMAKE_BINARY_DIR}/third_party/unitree_sdk2")
    set(_patched "${_out_dir}/${_patched_name}")

    # Re-run when the upstream archive gets newer.
    set(_stamp "${_patched}.stamp")
    set(_needs_patch TRUE)
    if(EXISTS "${_patched}" AND EXISTS "${_stamp}")
        file(TIMESTAMP "${SRC_AR}"  _src_ts "%s")
        file(TIMESTAMP "${_stamp}"  _dst_ts "%s")
        if(NOT "${_src_ts}" STREQUAL "" AND NOT "${_dst_ts}" STREQUAL ""
           AND "${_dst_ts}" GREATER_EQUAL "${_src_ts}")
            set(_needs_patch FALSE)
        endif()
    endif()

    if(_needs_patch)
        file(MAKE_DIRECTORY "${_out_dir}")
        file(COPY "${SRC_AR}" DESTINATION "${_out_dir}")
        file(RENAME "${_out_dir}/${_ar_name}" "${_patched}")

        # `ar d` silently ignores .o names that aren't in the archive, so
        # passing both unconditionally is safe across SDK builds.
        execute_process(
            COMMAND "${CMAKE_AR}" d "${_patched}"
                    service_application.cpp.o service_main.cpp.o
            RESULT_VARIABLE _rc
            OUTPUT_VARIABLE _out
            ERROR_VARIABLE  _err
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE)
        if(NOT _rc EQUAL 0)
            message(WARNING
                "[UnitreeSDK2] ar d failed (rc=${_rc}): ${_err}. "
                "Falling back to the unpatched archive; the service_application "
                "hook will fire at startup and hijack argv.")
            set(${OUT_VAR} "${SRC_AR}" PARENT_SCOPE)
            return()
        endif()

        # Regenerate the archive symbol index so the linker sees fresh table.
        execute_process(
            COMMAND "${CMAKE_AR}" s "${_patched}"
            RESULT_VARIABLE _rc)
        if(NOT _rc EQUAL 0)
            message(WARNING "[UnitreeSDK2] ar s failed on patched archive")
        endif()

        file(WRITE "${_stamp}" "patched at configure time\n")
        message(STATUS
            "[UnitreeSDK2] patched archive: ${_patched}  "
            "(stripped service_application.cpp.o + service_main.cpp.o)")
    endif()

    set(${OUT_VAR} "${_patched}" PARENT_SCOPE)
endfunction()

# ─── 1) Try upstream's installed CMake package config ───────────────────────
find_package(unitree_sdk2 CONFIG QUIET
    HINTS
        $ENV{UNITREE_SDK2_ROOT}/lib/cmake/unitree_sdk2
        /opt/unitree_robotics/lib/cmake/unitree_sdk2
        /usr/local/lib/cmake/unitree_sdk2
        /opt/unitree_sdk2/lib/cmake/unitree_sdk2)

if(unitree_sdk2_FOUND AND TARGET unitree_sdk2)
    # Upstream's IMPORTED target points at the unpatched archive. Swap in our
    # patched copy so the service-hook ctor can never run.
    get_target_property(_orig_loc unitree_sdk2 IMPORTED_LOCATION)
    if(_orig_loc AND EXISTS "${_orig_loc}")
        _unitree_sdk2_strip_service_hook("${_orig_loc}" _patched_loc)
        if(NOT "${_patched_loc}" STREQUAL "${_orig_loc}")
            set_target_properties(unitree_sdk2 PROPERTIES
                IMPORTED_LOCATION "${_patched_loc}")
        endif()
    endif()

    set(UnitreeSDK2_FOUND TRUE)
    if(NOT TARGET UnitreeSDK2::UnitreeSDK2)
        add_library(UnitreeSDK2::UnitreeSDK2 INTERFACE IMPORTED)
        target_link_libraries(UnitreeSDK2::UnitreeSDK2 INTERFACE unitree_sdk2)
    endif()
    return()
endif()

# ─── 2) Manual fallback: locate headers + static lib + DDS deps ─────────────

# Probe for the dds_wrapper CRC header specifically. An SDK tree that's missing
# it (e.g. the partial copy that can sit under /usr/local/include) cannot build
# the G1 backend, so we must not select it — this is stricter than probing for
# channel_publisher.hpp, which the incomplete copy also carries.
find_path(UnitreeSDK2_INCLUDE_DIR
    NAMES unitree/dds_wrapper/common/crc.h
    HINTS
        $ENV{UNITREE_SDK2_ROOT}/include
        /opt/unitree_robotics/include
        /usr/local/unitree_sdk2/include
        /usr/local/include
        /opt/unitree_sdk2/include
        /usr/include)

find_library(UnitreeSDK2_LIBRARY
    NAMES unitree_sdk2
    HINTS
        $ENV{UNITREE_SDK2_ROOT}/lib
        $ENV{UNITREE_SDK2_ROOT}/lib/${CMAKE_SYSTEM_PROCESSOR}
        /opt/unitree_robotics/lib
        /usr/local/unitree_sdk2/lib
        /usr/local/lib
        /opt/unitree_sdk2/lib
        /usr/lib)

find_library(UnitreeSDK2_ddsc_LIBRARY   NAMES ddsc   HINTS /usr/local/lib /usr/lib)
find_library(UnitreeSDK2_ddscxx_LIBRARY NAMES ddscxx HINTS /usr/local/lib /usr/lib)
find_package(Threads REQUIRED)

find_package_handle_standard_args(UnitreeSDK2
    REQUIRED_VARS UnitreeSDK2_INCLUDE_DIR UnitreeSDK2_LIBRARY
                  UnitreeSDK2_ddsc_LIBRARY UnitreeSDK2_ddscxx_LIBRARY)

if(UnitreeSDK2_FOUND AND NOT TARGET UnitreeSDK2::UnitreeSDK2)
    _unitree_sdk2_strip_service_hook("${UnitreeSDK2_LIBRARY}" UnitreeSDK2_EFFECTIVE_LIBRARY)

    add_library(UnitreeSDK2::UnitreeSDK2 STATIC IMPORTED)
    set_target_properties(UnitreeSDK2::UnitreeSDK2 PROPERTIES
        IMPORTED_LOCATION "${UnitreeSDK2_EFFECTIVE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${UnitreeSDK2_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES
            "${UnitreeSDK2_ddsc_LIBRARY};${UnitreeSDK2_ddscxx_LIBRARY};Threads::Threads")
endif()

mark_as_advanced(
    UnitreeSDK2_INCLUDE_DIR
    UnitreeSDK2_LIBRARY
    UnitreeSDK2_ddsc_LIBRARY
    UnitreeSDK2_ddscxx_LIBRARY)
