# Guards for the installed nvimgcodec CMake package. Run with:
#
#   cmake -DNVIMGCODEC_PACKAGE_PREFIX=<installed prefix> -P run_package_tests.cmake
#   cmake -DNVIMGCODEC_BUILD_DIR=<build dir> -P run_package_tests.cmake
#
# The first form tests an existing installed/packaged tree - e.g. after a real
# Kitmaker/conda/tarball install, run by hand; it is not exercised in CI. The
# second stages its own install first and is what both CI (via
# docker/build_helper.sh / build_helper.bat) and a local ctest run use.
#
# Each case fails if a specific guard in Config.cmake.in is dropped:
#   1. decoy      - NO_DEFAULT_PATH on the find_path() probes
#   2. cache/normal - the unset() + unset(... CACHE) pair before each probe
#   3. dispatcher - the unversioned share/cmake/nvimgcodec config

cmake_minimum_required(VERSION 3.18)

set(_self "${CMAKE_CURRENT_LIST_DIR}")
set(_consumer "${_self}/consumer")

if(NOT DEFINED NVIMGCODEC_PACKAGE_PREFIX AND NOT DEFINED NVIMGCODEC_BUILD_DIR)
    message(FATAL_ERROR "Provide -DNVIMGCODEC_PACKAGE_PREFIX=<prefix> or -DNVIMGCODEC_BUILD_DIR=<build dir>")
endif()

set(_work "${CMAKE_CURRENT_BINARY_DIR}/nvimgcodec_pkg_guard")
if(NOT CMAKE_CURRENT_BINARY_DIR OR CMAKE_CURRENT_BINARY_DIR STREQUAL "")
    set(_work "$ENV{TMPDIR}")
    if(NOT _work)
        set(_work "/tmp")
    endif()
    set(_work "${_work}/nvimgcodec_pkg_guard")
endif()
file(REMOVE_RECURSE "${_work}")
file(MAKE_DIRECTORY "${_work}")

# --- resolve the prefix under test ------------------------------------------
if(DEFINED NVIMGCODEC_PACKAGE_PREFIX)
    set(_prefix "${NVIMGCODEC_PACKAGE_PREFIX}")
else()
    set(_destdir "${_work}/destdir")
    if(WIN32)
        # DESTDIR is a Makefile/Ninja-generator convention that Windows installs
        # cannot rely on; --prefix installs directly into _destdir instead, with
        # no extra "<configured prefix>" path segment to strip afterwards.
        message(STATUS "Staging install from ${NVIMGCODEC_BUILD_DIR} into ${_destdir}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" --install "${NVIMGCODEC_BUILD_DIR}" --prefix "${_destdir}"
            RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "staged install failed:\n${_out}")
        endif()
        set(_prefix "${_destdir}")
    else()
        # DESTDIR rather than --prefix: some components install to absolute paths
        # (e.g. /etc/ld.so.conf.d), which --prefix does not redirect.
        message(STATUS "Staging install from ${NVIMGCODEC_BUILD_DIR} into ${_destdir}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env "DESTDIR=${_destdir}"
                    "${CMAKE_COMMAND}" --install "${NVIMGCODEC_BUILD_DIR}"
            RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "staged install failed:\n${_out}")
        endif()
        # The real prefix sits at <destdir>/<configured prefix>; locate it via the
        # unversioned dispatcher, which always lands at <prefix>/share/cmake/nvimgcodec.
        file(GLOB_RECURSE _disp "${_destdir}/*/share/cmake/nvimgcodec/nvimgcodecConfig.cmake")
        if(NOT _disp)
            message(FATAL_ERROR "no share/cmake/nvimgcodec dispatcher found under ${_destdir}")
        endif()
        list(GET _disp 0 _disp0)
        get_filename_component(_prefix "${_disp0}" DIRECTORY)   # .../share/cmake/nvimgcodec
        get_filename_component(_prefix "${_prefix}" DIRECTORY)  # .../share/cmake
        get_filename_component(_prefix "${_prefix}" DIRECTORY)  # .../share
        get_filename_component(_prefix "${_prefix}" DIRECTORY)  # <prefix>
    endif()
endif()
get_filename_component(_prefix "${_prefix}" REALPATH)

file(GLOB_RECURSE _configs "${_prefix}/*/nvimgcodecConfig.cmake")
if(NOT _configs)
    message(FATAL_ERROR "no nvimgcodecConfig.cmake found under ${_prefix}")
endif()
# Prefer the real (versioned, lib-side) config over the share/cmake dispatcher.
set(_real_config_dir "")
foreach(_c IN LISTS _configs)
    get_filename_component(_d "${_c}" DIRECTORY)
    if(NOT _d MATCHES "/share/cmake/")
        set(_real_config_dir "${_d}")
        break()
    endif()
endforeach()
if(NOT _real_config_dir)
    message(FATAL_ERROR "only a dispatcher config was found under ${_prefix}; expected the versioned one too")
endif()
message(STATUS "Package under test: ${_prefix}")
message(STATUS "Real config dir:    ${_real_config_dir}")

# --- fabricate a decoy install ----------------------------------------------
# Only needs to look plausible to find_path(): matching PATH_SUFFIXES plus the
# header/library names the probes search for. No build or install required.
set(_decoy "${_work}/decoy")
file(MAKE_DIRECTORY "${_decoy}/include")
file(WRITE "${_decoy}/include/nvimgcodec.h" "#error decoy header - must never be selected\n")
file(MAKE_DIRECTORY "${_decoy}/lib")
foreach(_n libnvimgcodec.so nvimgcodec.lib libnvimgcodec.dylib)
    file(WRITE "${_decoy}/lib/${_n}" "decoy\n")
endforeach()

set(_failures "")

function(run_case name)
    cmake_parse_arguments(C "" "" "ARGS" ${ARGN})
    set(_bin "${_work}/build-${name}")
    file(MAKE_DIRECTORY "${_bin}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${_consumer}" -B "${_bin}" ${C_ARGS}
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
    if(_rc EQUAL 0)
        message(STATUS "PASS  ${name}")
    else()
        message(STATUS "FAIL  ${name}")
        message(STATUS "${_out}${_err}")
        set(_failures "${_failures};${name}" PARENT_SCOPE)
    endif()
endfunction()

# 1. A decoy install on CMAKE_PREFIX_PATH must not win. find_path() searches
#    CMAKE_PREFIX_PATH *before* HINTS, so without NO_DEFAULT_PATH the decoy wins.
run_case(decoy_hijack ARGS
    "-Dnvimgcodec_DIR=${_real_config_dir}"
    "-DCMAKE_PREFIX_PATH=${_decoy}"
    "-DEXPECTED_ROOT=${_prefix}"
    "-DFORBIDDEN_ROOT=${_decoy}")

# 2a. A stale *cache* entry from an earlier find_package must not be reused.
run_case(cache_hijack ARGS
    "-Dnvimgcodec_DIR=${_real_config_dir}"
    "-D_nvimgcodec_include_dir=${_decoy}/include"
    "-D_nvimgcodec_lib_dir=${_decoy}/lib"
    "-DEXPECTED_ROOT=${_prefix}"
    "-DFORBIDDEN_ROOT=${_decoy}")

# 2b. Same, for a pre-existing *normal* variable in the caller's scope, which
#     unset(... CACHE) alone does not clear.
run_case(normal_var_hijack ARGS
    "-Dnvimgcodec_DIR=${_real_config_dir}"
    "-DPOISON_NORMAL_VAR=${_decoy}/include"
    "-DEXPECTED_ROOT=${_prefix}"
    "-DFORBIDDEN_ROOT=${_decoy}")

# 3. A bare find_package(nvimgcodec) with no nvimgcodec_DIR must resolve via the
#    unversioned share/cmake/nvimgcodec dispatcher.
run_case(bare_find_package ARGS
    "-DCMAKE_PREFIX_PATH=${_prefix}"
    "-DEXPECTED_ROOT=${_prefix}")

if(_failures)
    list(REMOVE_ITEM _failures "")
    # Leave the work dir in place on failure so the staged tree can be inspected.
    message(FATAL_ERROR "CMake package guard failures:${_failures}\nWork dir kept at: ${_work}")
endif()

# The staged install is a full copy of the package; do not leave it behind.
file(REMOVE_RECURSE "${_work}")
message(STATUS "All CMake package guards passed.")
