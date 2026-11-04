# Consumer used by run_package_tests.cmake to assert what find_package(nvimgcodec)
# resolved to. Configure-only: the guards under test are all directory discovery,
# so nothing needs to be compiled or linked.
cmake_minimum_required(VERSION 3.18)
project(nvimgcodec_package_guard_consumer LANGUAGES NONE)

if(NOT DEFINED EXPECTED_ROOT)
    message(FATAL_ERROR "EXPECTED_ROOT must be provided")
endif()

# Poison a *normal* variable of the same name the config uses internally. find_path()
# short-circuits on a pre-existing normal variable, which unset(... CACHE) does not
# clear - only the plain unset() in the config does.
if(POISON_NORMAL_VAR)
    set(_nvimgcodec_include_dir "${POISON_NORMAL_VAR}")
    set(_nvimgcodec_lib_dir "${POISON_NORMAL_VAR}")
endif()

find_package(nvimgcodec CONFIG REQUIRED)

function(_assert_under label value root)
    if(NOT value)
        message(FATAL_ERROR "GUARD FAIL [${label}]: resolved to an empty path")
    endif()
    get_filename_component(_v "${value}" REALPATH)
    get_filename_component(_r "${root}" REALPATH)
    # Relative-path containment, not a string prefix: "/tmp/root2" has "/tmp/root"
    # as a string prefix but is not under it.
    file(RELATIVE_PATH _rel "${_r}" "${_v}")
    if(IS_ABSOLUTE "${_rel}" OR _rel MATCHES "^\\.\\./" OR _rel STREQUAL "..")
        message(FATAL_ERROR
            "GUARD FAIL [${label}]: resolved to\n    ${_v}\n  which is not under the expected root\n    ${_r}")
    endif()
endfunction()

_assert_under("include" "${nvimgcodec_INCLUDE_DIR}" "${EXPECTED_ROOT}")
_assert_under("lib" "${nvimgcodec_LIB_DIR}" "${EXPECTED_ROOT}")

if(DEFINED FORBIDDEN_ROOT AND FORBIDDEN_ROOT)
    get_filename_component(_forbidden "${FORBIDDEN_ROOT}" REALPATH)
    foreach(_pair "include;${nvimgcodec_INCLUDE_DIR}" "lib;${nvimgcodec_LIB_DIR}")
        list(GET _pair 0 _label)
        list(GET _pair 1 _val)
        get_filename_component(_val "${_val}" REALPATH)
        file(RELATIVE_PATH _rel "${_forbidden}" "${_val}")
        if(NOT IS_ABSOLUTE "${_rel}" AND NOT _rel MATCHES "^\\.\\./" AND NOT _rel STREQUAL "..")
            message(FATAL_ERROR
                "GUARD FAIL [${_label}]: hijacked by the decoy install at\n    ${_forbidden}")
        endif()
    endforeach()
endif()

if(NOT TARGET nvimgcodec::nvimgcodec)
    message(FATAL_ERROR "GUARD FAIL: imported target nvimgcodec::nvimgcodec was not defined")
endif()

message(STATUS "GUARD OK: include=${nvimgcodec_INCLUDE_DIR}")
message(STATUS "GUARD OK: lib=${nvimgcodec_LIB_DIR}")
