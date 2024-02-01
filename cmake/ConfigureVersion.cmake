# Configures project version in library headers based on project version
function(configure_library_version libname input_file output_file)
    math(EXPR TEMP_RESULT "${PROJECT_VERSION_MAJOR} * 1000 + ${PROJECT_VERSION_MINOR} * 100 + ${PROJECT_VERSION_PATCH}")
    set(${libname}_PROJECT_VERSION_NUMBER ${TEMP_RESULT})

    # Configure a header file to pass project version
    configure_file(
        "${input_file}"
        "${output_file}"
        @ONLY
    )
endfunction(configure_library_version)
