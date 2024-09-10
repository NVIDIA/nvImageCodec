if(BUILD_PYTHON)
    add_subdirectory(pybind11 EXCLUDE_FROM_ALL)
    add_subdirectory(dlpack EXCLUDE_FROM_ALL)
endif(BUILD_PYTHON)
