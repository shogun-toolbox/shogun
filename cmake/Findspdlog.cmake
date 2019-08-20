find_package(PkgConfig)

if(NOT EXISTS "${spdlog_INCLUDE_DIR}")
    find_path(rxcpp_INCLUDE_DIR
            NAMES spdlog/spdlog.h
            DOC "spdlog library header files"
            )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(spdlog REQUIRED_VARS spdlog_INCLUDE_DIR)
