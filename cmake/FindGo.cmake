find_program(GO NAMES go)

if (GO)
    execute_process(COMMAND ${GO} version ERROR_QUIET OUTPUT_VARIABLE GO_VERSION_OUT)
    string(REGEX MATCH "go[ \t]+version[ \t]+go([0-9.]+)" _go_version "${GO_VERSION_OUT}")
    SET(GO_VERSION "${CMAKE_MATCH_1}")
endif()

# handle REQUIRED and QUIET options
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Go REQUIRED_VARS GO GO_VERSION)
mark_as_advanced (GO)
