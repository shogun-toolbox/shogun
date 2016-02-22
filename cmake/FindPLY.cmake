# Finding PLY involves calling the Python interpreter
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
        "import ply; "
        RESULT_VARIABLE PLY_SEARCH_SUCCESS
        OUTPUT_VARIABLE PLY_VALUES_OUTPUT
        ERROR_VARIABLE PLY_ERROR_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE)

if(PLY_SEARCH_SUCCESS MATCHES 0)
        SET(PLY_IMPORT_SUCCESS 1)
ENDIF()

include(FindPackageHandleStandardArgs)
if (CMAKE_VERSION LESS 2.8.3)
  find_package_handle_standard_args(PLY DEFAULT_MSG PLY_IMPORT_SUCCESS)
else ()
  find_package_handle_standard_args(PLY REQUIRED_VARS PLY_IMPORT_SUCCESS)
endif ()
