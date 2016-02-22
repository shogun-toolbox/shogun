# Finding Pyparsing involves calling the Python interpreter
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
        "import pyparsing; "
        RESULT_VARIABLE PYPARSING_SEARCH_SUCCESS
        OUTPUT_VARIABLE PYPARSING_VALUES_OUTPUT
        ERROR_VARIABLE PYPARSING_ERROR_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE)

if(PYPARSING_SEARCH_SUCCESS MATCHES 0)
        SET(PYPARSING_IMPORT_SUCCESS 1)
ENDIF()

include(FindPackageHandleStandardArgs)
if (CMAKE_VERSION LESS 2.8.3)
  find_package_handle_standard_args(PYPARSING DEFAULT_MSG PYPARSING_IMPORT_SUCCESS)
else ()
  find_package_handle_standard_args(PYPARSING REQUIRED_VARS PYPARSING_IMPORT_SUCCESS)
endif ()
