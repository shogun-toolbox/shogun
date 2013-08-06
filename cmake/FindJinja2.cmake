# Finding Jinja2 involves calling the Python interpreter
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
	"from jinja2 import Template; "
	RESULT_VARIABLE JINJA2_SEARCH_SUCCESS
	OUTPUT_VARIABLE JINJA2_VALUES_OUTPUT
	ERROR_VARIABLE JINJA2_ERROR_VALUE
	OUTPUT_STRIP_TRAILING_WHITESPACE)

if(JINJA2_SEARCH_SUCCESS MATCHES 0)
	MESSAGE(${JINJA2_SEARCH_SUCCESS} found)
	SET(JINJA2_IMPORT_SUCCESS 1)
ENDIF()

include(FindPackageHandleStandardArgs)
if (CMAKE_VERSION LESS 2.8.3)
  find_package_handle_standard_args(Jinja2 DEFAULT_MSG JINJA2_IMPORT_SUCCESS)
else ()
  find_package_handle_standard_args(Jinja2 REQUIRED_VARS JINJA2_IMPORT_SUCCESS)
endif ()
