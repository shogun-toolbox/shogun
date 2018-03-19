# - Find the PyArrow libraries
# This module finds if PyArrow is installed, and sets the following variables
# indicating where it is.
#
# PYARROW_FOUND - was PyArrow found
# PYARROW_VERSION - the version of PyArrow found as a string
# PYARROW_VERSION_MAJOR - the major version number of PyArrow
# PYARROW_VERSION_MINOR - the minor version number of PyArrow
# PYARROW_VERSION_PATCH - the patch version number of PyArrow
# PYARROW_VERSION_DECIMAL - e.g. version 1.6.1 is 10601
# PYARROW_INCLUDE_DIRS - path to the PyArrow include files

if(PYARROW_FIND_REQUIRED)
	find_package(PythonInterp REQUIRED)
else()
	find_package(PythonInterp)
endif()

if(NOT PYTHONINTERP_FOUND)
	set(PYARROW_FOUND FALSE)
	return()
endif()

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
	"import pyarrow as pa; print(pa.__version__); print(pa.get_include());"
	RESULT_VARIABLE _PYARROW_SEARCH_SUCCESS
	OUTPUT_VARIABLE _PYARROW_VALUES_OUTPUT
	ERROR_VARIABLE _PYARROW_ERROR_VALUE
	OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _PYARROW_SEARCH_SUCCESS MATCHES 0)
	if(PYARROW_FIND_REQUIRED)
		message(FATAL_ERROR
		"PyArrow import failure:\n${_PYARROW_ERROR_VALUE}")
	endif()
	set(PYARROW_FOUND FALSE)
	return()
endif()

# Convert the process output into a list
string(REGEX REPLACE ";" "\\\\;" _PYARROW_VALUES ${_PYARROW_VALUES_OUTPUT})
string(REGEX REPLACE "\n" ";" _PYARROW_VALUES ${_PYARROW_VALUES})
list(GET _PYARROW_VALUES 0 PYARROW_VERSION)
list(GET _PYARROW_VALUES 1 PYARROW_INCLUDE_DIRS)

string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _VER_CHECK "${PYARROW_VERSION}")
if("${_VER_CHECK}" STREQUAL "")
	# The output from Python was unexpected. Raise an error always
	# here, because we found NumPy, but it appears to be corrupted somehow.
	message(FATAL_ERROR
	"Requested version and include path from PyArrow, got instead:\n${_PYARROW_VALUES_OUTPUT}\n")
	return()
endif()

# Make sure all directory separators are '/'
string(REGEX REPLACE "\\\\" "/" PYARROW_INCLUDE_DIRS ${PYARROW_INCLUDE_DIRS})

# Get the major and minor version numbers
string(REGEX REPLACE "\\." ";" _PYARROW_VERSION_LIST ${PYARROW_VERSION})
list(GET _PYARROW_VERSION_LIST 0 PYARROW_VERSION_MAJOR)
list(GET _PYARROW_VERSION_LIST 1 PYARROW_VERSION_MINOR)
list(GET _PYARROW_VERSION_LIST 2 PYARROW_VERSION_PATCH)
string(REGEX MATCH "[0-9]*" PYARROW_VERSION_PATCH ${PYARROW_VERSION_PATCH})
math(EXPR PYARROW_VERSION_DECIMAL
"(${PYARROW_VERSION_MAJOR} * 10000) + (${PYARROW_VERSION_MINOR} * 100) + ${PYARROW_VERSION_PATCH}")

find_package_handle_standard_args(PYARROW REQUIRED_VARS PYARROW_INCLUDE_DIRS VERSION_VAR PYARROW_VERSION)
