# loosely based on FindNumPy.cmake, modifications by Heiko Strathmann

# Finding PLY involves calling the Python interpreter
if(PLY_FIND_REQUIRED)
	find_package(PythonInterp REQUIRED)
else()
	find_package(PythonInterp)
endif()

if(NOT PYTHONINTERP_FOUND)
	set(PLY_FOUND FALSE)
	return()
endif()

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
	"import ply as p"
	RESULT_VARIABLE _PLY_SEARCH_SUCCESS
	OUTPUT_VARIABLE _PLY_VALUES_OUTPUT
	ERROR_VARIABLE _PLY_ERROR_VALUE
	OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _PLY_SEARCH_SUCCESS MATCHES 0)
	if(PLY_FIND_REQUIRED)
		message(FATAL_ERROR
		"ply import failure:\n${_NUMPY_ERROR_VALUE}")
	endif()
	set(PLY_FOUND FALSE)
	return()
endif()

set(PLY_FOUND TRUE)
