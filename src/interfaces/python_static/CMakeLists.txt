include_directories(${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_DIRS} ${SYSTEM_INCLUDES})

add_library(python_static SHARED PythonInterface.cpp)
target_link_libraries(python_static shogun ${PYTHON_LIBRARIES})

# Add static-interface to the dependencies of modular interfaces to make
# sure nothing will infer with them being build single-threaded.
IF(SWIG_SINGLE_THREADED)
	FOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
		ADD_DEPENDENCIES(${SG_MODULAR_INTERFACE_TARGET} python_static)
	ENDFOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
ENDIF(SWIG_SINGLE_THREADED)
