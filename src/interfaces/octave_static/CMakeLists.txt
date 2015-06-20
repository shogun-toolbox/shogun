include_directories(${OCTAVE_INCLUDE_DIRS} ${SYSTEM_INCLUDES} ${INCLUDES})

add_library(octave_static SHARED OctaveInterface.cpp)
target_link_libraries(octave_static shogun ${OCTAVE_LIBRARY})

# Add static-interface to the dependencies of modular interfaces to make
# sure nothing will infer with them being build single-threaded.
IF(SWIG_SINGLE_THREADED)
	FOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
		ADD_DEPENDENCIES(${SG_MODULAR_INTERFACE_TARGET} octave_static)
	ENDFOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
ENDIF(SWIG_SINGLE_THREADED)
