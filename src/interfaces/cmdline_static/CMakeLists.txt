include_directories(${SYSTEM_INCLUDES})
SET(CMDLINE_STATIC_EXEC "${CMAKE_CURRENT_BINARY_DIR}/cmdline_static" PARENT_SCOPE)
add_executable(cmdline_static cmdline.cpp CmdLineInterface.cpp)
target_link_libraries(cmdline_static shogun)

# Add static-interface to the dependencies of modular interfaces to make
# sure nothing will infer with them being build single-threaded.
IF(SWIG_SINGLE_THREADED)
	FOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
		ADD_DEPENDENCIES(${SG_MODULAR_INTERFACE_TARGET} cmdline_static)
	ENDFOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
ENDIF(SWIG_SINGLE_THREADED)

INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/cmdline_static
		DESTINATION bin
		COMPONENT cmdline
		RENAME shogun)
