include_directories(${SYSTEM_INCLUDES})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${R_CXX_FLAGS}")

add_library(r_static SHARED RInterface.cpp)
target_link_libraries(r_static shogun ${R_LIBRARIES})
set_target_properties(r_static PROPERTIES PREFIX "")

# Add static-interface to the dependencies of modular interfaces to make
# sure nothing will infer with them being build single-threaded.
IF(SWIG_SINGLE_THREADED)
	FOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
		ADD_DEPENDENCIES(${SG_MODULAR_INTERFACE_TARGET} r_static)
	ENDFOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
ENDIF(SWIG_SINGLE_THREADED)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/r_static.so
       DESTINATION ${R_COMPONENT_LIB_PATH}/sg/libs
       COMPONENT r
   	   RENAME sg.so)

install(CODE "FILE(MAKE_DIRECTORY \$ENV{DESTDIR}/${R_COMPONENT_LIB_PATH}/sg/Meta)")
install(CODE "FILE(MAKE_DIRECTORY \$ENV{DESTDIR}/${R_COMPONENT_LIB_PATH}/sg/R)")
install(CODE "EXECUTE_PROCESS(COMMAND ${CMAKE_SOURCE_DIR}/src/.r-install.sh ${R_COMPONENT_LIB_PATH} sg so saveRDS WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
