include(CommonModularInterface)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${JNI_INCLUDE_DIRS})

GENERATE_MODULAR_TARGET(java ${CMAKE_CURRENT_SOURCE_DIR} "")

ADD_CUSTOM_COMMAND(TARGET java_modular
				   POST_BUILD
				   COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/build.xml ${CMAKE_CURRENT_BINARY_DIR}/build.xml
				   COMMAND "${ANT_EXECUTABLE}" dist
				   COMMENT "Creating shogun.jar")

INSTALL(TARGETS java_modular
	LIBRARY DESTINATION lib/jni)

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/shogun.jar 
		DESTINATION share/java)