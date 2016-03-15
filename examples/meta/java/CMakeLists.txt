SET(CLASSPATH "${JBLAS}:${SHOGUN_JAR}:${CMAKE_CURRENT_BINARY_DIR}")
SET(JAVAOPTS "-Xmx1024m")
SET(JAVA_LIB_PATH "${JAVA_MODULAR_BUILD_DIR}")

# add test case for each generated example
# (not generated yet so have to fake filenames from META_EXAMPLES list)
FOREACH(META_EXAMPLE ${META_EXAMPLES})
    # assume a structure <target_language>/<category>/listing.sg
	get_filename_component(EXAMPLE_NAME ${META_EXAMPLE} NAME_WE)
	IF(${CMAKE_VERSION} VERSION_LESS 2.8.12)
	    get_filename_component(FULL_DIR ${META_EXAMPLE} PATH)
	ELSE()
	    # only supported from >2.8.11
	    get_filename_component(FULL_DIR ${META_EXAMPLE} DIRECTORY)
	ENDIF()
	get_filename_component(EXAMPLE_REL_DIR ${FULL_DIR} NAME)
	set(EXAMPLE_NAME_WITH_DIR "${EXAMPLE_REL_DIR}-${EXAMPLE_NAME}")

	ADD_CUSTOM_COMMAND(OUTPUT compiled_java_${EXAMPLE_NAME_WITH_DIR}
				  COMMAND ${Java_JAVAC_EXECUTABLE} -cp ${CLASSPATH}
					-d ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}/${EXAMPLE_NAME}.java
				  DEPENDS java_modular)
	LIST(APPEND GENERATED_JAVA_EXAMPLES compiled_java_${EXAMPLE_NAME_WITH_DIR})

	add_test(NAME generated_java-${EXAMPLE_NAME_WITH_DIR}
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}
			COMMAND ${Java_JAVA_EXECUTABLE} ${JAVAOPTS}
				-cp ${CLASSPATH}:${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}
				-Djava.library.path=${JAVA_LIB_PATH} ${EXAMPLE_NAME})
ENDFOREACH()

add_custom_target(build_java_meta_examples ALL
				 DEPENDS ${GENERATED_JAVA_EXAMPLES}
				 COMMENT "Compiled generated java examples")
