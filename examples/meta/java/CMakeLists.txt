FILE(GLOB EXAMPLES_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.java)

SET(CLASSPATH "${JBLAS}:${SHOGUN_JAR}:${CMAKE_CURRENT_BINARY_DIR}")
SET(JAVAOPTS "-Xmx1024m")
SET(JAVA_LIB_PATH "${JAVA_MODULAR_BUILD_DIR}")

# add test case for each generated example
# (not generated yet so have to fake filenames from META_EXAMPLES list)
FOREACH(META_EXAMPLE ${META_EXAMPLES})
    # assume a structure <target_language>/<category>/listing.sg
	STRING(REGEX REPLACE ".*/(.*).sg" "\\1" EXAMPLE_NAME ${META_EXAMPLE})
	STRING(REGEX REPLACE ".*/(.*/.*).sg" "\\1" EXAMPLE_NAME_WITH_DIR ${META_EXAMPLE})
	STRING(REGEX REPLACE "/" "-" EXAMPLE_NAME_WITH_DIR ${EXAMPLE_NAME_WITH_DIR})
    STRING(REGEX REPLACE ".*/(.*)/.*.sg" "\\1" EXAMPLE_REL_DIR ${META_EXAMPLE})

	ADD_CUSTOM_COMMAND(OUTPUT ${EXAMPLE_NAME_WITH_DIR}.class
				  COMMAND ${Java_JAVAC_EXECUTABLE} -cp ${CLASSPATH}
					-d ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}/${EXAMPLE_NAME}.java
				  DEPENDS java_modular)
	LIST(APPEND JAVA_EXAMPLES ${EXAMPLE_NAME_WITH_DIR}.class)

    # run test in source dir, to have access to "data" folder
    # run class from binary dir though as it is generated there
    # add class location to classpath to do that
	add_test(NAME generated_java-${EXAMPLE_NAME_WITH_DIR}
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			COMMAND ${Java_JAVA_EXECUTABLE} ${JAVAOPTS}
				-cp ${CLASSPATH}:${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}
				-Djava.library.path=${JAVA_LIB_PATH} ${EXAMPLE_NAME})
ENDFOREACH()

add_custom_target(build_generated_java_examples ALL
				 DEPENDS ${JAVA_EXAMPLES}
				 COMMENT "Compiled generated java examples")
