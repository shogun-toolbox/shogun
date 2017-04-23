string( REGEX REPLACE "/scala$" "/java" CMAKE_JAVA_DIR "${CMAKE_CURRENT_BINARY_DIR}" )
SET(CLASSPATH "${JBLAS}:${SHOGUN_JAR}:${Scala_JAR_EXECUTABLE}:${CMAKE_CURRENT_BINARY_DIR}")
SET(JAVA_LIB_PATH "${JAVA_MODULAR_BUILD_DIR}")

# add test case for each generated example
# (not generated yet so have to fake filenames from META_EXAMPLES list)
FOREACH(META_EXAMPLE ${META_EXAMPLES})
    # assume a structure <target_language>/<category>/listing.sg
    GET_META_EXAMPLE_VARS(${META_EXAMPLE} EXAMPLE_NAME EXAMPLE_REL_DIR EXAMPLE_NAME_WITH_DIR)
    set(SCALA_FOLDER "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}")

    file(MAKE_DIRECTORY ${SCALA_FOLDER})

    add_test(NAME generated_scala-${EXAMPLE_NAME_WITH_DIR}
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}
			COMMAND ${Scala_SCALA_EXECUTABLE}
				-cp ${CLASSPATH}:${CMAKE_JAVA_DIR}/${EXAMPLE_REL_DIR}
				-Djava.library.path=${JAVA_LIB_PATH} ${EXAMPLE_NAME})

ENDFOREACH()

# Set up this target to be consistent
# with all the other meta example targets.
add_custom_target(build_scala_meta_examples ALL
            DEPENDS build_java_meta_examples
            COMMENT "Compiled java meta examples to be used with scala")
