INCLUDE(PythonEnvironment)
GET_PYTHON_ENV()

# add test case for each generated python example
# (not generated yet so have to fake filenames from META_EXAMPLES list)
FOREACH(META_EXAMPLE ${META_EXAMPLES})
    # ".*/(.*/.*).sg"
	STRING(REGEX REPLACE ".*/(.*).sg" "\\1" EXAMPLE_NAME ${META_EXAMPLE})
	STRING(REGEX REPLACE ".*/(.*/.*).sg" "\\1" EXAMPLE_NAME_WITH_DIR ${META_EXAMPLE})
	STRING(REGEX REPLACE "/" "-" EXAMPLE_NAME_WITH_DIR ${EXAMPLE_NAME_WITH_DIR})
    STRING(REGEX REPLACE ".*/(.*)/.*.sg" "\\1" EXAMPLE_REL_DIR ${META_EXAMPLE})
    
    # run test in source dir, to have access to "data" folder
    # run listing from binary dir though as it is generated there
	add_test(NAME generated_python_modular-${EXAMPLE_NAME_WITH_DIR}
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}/${EXAMPLE_NAME}.py)
	set_property(TEST generated_python_modular-${EXAMPLE_NAME_WITH_DIR} PROPERTY
				ENVIRONMENT "${PYTHON_ENV_VARS}")

ENDFOREACH()
