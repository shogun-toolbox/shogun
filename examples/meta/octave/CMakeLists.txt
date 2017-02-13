# add test case for each generated example
# (not generated yet so have to fake filenames from META_EXAMPLES list)
FOREACH(META_EXAMPLE ${META_EXAMPLES})
    # assume a structure <target_language>/<category>/listing.sg
    GET_META_EXAMPLE_VARS(${META_EXAMPLE} EXAMPLE_NAME EXAMPLE_REL_DIR EXAMPLE_NAME_WITH_DIR)
    
	add_test(NAME generated_octave-${EXAMPLE_NAME_WITH_DIR}
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}
			COMMAND ${OCTAVE_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}/${EXAMPLE_NAME}.m)
	set_property(TEST generated_octave-${EXAMPLE_NAME_WITH_DIR} PROPERTY
				ENVIRONMENT "OCTAVE_PATH=${OCTAVE_MODULAR_BINARY_DIR}")

ENDFOREACH()
