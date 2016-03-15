# add test case for each generated example
# (not generated yet so have to fake filenames from META_EXAMPLES list)
FOREACH(META_EXAMPLE ${META_EXAMPLES})
    # assume a structure <target_language>/<category>/listing.sg
	get_filename_component(EXAMPLE_NAME ${META_EXAMPLE} NAME_WE)
	get_filename_component(FULL_DIR ${META_EXAMPLE} DIRECTORY)
	get_filename_component(EXAMPLE_REL_DIR ${FULL_DIR} NAME)
	set(EXAMPLE_NAME_WITH_DIR "${EXAMPLE_REL_DIR}-${EXAMPLE_NAME}")
    
	add_test(NAME generated_octave-${EXAMPLE_NAME_WITH_DIR}
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}
			COMMAND ${OCTAVE_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}/${EXAMPLE_NAME}.m)
	set_property(TEST generated_octave-${EXAMPLE_NAME_WITH_DIR} PROPERTY
				ENVIRONMENT "OCTAVE_PATH=${OCTAVE_MODULAR_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/tools")

ENDFOREACH()
