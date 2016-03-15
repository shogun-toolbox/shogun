STRING(REGEX REPLACE "(.*)/narray.*$" "\\1" NARRAY_PATH ${NARRAY_LIB})

# add test case for each generated example
# (not generated yet so have to fake filenames from META_EXAMPLES list)
FOREACH(META_EXAMPLE ${META_EXAMPLES})
    # assume a structure <target_language>/<category>/listing.sg
	get_filename_component(EXAMPLE_NAME ${META_EXAMPLE} NAME_WE)
	IF((${CMAKE_MAJOR_VERSION}<=2) AND (${CMAKE_MINOR_VERSION}<=8) AND (${CMAKE_PATCH_VERSION}<=11))
	    get_filename_component(FULL_DIR ${META_EXAMPLE} PATH)
	ELSE()
	    # only supported from >2.8.11
	    get_filename_component(FULL_DIR ${META_EXAMPLE} DIRECTORY)
	ENDIF()
	get_filename_component(EXAMPLE_REL_DIR ${FULL_DIR} NAME)
	set(EXAMPLE_NAME_WITH_DIR "${EXAMPLE_REL_DIR}-${EXAMPLE_NAME}")
    
	add_test(NAME generated_ruby-${EXAMPLE_NAME_WITH_DIR}
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}
			COMMAND ${RUBY_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}/${EXAMPLE_NAME}.rb)
	set_property(TEST generated_ruby-${EXAMPLE_NAME_WITH_DIR} PROPERTY
		ENVIRONMENT "RUBYLIB=${RUBY_MODULAR_SOURCE_DIR}:${RUBY_MODULAR_BINARY_DIR}:.")
ENDFOREACH()
