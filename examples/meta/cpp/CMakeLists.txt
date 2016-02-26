# add test case for each generated example
# (not generated yet so have to fake filenames from META_EXAMPLES list)
FOREACH(META_EXAMPLE ${META_EXAMPLES})
    # assume a structure <target_language>/<category>/listing.sg
	STRING(REGEX REPLACE ".*/(.*).sg" "\\1" EXAMPLE_NAME ${META_EXAMPLE})
	STRING(REGEX REPLACE ".*/(.*/.*).sg" "\\1" EXAMPLE_NAME_WITH_DIR ${META_EXAMPLE})
	STRING(REGEX REPLACE "/" "-" EXAMPLE_NAME_WITH_DIR ${EXAMPLE_NAME_WITH_DIR})
    STRING(REGEX REPLACE ".*/(.*)/.*.sg" "\\1" EXAMPLE_REL_DIR ${META_EXAMPLE})

	ADD_CUSTOM_COMMAND(OUTPUT ${EXAMPLE_NAME_WITH_DIR}
			  COMMAND g++ -std=c++11 -I /home/heiko/git/shogun/shogun_develop/shogun/src -L /home/heiko/git/shogun/shogun_develop/shogun/build/src/shogun  -o ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}/${EXAMPLE_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}/${EXAMPLE_NAME}.cpp -lshogun
			  DEPENDS libshogun)

	LIST(APPEND CPP_EXAMPLES ${EXAMPLE_NAME_WITH_DIR})

    # run test in source dir, to have access to "data" folder
    # run class from binary dir though as it is generated there
    # add class location to classpath to do that
	add_test(generated_cpp-${EXAMPLE_NAME_WITH_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}/${EXAMPLE_NAME})
	set_property(TEST generated_cpp-${EXAMPLE_NAME_WITH_DIR} PROPERTY
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
	LIST(APPEND INSTALL_EXAMPLES ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR}/${EXAMPLE_NAME})
	# TODO: Need some way to have finished building libshogun.so before this
	# when running in multithreaded way. Adding modular interfaces as dependency here doesnt work as there might be None activated.

ENDFOREACH()

add_custom_target(build_cpp_meta_examples ALL
				 DEPENDS ${CPP_EXAMPLES}
				 COMMENT "Compiled generated java examples")

#INSTALL(PROGRAMS ${INSTALL_EXAMPLES}
#	DESTINATION share/shogun/examples/libshogun
#	COMPONENT libshogun-examples)
