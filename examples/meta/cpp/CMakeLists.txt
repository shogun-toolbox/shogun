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

    # meta examples have to be generated before executable and test is added
    SET(GENERATED_CPP ${EXAMPLE_REL_DIR}/${EXAMPLE_NAME}.cpp)
    SET(GENERATED_CPP_BINARY ${EXAMPLE_NAME})

    # generate binary in relative directory
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_REL_DIR})

    ADD_EXECUTABLE(${GENERATED_CPP_BINARY} ${GENERATED_CPP})
    ADD_DEPENDENCIES(${GENERATED_CPP_BINARY} meta_examples shogun)
    TARGET_LINK_LIBRARIES(${GENERATED_CPP_BINARY} shogun)
    SET_SOURCE_FILES_PROPERTIES(${GENERATED_CPP} PROPERTIES GENERATED 1)
    SET_TARGET_PROPERTIES(${GENERATED_CPP_BINARY} PROPERTIES COMPILE_FLAGS "-Wno-unused-variable")

    LIST(APPEND GENERATED_CPP_EXAMPLES ${GENERATED_CPP_BINARY})

    ADD_TEST(NAME generated_cpp-${EXAMPLE_NAME}
             COMMAND ${GENERATED_CPP_BINARY}
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_REL_DIR})

ENDFOREACH()

add_custom_target(build_cpp_meta_examples ALL
    DEPENDS ${GENERATED_CPP_EXAMPLES}
    COMMENT "Compiled generated cpp examples")
