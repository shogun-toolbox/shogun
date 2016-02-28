# add test case for each generated example
# (not generated yet so have to fake filenames from META_EXAMPLES list)
FOREACH(META_EXAMPLE ${META_EXAMPLES})
    # assume a structure <target_language>/<category>/listing.sg
    STRING(REGEX REPLACE ".*/(.*).sg" "\\1" EXAMPLE_NAME ${META_EXAMPLE})
    STRING(REGEX REPLACE ".*/(.*/.*).sg" "\\1" EXAMPLE_NAME_WITH_DIR ${META_EXAMPLE})
    STRING(REGEX REPLACE "/" "-" EXAMPLE_NAME_WITH_DIR ${EXAMPLE_NAME_WITH_DIR})
    STRING(REGEX REPLACE ".*/(.*)/.*.sg" "\\1" EXAMPLE_REL_DIR ${META_EXAMPLE})

    # meta examples have to be generated before executable and test is added
    SET(GENERATED_CPP ${EXAMPLE_REL_DIR}/${EXAMPLE_NAME}.cpp)
    SET(GENERATED_CPP_BINARY ${EXAMPLE_NAME_WITH_DIR})

    ADD_EXECUTABLE(${GENERATED_CPP_BINARY} ${GENERATED_CPP})
    ADD_DEPENDENCIES(${GENERATED_CPP_BINARY} meta_examples shogun)
    TARGET_LINK_LIBRARIES(${GENERATED_CPP_BINARY} shogun)
    SET_SOURCE_FILES_PROPERTIES(${GENERATED_CPP} PROPERTIES GENERATED 1)

    LIST(APPEND CPP_EXAMPLES ${GENERATED_CPP_BINARY})

    # run test in source dir, to have access to "data" folder
    ADD_TEST(NAME generated_cpp-${EXAMPLE_NAME_WITH_DIR}
             COMMAND ${GENERATED_CPP_BINARY}
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    LIST(APPEND INSTALL_EXAMPLES ${CMAKE_CURRENT_BINARY_DIR}/${GENERATED_CPP_BINARY})

ENDFOREACH()

add_custom_target(build_cpp_meta_examples ALL
    DEPENDS ${CPP_EXAMPLES}
    COMMENT "Compiled generated cpp examples")

INSTALL(PROGRAMS ${INSTALL_EXAMPLES}
        DESTINATION share/shogun/examples/meta/cpp
        COMPONENT meta-examples-cpp)
