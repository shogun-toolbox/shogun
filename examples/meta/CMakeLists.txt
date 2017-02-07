# Meta dependencies
FIND_PACKAGE(PLY REQUIRED)
IF(NOT PLY_FOUND)
    message(FATAL_ERROR "Python module ply required for meta examples. Disable meta examples or install python-ply")
ENDIF()

FIND_PACKAGE(Ctags)
IF(NOT CTAGS_FOUND)
    message(FATAL_ERROR "Ctags required for meta examples.")
ENDIF()

# get list of meta examples that can be built
INCLUDE(FindMetaExamples)
find_meta_examples()

# parser output
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/parser_files)

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

    SET(EXAMPLE_LISTINGS)
    SET(BASENAME ${EXAMPLE_REL_DIR}/${EXAMPLE_NAME})
    LIST(APPEND EXAMPLE_LISTINGS ${CMAKE_CURRENT_BINARY_DIR}/cpp/${BASENAME}.cpp)
    LIST(APPEND EXAMPLE_LISTINGS ${CMAKE_CURRENT_BINARY_DIR}/python/${BASENAME}.py)
    LIST(APPEND EXAMPLE_LISTINGS ${CMAKE_CURRENT_BINARY_DIR}/octave/${BASENAME}.m)
    LIST(APPEND EXAMPLE_LISTINGS ${CMAKE_CURRENT_BINARY_DIR}/ruby/${BASENAME}.rb)
    LIST(APPEND EXAMPLE_LISTINGS ${CMAKE_CURRENT_BINARY_DIR}/csharp/${BASENAME}.cs)
    LIST(APPEND EXAMPLE_LISTINGS ${CMAKE_CURRENT_BINARY_DIR}/r/${BASENAME}.R)
    LIST(APPEND EXAMPLE_LISTINGS ${CMAKE_CURRENT_BINARY_DIR}/lua/${BASENAME}.lua)

    # Run example generation script
    SET(TEMP ${CMAKE_CURRENT_BINARY_DIR}/python/${EXAMPLE_REL_DIR}/${EXAMPLE_NAME}.py)
    add_custom_command(OUTPUT ${EXAMPLE_LISTINGS}
        COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/generator/generate.py
        -i ${META_EXAMPLE}
        -o ${CMAKE_CURRENT_BINARY_DIR}
        -t ${CMAKE_CURRENT_SOURCE_DIR}/generator/targets
        -g ${CTAGS_FILE}
        --store-vars
        --parser_files_dir=${CMAKE_CURRENT_BINARY_DIR}/parser_files
        COMMENT "Generating example ${EXAMPLE_NAME_WITH_DIR}"
        DEPENDS ctags ${META_EXAMPLE})

    LIST(APPEND TRANSLATED_META_EXAMPLES ${EXAMPLE_LISTINGS})
ENDFOREACH()
SET_SOURCE_FILES_PROPERTIES(${TRANSLATED_META_EXAMPLES} PROPERTIES GENERATED TRUE)

add_custom_target(meta_examples
    DEPENDS ${TRANSLATED_META_EXAMPLES})

# If testing is enabled we run the examples
IF(ENABLE_TESTING)
    IF (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/data)
        # copy data symlink to cmake binary dir so that we can execute meta examples in there
        CREATE_DATA_SYMLINK(${CMAKE_CURRENT_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR}/data)
    ENDIF()
                                                              

    # c++
    IF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpp AND NOT ${TRAVIS_DISABLE_META_CPP})
        IF(HAVE_CXX0X OR HAVE_CXX11)
            add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp)
        ELSE()
            MESSAGE(WARNING "Skipping C++ meta examples; requires a c++11 compatible compiler.")
        ENDIF()
    ENDIF()

    # Python
    IF (PythonModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/python)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/python)
    ENDIF()

    # java
    IF (JavaModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/java)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/java)
    ENDIF()

    # octave
    IF (OctaveModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/octave)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/octave)
    ENDIF()

    # r
    IF (RModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/r)
        # temporarily disabled as R and static calls has to be fixed
        #add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/r)
    ENDIF()
    
    # csharp
    IF (CSharpModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/csharp)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/csharp)
    ENDIF()
    
    # ruby
    IF (RubyModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ruby)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ruby)
    ENDIF()

    # lua
    IF (LuaModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lua)
        # temporarily removed since lua modular currently
        # does not support overloaded c++ methods
        # see https://github.com/shogun-toolbox/shogun/issues/3018
        #add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lua)
    ENDIF()
ENDIF()
