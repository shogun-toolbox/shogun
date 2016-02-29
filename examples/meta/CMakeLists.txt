# Check python dependency and return error
# Generate examples
# Check which interfaces are turned on
# Add CMakeLists file for each turned on interface

# Disabled (temporary) as non-portable solution. This is replaced by allowing any [a-zA-Z_][a-zA-Z0-9_]* as shogun keyword
# Generate type list
# execute_process(COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/generator/types/get_type_list.sh > ${CMAKE_CURRENT_SOURCE_DIR}/generator/types/typelist)

# Run example generation script
add_custom_target(meta_examples
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/generator/generate.py
    -i ${CMAKE_CURRENT_SOURCE_DIR}/src
    -o ${CMAKE_CURRENT_BINARY_DIR}
    -t ${CMAKE_CURRENT_SOURCE_DIR}/generator/targets
    -g ${CTAGS_FILE}
    --store-vars
    COMMENT "Generating examples from meta-language"
    DEPENDS ctags)

# Collect all meta example listings and store in variable
FILE(GLOB_RECURSE META_EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.sg)

# run the Python generator tests
IF(ENABLE_TESTING)
    find_program(NOSETESTS nosetests)
    IF(NOSETESTS)
	    add_test(NAME meta-examples-generator
	    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/generator
	    COMMAND ${NOSETESTS})
    ENDIF(NOSETESTS)
ENDIF(ENABLE_TESTING)

# If testing is enabled we run the examples
IF(ENABLE_TESTING)
    # c++
    IF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp)
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
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/r)
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
