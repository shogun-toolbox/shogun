# Check python dependency and return error
# Generate examples
# Check which interfaces are turned on
# Add CMakeLists file for each turned on interface

# Run example generation script
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py -i ${CMAKE_CURRENT_SOURCE_DIR}/examples -o ${CMAKE_CURRENT_SOURCE_DIR}/generated-examples -t ${CMAKE_CURRENT_SOURCE_DIR}/targets)

# If testing is enabled we run the examples
IF(ENABLE_TESTING)
    # Python
    IF (PythonModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/generated-examples/python)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/generated-examples/python)
    ENDIF()

    # java
    IF (JavaModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/generated-examples/java)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/generated-examples/java)
    ENDIF()

    # octave
    IF (OctaveModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/generated-examples/octave)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/generated-examples/octave)
    ENDIF()

    # r
    IF (RModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/generated-examples/r)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/generated-examples/r)
    ENDIF()
ENDIF()
