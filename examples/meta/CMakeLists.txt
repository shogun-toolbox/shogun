# Check python dependency and return error
# Generate examples
# Check which interfaces are turned on
# Add CMakeLists file for each turned on interface


# Check if PLY is installed
find_package(PLY)
IF(NOT PLY_FOUND)
    message(WARNING "Python module PLY was not found. This may cause example generation to be very slow.")
ENDIF()

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
    COMMENT "Generating examples from meta-language")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/targets)
