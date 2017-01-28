IF (NOT TRAVIS_DISABLE_LIBSHOGUN_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/libshogun)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/libshogun)
ENDIF()

IF (PythonModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/python_modular)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/python_modular)
ENDIF()

IF(BUILD_META_EXAMPLES)
    # meta dir was added in root CMakeLists
    add_custom_target(examples DEPENDS meta_examples)
ELSE()
    add_custom_target(examples)
ENDIF()

# symlinks for windows does not work, not even cmake's create_symlink.
# make sure that the data directories are populated everywhere
IF(WIN32)
	EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/data/toy" "${CMAKE_CURRENT_BINARY_DIR}/undocumented/data")
ELSE()
	EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/data/toy" "${CMAKE_CURRENT_BINARY_DIR}/undocumented/data")
ENDIF()
