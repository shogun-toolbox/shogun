INCLUDE(PythonEnvironment)
GET_PYTHON_ENV()

# working directory of tester should be binary dir so link data folders
IF (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/reference_results)
	IF (NOT EXISTS ${CMAKE_SOURCE_DIR}/data/testsuite/meta/)
		MESSAGE(FATAL_ERROR "Test suite for meta examples does not exists!")
	ENDIF()
	EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/data/testsuite/meta/" "${CMAKE_CURRENT_BINARY_DIR}/reference_results")
ENDIF()
IF (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/generated_results)
	IF (NOT EXISTS ${CMAKE_BINARY_DIR}/examples/meta/)
		#MESSAGE(FATAL_ERROR "Meta examples directory does not exists!")
	ENDIF()
	EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_BINARY_DIR}/examples/meta" "${CMAKE_CURRENT_BINARY_DIR}/generated_results")
ENDIF()

# compile tester c++ binary
ADD_EXECUTABLE(meta_example_integration_tester ${CMAKE_CURRENT_SOURCE_DIR}/tester.cpp)
ADD_DEPENDENCIES(meta_example_integration_tester shogun)
TARGET_LINK_LIBRARIES(meta_example_integration_tester shogun ${SANITIZER_LIBRARY})
IF(SANITIZER_FLAGS)
    SET_TARGET_PROPERTIES(meta_example_integration_tester PROPERTIES COMPILE_FLAGS ${SANITIZER_FLAGS})
ENDIF()
# Add tester to the dependencies of modular interfaces to make sure
# nothing will infer with them being build single-threaded.
IF(SWIG_SINGLE_THREADED)
	FOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
		ADD_DEPENDENCIES(${SG_MODULAR_INTERFACE_TARGET} meta_example_integration_tester)
	ENDFOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
ENDIF(SWIG_SINGLE_THREADED)

FILE(GLOB META_INTEGRATION_REFERENCES ${CMAKE_CURRENT_BINARY_DIR}/reference_results/*/*.dat)
FOREACH(REFERENCE_FILE ${META_INTEGRATION_REFERENCES})
	# assume a structure <target_language>/<category>/result.dat
	get_filename_component(NAME ${REFERENCE_FILE} NAME_WE)
	IF(${CMAKE_VERSION} VERSION_LESS 2.8.12)
	    get_filename_component(FULL_DIR ${REFERENCE_FILE} PATH)
	ELSE()
	    # only supported from >2.8.11
	    get_filename_component(FULL_DIR ${REFERENCE_FILE} DIRECTORY)
	ENDIF()
	get_filename_component(REL_DIR ${FULL_DIR} NAME)
	set(NAME_WITH_DIR "${REL_DIR}-${NAME}")

	# cpp
	# TODO implement other languages tests
	IF(HAVE_CXX0X OR HAVE_CXX11)
	    add_test(NAME integration_meta_cpp-${NAME_WITH_DIR}
		    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		    COMMAND meta_example_integration_tester ${REL_DIR} ${NAME}.dat cpp generated_results reference_results)
	    set_tests_properties(
		    integration_meta_cpp-${NAME_WITH_DIR}
			    PROPERTIES
			    DEPENDS generated_cpp-${NAME}
	    )
	ELSE()
	    MESSAGE(WARNING "Skipping C++ meta integration tests; requires a c++11 compatible compiler.")
	ENDIF()
ENDFOREACH()
