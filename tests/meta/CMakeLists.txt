
INCLUDE(PythonEnvironment)
GET_PYTHON_ENV()

# working directory of tester should be binary dir so link data folders
IF (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/reference_results)
	IF (NOT EXISTS ${CMAKE_SOURCE_DIR}/data/testsuite/meta/)
		MESSAGE(FATAL_ERROR "Test suite for meta examples does not exist! Please initialise and/or update the data submodule.")
	ENDIF()
	CREATE_DATA_SYMLINK(${CMAKE_SOURCE_DIR}/data/testsuite/meta/ ${CMAKE_CURRENT_BINARY_DIR}/reference_results)
ENDIF()
IF (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/generated_results)
	IF (NOT EXISTS ${CMAKE_BINARY_DIR}/examples/meta/)
		#MESSAGE(FATAL_ERROR "Meta examples directory does not exist!")
	ENDIF()
	CREATE_DATA_SYMLINK(${CMAKE_BINARY_DIR}/examples/meta ${CMAKE_CURRENT_BINARY_DIR}/generated_results)
ENDIF()

# compile tester c++ binary
ADD_EXECUTABLE(meta_example_integration_tester ${CMAKE_CURRENT_SOURCE_DIR}/tester.cpp)
ADD_DEPENDENCIES(meta_example_integration_tester shogun::shogun)
TARGET_LINK_LIBRARIES(meta_example_integration_tester shogun::shogun ${SANITIZER_LIBRARY})
IF(SANITIZER_FLAGS)
    SET_TARGET_PROPERTIES(meta_example_integration_tester PROPERTIES COMPILE_FLAGS ${SANITIZER_FLAGS})
ENDIF()
# Add tester to the dependencies of modular interfaces to make sure
# nothing will infer with them being build single-threaded.
IF(SWIG_SINGLE_THREADED)
	FOREACH(SG_INTERFACE_TARGET ${SG_INTERFACE_TARGETS})
		ADD_DEPENDENCIES(${SG_INTERFACE_TARGET} meta_example_integration_tester)
	ENDFOREACH(SG_INTERFACE_TARGET ${SG_INTERFACE_TARGETS})
ENDIF(SWIG_SINGLE_THREADED)

# get list of meta examples that can be built and skip test if if cannot
INCLUDE(FindMetaExamples)
find_meta_examples()

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

	list(FIND META_EXAMPLES ${CMAKE_SOURCE_DIR}/examples/meta/src/${REL_DIR}/${NAME}.sg LISTING_INDEX)
	# continue keyword is not available in older cmake versions (not documented!!!)
	# so use if-then-else
	IF(${LISTING_INDEX} EQUAL -1)
		MESSAGE(WARNING "Skipping integration test for meta example ${REL_DIR}/${NAME}.sg")
	ELSE()
		IF(HAVE_CXX0X OR HAVE_CXX11)
		    IF (NOT ${TRAVIS_DISABLE_META_CPP} AND NOT ${DISABLE_META_INTEGRATION_TESTS})
	            AddMetaIntegrationTest(cpp 1)
	        ENDIF()
			IF (NOT ${DISABLE_META_INTEGRATION_TESTS})
				AddMetaIntegrationTest(python INTERFACE_PYTHON)
				AddMetaIntegrationTest(java INTERFACE_JAVA)
				AddMetaIntegrationTest(r INTERFACE_R)
				AddMetaIntegrationTest(octave INTERFACE_OCTAVE)
				AddMetaIntegrationTest(csharp INTERFACE_CSHARP)
				AddMetaIntegrationTest(ruby INTERFACE_RUBY)
				AddMetaIntegrationTest(scala INTERFACE_SCALA)
				#AddMetaIntegrationTest(lua INTERFACE_LUA) # currently doesn't have meta examples
			ENDIF()
		ELSE()
			MESSAGE(WARNING "Skipping C++ meta integration tests; requires a c++11 compatible compiler.")
		ENDIF()
	ENDIF()
ENDFOREACH()
