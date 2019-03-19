FIND_PACKAGE(Jinja2)
FIND_PACKAGE(Ctags)

# Find GTEST and GMOCK frameworks
include(external/GoogleTestNMock)
ExternalProject_Get_Property(GoogleMock source_dir)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
LINK_DIRECTORIES(${THIRD_PARTY_DIR}/libs/gmock)

function (add_unit_test_executable EXECUTABLE TARGET SRC)
	ADD_EXECUTABLE(${EXECUTABLE} ${SRC} utils/Utils.cpp)
	add_dependencies(${EXECUTABLE} GoogleMock shogun::shogun)
	target_include_directories(${EXECUTABLE} PRIVATE ${source_dir}/googlemock/include ${INCLUDES} ${source_dir}/googletest/include)
	target_link_libraries(${EXECUTABLE} PRIVATE shogun_deps gmock gtest ${SANITIZER_LIBRARY})
	IF (SANITIZER_FLAGS)
		set_target_properties(${EXECUTABLE} PROPERTIES COMPILE_FLAGS ${SANITIZER_FLAGS})
	ENDIF()
	set_target_properties (${EXECUTABLE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
	set_target_properties (${EXECUTABLE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
	set_target_properties (${EXECUTABLE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)

	# In some cases gTest segfaults with GCC >= 6.0.0.  This is a dirty fix.
	# TODO: Update to gTest-release with proper support for GCC >= 6.0.0.
	# See:  https://github.com/google/googletest/issues/705
	IF(CMAKE_COMPILER_IS_GNUCXX)
		IF(NOT "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "6.0.0")
			target_compile_options(${EXECUTABLE}
					PUBLIC -fno-delete-null-pointer-checks)
		ENDIF(NOT "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "6.0.0")
	ELSEIF(MSVC)
		set_property(TARGET ${EXECUTABLE} APPEND_STRING PROPERTY COMPILE_FLAGS " ${GTEST_TR1_SILENCER}")
	ENDIF()

	ADD_CUSTOM_TARGET(${TARGET}
			COMMAND ${CMAKE_BINARY_DIR}/bin/${EXECUTABLE}
			DEPENDS ${EXECUTABLE})

	shogun_discover_tests(${EXECUTABLE})
endfunction ()

# Generate automatic unittest from jinja2 templates
if(JINJA2_IMPORT_SUCCESS)
    ADD_CUSTOM_COMMAND(OUTPUT clone_unittest.cc
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/base/clone_unittest.cc.py
        ${CMAKE_CURRENT_SOURCE_DIR}/base/clone_unittest.cc.jinja2
        clone_unittest.cc
        ${CMAKE_BINARY_DIR}/src/shogun/base/class_list.cpp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/base/clone_unittest.cc.py
        ${CMAKE_CURRENT_SOURCE_DIR}/base/clone_unittest.cc.jinja2
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating clone_unittest.cc")
    LIST(APPEND TEMPLATE_GENERATED_UNITTEST clone_unittest.cc)
ELSE()
    MESSAGE(WARNING "Please install jinja2 for automatic generated tests.")
ENDIF()

IF(NOT CTAGS_FOUND)
	MESSAGE("Please install Ctags for trained models serialization tests.")
ELSEIF(NOT HAVE_HDF5)
	MESSAGE("Please install HDF5 for trained models serialization tests.")
ELSE()
	ADD_CUSTOM_COMMAND(OUTPUT trained_model_serialization_test.h
			COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/base/trained_model_serialization_test.cc.py
			${CTAGS_FILE}
			${CMAKE_CURRENT_BINARY_DIR}/trained_model_serialization_test.h
			${CMAKE_BINARY_DIR}/src/shogun/lib/config.h
			DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/base/trained_model_serialization_test.cc.py
			ctags
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			COMMENT "Generating trained_model_serialization_test.h")
	LIST(APPEND SERIALIZATION_UNITTEST ${CMAKE_CURRENT_SOURCE_DIR}/base/trained_model_serialization_test.cc)
	LIST(APPEND SERIALIZATION_UNITTEST ${CMAKE_CURRENT_BINARY_DIR}/trained_model_serialization_test.h)

	LIST(APPEND SERIALIZATION_UNITTEST base/main_unittest.cc)
	add_unit_test_executable(shogun-serialization-unit-test serialization-unit-tests "${SERIALIZATION_UNITTEST}")
ENDIF()

add_executable (discover_gtest_tests ${CMAKE_CURRENT_SOURCE_DIR}/discover_gtest_tests.cpp)
set_target_properties (discover_gtest_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties (discover_gtest_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
set_target_properties (discover_gtest_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)

# collect the unittests
FILE(GLOB_RECURSE UNITTEST_SRC *_unittest.cc)
LIST(APPEND UNITTEST_SRC ${TEMPLATE_GENERATED_UNITTEST})

add_unit_test_executable(shogun-unit-test unit-tests "${UNITTEST_SRC}")

# Add unittests to the dependencies of modular interfaces to make sure nothing
# will infer with them being build single-threaded.
IF(SWIG_SINGLE_THREADED)
	FOREACH(SG_INTERFACE_TARGET ${SG_INTERFACE_TARGETS})
		ADD_DEPENDENCIES(${SG_INTERFACE_TARGET} shogun-unit-test)
	ENDFOREACH(SG_INTERFACE_TARGET ${SG_INTERFACE_TARGETS})
ENDIF(SWIG_SINGLE_THREADED)

unset(CMAKE_DEFINITIONS)
