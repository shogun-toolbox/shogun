# transform defines to -D<definition> string
foreach(D IN LISTS DEFINES)
	SET(CMAKE_DEFINITIONS "${CMAKE_DEFINITIONS} -D${D}")
endforeach()

# Find GTEST and GMOCK frameworks
include(external/GoogleTestNMock)
ExternalProject_Get_Property(GoogleMock source_dir)
if(SYSTEM_INCLUDES)
	INCLUDE_DIRECTORIES(SYSTEM ${SYSTEM_INCLUDES})
endif()
include_directories(${source_dir}/include ${INCLUDES} ${source_dir}/gtest/include ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
LINK_DIRECTORIES(${THIRD_PARTY_DIR}/libs/gmock)

# Generate automatic unittest from jinja2 templates
SET(clone_unittest.cc_CMD
	${CMAKE_CURRENT_SOURCE_DIR}/base/clone_unittest.cc.py 
	${CMAKE_CURRENT_SOURCE_DIR}/base/clone_unittest.cc.jinja2 
	clone_unittest.cc 
	${LIBSHOGUN_SRC_DIR}/base/class_list.cpp)
LIST(APPEND UNITTEST_TEMPLATES clone_unittest.cc)

SET(SerializationAscii_unittest.cc_CMD
	${CMAKE_CURRENT_SOURCE_DIR}/base/clone_unittest.cc.py 
	${CMAKE_CURRENT_SOURCE_DIR}/io/SerializationAscii_unittest.cc.jinja2 
	SerializationAscii_unittest.cc
	${LIBSHOGUN_SRC_DIR}/base/class_list.cpp)
LIST(APPEND UNITTEST_TEMPLATES SerializationAscii_unittest.cc)
#
#SET(SerializationJSON_unittest.cc_CMD 
#	${CMAKE_CURRENT_SOURCE_DIR}/base/clone_unittest.cc.py
#	${CMAKE_CURRENT_SOURCE_DIR}/io/SerializationJSON_unittest.cc.jinja2 
#	SerializationJSON_unittest.cc 
#	${LIBSHOGUN_SRC_DIR}/base/class_list.cpp)
#LIST(APPEND UNITTEST_TEMPLATES SerializationJSON_unittest.cc)
#
SET(SerializationXML_unittest.cc_CMD 
	${CMAKE_CURRENT_SOURCE_DIR}/base/clone_unittest.cc.py
	${CMAKE_CURRENT_SOURCE_DIR}/io/SerializationXML_unittest.cc.jinja2 
	SerializationXML_unittest.cc 
	${LIBSHOGUN_SRC_DIR}/base/class_list.cpp)
LIST(APPEND UNITTEST_TEMPLATES SerializationXML_unittest.cc)

SET(SerializationHDF5_unittest.cc_CMD 
	${CMAKE_CURRENT_SOURCE_DIR}/base/clone_unittest.cc.py
	${CMAKE_CURRENT_SOURCE_DIR}/io/SerializationHDF5_unittest.cc.jinja2 
	SerializationHDF5_unittest.cc 
	${LIBSHOGUN_SRC_DIR}/base/class_list.cpp)
LIST(APPEND UNITTEST_TEMPLATES SerializationHDF5_unittest.cc)

FOREACH(out_cxx ${UNITTEST_TEMPLATES})
	ADD_CUSTOM_COMMAND(OUTPUT ${out_cxx}
			COMMAND ${PYTHON_EXECUTABLE} ${${out_cxx}_CMD}
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			COMMENT "Generating ${out_cxx}")
	LIST(APPEND TEMPLATE_GENERATED_UNITTEST ${out_cxx})
ENDFOREACH()

# collect the unittests
FILE(GLOB_RECURSE UNITTEST_SRC *_unittest.cc)

ADD_EXECUTABLE(shogun-unit-test ${UNITTEST_SRC} ${TEMPLATE_GENERATED_UNITTEST})
add_dependencies(shogun-unit-test gmock gtest)
target_link_libraries(shogun-unit-test shogun gmock gtest)
set_target_properties(shogun-unit-test PROPERTIES COMPILE_DEFINITIONS "${DEFINES}")

ADD_CUSTOM_TARGET(unit-tests 
	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/shogun-unit-test
	DEPENDS shogun-unit-test)

add_test(unit-test shogun-unit-test --only-on-failure)

unset(CMAKE_DEFINITIONS)
