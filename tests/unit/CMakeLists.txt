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
include_directories(${source_dir}/include ${INCLUDES} ${source_dir}/gtest/include ${CMAKE_SOURCE_DIR}/src)
LINK_DIRECTORIES(${THIRD_PARTY_DIR}/libs/gmock)

# Generate automatic unittest from jinja2 templates
FILE(GLOB_RECURSE UNITTEST_SRC_TMP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *${EXT_SRC_TEST_TMP})
FOREACH(template ${UNITTEST_SRC_TMP})
	STRING(REGEX REPLACE "(.*).jinja2$" "\\1.py" script ${template})
	STRING(REGEX REPLACE ".*/(.*).jinja2$" "\\1" out_cxx ${template})
	
	ADD_CUSTOM_COMMAND(OUTPUT ${out_cxx}
			COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script} 
				${CMAKE_CURRENT_SOURCE_DIR}/${template} ${out_cxx} ${LIBSHOGUN_SRC_DIR}/base/class_list.cpp
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			COMMENT "Generating ${out_cxx}")
	LIST(APPEND TEMPLATE_GENERATED_UNITTEST ${out_cxx})
	UNSET(${script})
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

add_test(unit-test shogun-unit-test)

unset(CMAKE_DEFINITIONS)
