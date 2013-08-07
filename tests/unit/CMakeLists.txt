# transform defines to -D<definition> string
foreach(D IN LISTS DEFINES)
	SET(CMAKE_DEFINITIONS "${CMAKE_DEFINITIONS} -D${D}")
endforeach()

# Find GTEST and GMOCK frameworks
include(external/GoogleTestNMock)
ExternalProject_Get_Property(GoogleMock source_dir)
include_directories(${INCLUDES} ${CMAKE_SOURCE_DIR}/src 
		${source_dir}/include ${source_dir}/gtest/include)
LINK_DIRECTORIES(${THIRD_PARTY_DIR}/libs/gmock)

# 


FILE(GLOB_RECURSE UNITTEST_SRC *_unittest.cc)

ADD_EXECUTABLE(shogun-unit-test ${UNITTEST_SRC})
add_dependencies(shogun-unit-test gmock gtest)
target_link_libraries(shogun-unit-test shogun gmock gtest)
set_target_properties(shogun-unit-test PROPERTIES COMPILE_DEFINITIONS "${DEFINES}")

ADD_CUSTOM_TARGET(unit-tests 
	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/shogun-unit-test
	DEPENDS shogun-unit-test)

add_test(unit-test shogun-unit-test)

unset(CMAKE_DEFINITIONS)
