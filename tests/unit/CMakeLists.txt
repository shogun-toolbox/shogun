add_subdirectory(${GOOGLE_MOCK_SOURCE_DIR} tests/unit EXCLUDE_FROM_ALL)

include_directories(${INCLUDES} ${CMAKE_SOURCE_DIR}/src 
					${GOOGLE_MOCK_SOURCE_DIR}/include
					${GOOGLE_MOCK_SOURCE_DIR}/gtest/include)

set_target_properties(gmock PROPERTIES COMPILE_DEFINITIONS "${DEFINES}")
set_target_properties(gtest PROPERTIES COMPILE_DEFINITIONS "${DEFINES}")

FILE(GLOB_RECURSE UNITTEST_SRC *_unittest.cc)

ADD_EXECUTABLE(shogun-unit-test EXCLUDE_FROM_ALL ${UNITTEST_SRC})
target_link_libraries(shogun-unit-test shogun gmock gtest)
set_target_properties(shogun-unit-test PROPERTIES COMPILE_DEFINITIONS "${DEFINES}")

ADD_CUSTOM_TARGET(unit-tests 
	COMMAND ./shogun-unit-test
	DEPENDS shogun-unit-test)

add_test(unit-tests shogun-unit-test)
