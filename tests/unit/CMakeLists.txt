# transform defines to -D<definition> string
foreach(D IN LISTS DEFINES)
	SET(CMAKE_DEFINITIONS "${CMAKE_DEFINITIONS} -D${D}")
endforeach()

# Find GTEST and GMOCK frameworks
include(ExternalProject)
SET(GOOGLE_MOCK_SOURCE_DIR "" CACHE PATH "Path to the GMock source")
IF ("${GOOGLE_MOCK_SOURCE_DIR}" STREQUAL "" OR NOT EXISTS "${GOOGLE_MOCK_SOURCE_DIR}/CMakeLists.txt")
	ExternalProject_Add(
		GoogleMock
		PREFIX ${CMAKE_BINARY_DIR}/GoogleMock
		DOWNLOAD_DIR ${THIRD_PARTY_DIR}/GoogleMock
		URL http://googlemock.googlecode.com/files/gmock-1.6.0.zip
		URL_MD5 f547f47321ca88d3965ca2efdcc2a3c1
		INSTALL_COMMAND ""
		CMAKE_ARGS 	-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${THIRD_PARTY_DIR}/libs/gmock
			-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${THIRD_PARTY_DIR}/libs/gmock
			-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}${CMAKE_DEFINITIONS}
			-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}${CMAKE_DEFINITIONS}
	)
ELSE()
	ExternalProject_Add(
		GoogleMock
		PREFIX ${CMAKE_BINARY_DIR}/GoogleMock
		SOURCE_DIR ${GOOGLE_MOCK_SOURCE_DIR}
		INSTALL_COMMAND ""
		CMAKE_ARGS 	-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${THIRD_PARTY_DIR}/libs/gmock
			-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${THIRD_PARTY_DIR}/libs/gmock
			-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}${CMAKE_DEFINITIONS}
			-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}${CMAKE_DEFINITIONS}
	)		
ENDIF()
ExternalProject_Get_Property(GoogleMock source_dir)

include_directories(${INCLUDES} ${CMAKE_SOURCE_DIR}/src 
		${source_dir}/include ${source_dir}/gtest/include)
LINK_DIRECTORIES(${THIRD_PARTY_DIR}/libs/gmock)

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
