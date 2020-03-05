SET(spdlog_SOURCE_DIR ${THIRD_PARTY_DIR}/spdlog)
SET(spdlog_INCLUDE_DIR ${THIRD_PARTY_INCLUDE_DIR})
set(spdlog_PREFIX "${CMAKE_BINARY_DIR}/spdlog")
set(spdlog_DIR "${spdlog_PREFIX}/src/SpdLog-build")
include(ExternalProject)
ExternalProject_Add(
	SpdLog
	PREFIX ${spdlog_PREFIX}
	DOWNLOAD_DIR ${spdlog_SOURCE_DIR}
	SOURCE_DIR ${spdlog_SOURCE_DIR}
	URL https://github.com/gabime/spdlog/archive/v1.5.0.tar.gz
	URL_MD5 a966eea01f81551527853d282896cb4d
	CMAKE_ARGS
	-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
	-DCMAKE_POSITION_INDEPENDENT_CODE=ON
	-DSPDLOG_BUILD_EXAMPLE=OFF
	-DSPDLOG_BUILD_BENCH=OFF
	-DSPDLOG_BUILD_TESTS=OFF
	INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${spdlog_SOURCE_DIR}/include ${spdlog_INCLUDE_DIR}
    BUILD_BYPRODUCTS ${spdlog_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}spdlogd${CMAKE_STATIC_LIBRARY_SUFFIX}
)

file(MAKE_DIRECTORY ${spdlog_INCLUDE_DIR})

add_library(spdlog IMPORTED STATIC GLOBAL)
set_target_properties(spdlog PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${spdlog_INCLUDE_DIR}")
if(CMAKE_GENERATOR MATCHES "Visual Studio.*" OR CMAKE_GENERATOR STREQUAL Xcode)
        set_target_properties(spdlog PROPERTIES
                IMPORTED_LOCATION_DEBUG "${spdlog_DIR}/Debug/${CMAKE_STATIC_LIBRARY_PREFIX}spdlogd${CMAKE_STATIC_LIBRARY_SUFFIX}"
                IMPORTED_LOCATION_RELEASE "${spdlog_DIR}/Release/${CMAKE_STATIC_LIBRARY_PREFIX}spdlog${CMAKE_STATIC_LIBRARY_SUFFIX}")
else()
        set_target_properties(spdlog PROPERTIES
                IMPORTED_LOCATION_DEBUG "${spdlog_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}spdlogd${CMAKE_STATIC_LIBRARY_SUFFIX}"
                IMPORTED_LOCATION_RELEASE "${spdlog_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}spdlog${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

add_dependencies(spdlog SpdLog)
add_dependencies(libshogun spdlog)
