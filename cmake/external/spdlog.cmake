SET(spdlog_SOURCE_DIR ${THIRD_PARTY_DIR}/spdlog)
SET(spdlog_INCLUDE_DIR ${THIRD_PARTY_INCLUDE_DIR})
include(ExternalProject)
ExternalProject_Add(
	spdlog
	PREFIX ${CMAKE_BINARY_DIR}/spdlog
	DOWNLOAD_DIR ${spdlog_SOURCE_DIR}
	SOURCE_DIR ${spdlog_SOURCE_DIR}
	URL https://github.com/gabime/spdlog/archive/v1.3.1.tar.gz
	URL_MD5 3c17dd6983de2a66eca8b5a0b213d29f
	CMAKE_ARGS
	-DCMAKE_BUILD_TYPE=Release
	-DSPDLOG_BUILD_EXAMPLES=OFF
	-DSPDLOG_BUILD_BENCH=OFF
	-DSPDLOG_BUILD_TESTS=OFF
	BUILD_COMMAND ""
	INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${spdlog_SOURCE_DIR}/include ${spdlog_INCLUDE_DIR}
)

add_dependencies(libshogun spdlog)
