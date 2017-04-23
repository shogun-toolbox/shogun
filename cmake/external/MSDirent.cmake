set(PREFIX ${CMAKE_BINARY_DIR}/MSDirent)
include(ExternalProject)
ExternalProject_Add(
	MSDirent
	PREFIX ${PREFIX}
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/MSDirent
	GIT_REPOSITORY https://github.com/tronkko/dirent.git
	GIT_TAG 8b1db5092479a73d47eafd3de739b27e876e6bf3
	INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PREFIX}/src/MSDirent/include/dirent.h ${THIRD_PARTY_INCLUDE_DIR}/MSDirent/dirent.h
	BUILD_COMMAND ""
	CONFIGURE_COMMAND ""
)

ExternalProject_Get_Property(MSDirent SOURCE_DIR)
SET(MSDIRENT_INCLUDE_DIR ${SOURCE_DIR}/include)
add_dependencies(libshogun MSDirent)
