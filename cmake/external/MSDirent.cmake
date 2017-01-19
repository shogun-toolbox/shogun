
include(ExternalProject)
ExternalProject_Add(
	MSDirent
	PREFIX ${CMAKE_BINARY_DIR}/MSDirent
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/MSDirent
	GIT_REPOSITORY https://github.com/tronkko/dirent.git
	GIT_TAG 8b1db5092479a73d47eafd3de739b27e876e6bf3
	INSTALL_COMMAND ""
	BUILD_COMMAND ""
	CONFIGURE_COMMAND ""
)

ExternalProject_Get_Property(MSDirent SOURCE_DIR)
SET(MSDIRENT_INCLUDE_DIR ${SOURCE_DIR}/include)
LIST(APPEND SHOGUN_DEPENDS MSDirent)
