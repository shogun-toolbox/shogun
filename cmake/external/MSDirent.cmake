
include(ExternalProject)
ExternalProject_Add(
	MSDirent
	PREFIX ${CMAKE_BINARY_DIR}/MSDirent
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/MSDirent
	URL http://www.softagalleria.net/download/dirent/dirent-1.13.zip
	URL_MD5 4a4fdd27848dde028bd5e9712e2636bc
	INSTALL_COMMAND ""
	BUILD_COMMAND ""
	CONFIGURE_COMMAND ""
)

ExternalProject_Get_Property(MSDirent SOURCE_DIR)
SET(MSDIRENT_INCLUDE_DIR ${SOURCE_DIR}/include)
LIST(APPEND SHOGUN_DEPENDS MSDirent)
