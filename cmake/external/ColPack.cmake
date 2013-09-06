include(ExternalProject)
ExternalProject_Add(
	COLPACK
	PREFIX ${CMAKE_BINARY_DIR}/COLPACK
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/COLPACK
	URL http://cscapes.cs.purdue.edu/download/ColPack/ColPack-1.0.9.tar.gz
	URL_MD5 54ae2daacd00a0a278d2e4fa94bba81b
	CONFIGURE_COMMAND rm <SOURCE_DIR>/config.status && <SOURCE_DIR>/configure --includedir=${THIRD_PARTY_DIR}/include/ColPack --libdir=${THIRD_PARTY_DIR}/libs/ColPack --disable-shared --with-pic
	)

SET(COLPACK_INCLUDE_DIRS ${THIRD_PARTY_DIR}/include/ColPack)
SET(COLPACK_LDFLAGS ${THIRD_PARTY_DIR}/libs/ColPack/libColPack.a)

LIST(APPEND SHOGUN_DEPENDS COLPACK)
