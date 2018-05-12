include(ExternalProject)
ExternalProject_Add(
	TCMalloc
	PREFIX ${CMAKE_BINARY_DIR}/TCMalloc
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/TCMalloc
	URL https://github.com/gperftools/gperftools/releases/download/gperftools-2.7/gperftools-2.7.tar.gz
	URL_MD5 c6a852a817e9160c79bdb2d3101b4601
	CONFIGURE_COMMAND CXX=${CMAKE_CXX_COMPILER} <SOURCE_DIR>/configure --includedir=${THIRD_PARTY_DIR}/include/tcmalloc --libdir=${THIRD_PARTY_DIR}/libs/tcmalloc --disable-shared --enable-minimal
	BUILD_COMMAND make libtcmalloc_minimal.la CXXFLAGS=${CXX11_COMPILER_FLAGS}
	INSTALL_COMMAND make install-libLTLIBRARIES install-nodist_perftoolsincludeHEADERS
	)

SET(TCMalloc_INCLUDE_DIR ${THIRD_PARTY_DIR}/include/tcmalloc)
SET(TCMalloc_LIBRARIES ${THIRD_PARTY_DIR}/libs/tcmalloc/libtcmalloc_minimal.a)
add_dependencies(libshogun TCMalloc)