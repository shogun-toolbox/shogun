GetCompilers()

# TODO: there's a problem with compiling c++11 and tcmalloc
# fix this via a simple patch and add it to the repo
include(ExternalProject)
ExternalProject_Add(
	TCMalloc
	PREFIX ${CMAKE_BINARY_DIR}/TCMalloc
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/TCMalloc
	URL http://gperftools.googlecode.com/files/gperftools-2.1.tar.gz
	URL_MD5 5e5a981caf9baa9b4afe90a82dcf9882
	CONFIGURE_COMMAND CXX=${CXX_COMPILER} <SOURCE_DIR>/configure --includedir=${THIRD_PARTY_DIR}/include/tcmalloc --libdir=${THIRD_PARTY_DIR}/libs/tcmalloc --disable-shared --enable-minimal
	BUILD_COMMAND make libtcmalloc_minimal.la CXXFLAGS=${CXX11_COMPILER_FLAGS}
	INSTALL_COMMAND make install-libLTLIBRARIES install-nodist_perftoolsincludeHEADERS
	#PATCH_COMMAND patch -p0 < ${CMAKE_SOURCE_DIR}/tcmalloc-cxx11_pthread_fix.patch
	)

SET(TCMalloc_INCLUDE_DIR ${THIRD_PARTY_DIR}/include/tcmalloc)
SET(TCMalloc_LIBRARIES ${THIRD_PARTY_DIR}/libs/tcmalloc/libtcmalloc_minimal.a)
LIST(APPEND SHOGUN_DEPENDS TCMalloc)
