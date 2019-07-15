include(ExternalProject)
ExternalProject_Add(
	Jemalloc
	PREFIX ${CMAKE_BINARY_DIR}/Jemalloc
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/Jemalloc
	URL https://github.com/jemalloc/jemalloc/releases/download/5.1.0/jemalloc-5.1.0.tar.bz2
	URL_MD5 1f47a5aff2d323c317dfa4cf23be1ce4
	CONFIGURE_COMMAND <SOURCE_DIR>/configure --includedir=${THIRD_PARTY_DIR}/include/jemalloc --libdir=${THIRD_PARTY_DIR}/libs/jemalloc --with-jemalloc-prefix=je_ --without-export
	BUILD_COMMAND make build_lib_static
	INSTALL_COMMAND make install_include install_lib_static
)

SET(Jemalloc_INCLUDE_DIR ${THIRD_PARTY_DIR}/include/jemalloc)
SET(Jemalloc_LIBRARIES ${THIRD_PARTY_DIR}/libs/jemalloc/libjemalloc_pic.a)
ADD_SHOGUN_DEPENDENCY(Jemalloc)