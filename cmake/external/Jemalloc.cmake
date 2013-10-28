include(ExternalProject)
ExternalProject_Add(
	JeMalloc
	PREFIX ${CMAKE_BINARY_DIR}/JeMalloc
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/JeMalloc
	URL http://www.canonware.com/download/jemalloc/jemalloc-3.4.0.tar.bz2
	URL_MD5 c4fa3da0096d5280924a5f7ebc8dbb1c
	CONFIGURE_COMMAND <SOURCE_DIR>/configure --includedir=${THIRD_PARTY_DIR}/include/jemalloc --libdir=${THIRD_PARTY_DIR}/libs/jemalloc --with-jemalloc-prefix=je_ --without-export
	BUILD_COMMAND make build_lib_static
	INSTALL_COMMAND make install_include install_lib_static
	)

SET(Jemalloc_INCLUDE_DIR ${THIRD_PARTY_DIR}/include/jemalloc)
SET(Jemalloc_LIBRARIES ${THIRD_PARTY_DIR}/libs/jemalloc/libjemalloc_pic.a)
LIST(APPEND SHOGUN_DEPENDS JeMalloc)
