
include(ExternalProject)
ExternalProject_Add(
	NLopt
	PREFIX ${CMAKE_BINARY_DIR}/NLopt
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/NLopt
	URL http://ab-initio.mit.edu/nlopt/nlopt-2.3.tar.gz
	URL_MD5 811a9f1c7a7f879c7d7b4caa059eb8d6
	CONFIGURE_COMMAND <SOURCE_DIR>/configure --includedir=${THIRD_PARTY_DIR}/include/nlopt --libdir=${THIRD_PARTY_DIR}/libs/nlopt --disable-shared --with-pic
	INSTALL_COMMAND make install-libLTLIBRARIES && make -C api install-includeHEADERS
	)

SET(NLOPT_INCLUDE_DIR ${THIRD_PARTY_DIR}/include/nlopt)
SET(NLOPT_LIBRARIES ${THIRD_PARTY_DIR}/libs/nlopt/libnlopt.a)
LIST(APPEND SHOGUN_DEPENDS NLopt)
