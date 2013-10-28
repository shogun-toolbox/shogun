
include(ExternalProject)
ExternalProject_Add(
	ARPREC
	PREFIX ${CMAKE_BINARY_DIR}/ARPREC
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/ARPREC
	URL http://crd.lbl.gov/~dhbailey/mpdist/arprec-2.2.16.tar.gz
	URL_MD5 88310502ca2b8e76d3b265475e401bbc
	CONFIGURE_COMMAND <SOURCE_DIR>/configure --includedir=${THIRD_PARTY_DIR}/include/arprec --libdir=${THIRD_PARTY_DIR}/libs/arprec --disable-shared --with-pic
	INSTALL_COMMAND make -C src install-libLTLIBRARIES && make -C include install-data-am
	)

SET(ARPREC_INCLUDE_DIR ${THIRD_PARTY_DIR}/include/arprec)
SET(ARPREC_LIBRARIES ${THIRD_PARTY_DIR}/libs/arprec/libarprec.a)
LIST(APPEND SHOGUN_DEPENDS ARPREC)
