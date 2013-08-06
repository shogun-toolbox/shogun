include(ExternalProject)
ExternalProject_Add(
	JSON
	PREFIX ${CMAKE_BINARY_DIR}/JSON
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/JSON
	URL http://s3.amazonaws.com/json-c_releases/releases/json-c-0.11-nodoc.tar.gz
	URL_MD5 4ac9dae7cc2975dba7bc04b4c0b98953
	CONFIGURE_COMMAND rm <SOURCE_DIR>/config.status && <SOURCE_DIR>/configure --includedir=${THIRD_PARTY_DIR}/include/json --libdir=${THIRD_PARTY_DIR}/libs/json --disable-shared
	)

SET(JSON_INCLUDE_DIRS ${THIRD_PARTY_DIR}/include/json/json-c)
SET(JSON_LDFLAGS ${THIRD_PARTY_DIR}/libs/json/libjson-c.a)
