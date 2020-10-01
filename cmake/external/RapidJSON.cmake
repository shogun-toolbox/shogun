set(RAPIDJSON_PREFIX ${CMAKE_BINARY_DIR}/RapidJSON)
set(RAPIDJSON_INCLUDE_DIRS "${RAPIDJSON_PREFIX}/src/RapidJSON/include")
include(ExternalProject)
ExternalProject_Add(
	RapidJSON
	PREFIX ${RAPIDJSON_PREFIX}
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/RapidJSON
	URL https://github.com/Tencent/rapidjson/archive/v1.1.0.tar.gz
	URL_MD5 badd12c511e081fec6c89c43a7027bce
	CMAKE_ARGS -DRAPIDJSON_BUILD_DOC:BOOL=OFF
		-DRAPIDJSON_BUILD_EXAMPLES:BOOL=OFF
		-DRAPIDJSON_BUILD_TESTS:BOOL=OFF
    INSTALL_COMMAND ""
	)

set(RapidJSON_FOUND FALSE CACHE INTERNAL BOOL)
ADD_SHOGUN_DEPENDENCY(RapidJSON)
