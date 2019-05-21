set(BITSERY_PREFIX ${CMAKE_BINARY_DIR}/bitsery)
set(BITSERY_INCLUDE_DIR "${BITSERY_PREFIX}/src/bitsery/include")

include(ExternalProject)
ExternalProject_Add(
        bitsery
        PREFIX ${BITSERY_PREFIX}
        DOWNLOAD_DIR ${THIRD_PARTY_DIR}/bitsery
        URL https://github.com/fraillt/bitsery/archive/v4.6.0.tar.gz
        URL_MD5 accb462f98a59ed2bc4ffa7de374c24b
        INSTALL_COMMAND ""
)

LIST(APPEND SHOGUN_DEPENDS bitsery)