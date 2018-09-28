include(ExternalProject)
SET(RXCPP_SOURCE_DIR ${THIRD_PARTY_DIR}/rxcpp)
ExternalProject_Add(
        rxcpp
        PREFIX ${CMAKE_BINARY_DIR}/rxcpp
        DOWNLOAD_DIR ${RXCPP_SOURCE_DIR}
        SOURCE_DIR ${RXCPP_SOURCE_DIR}/src
        URL https://github.com/ReactiveX/RxCpp/archive/4.1.0.tar.gz
        URL_MD5 6c283f36ce251f45146f7099aa9ef19a
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${RXCPP_SOURCE_DIR}/src/Rx/v2/src/ ${THIRD_PARTY_INCLUDE_DIR}/
        )

add_dependencies(libshogun rxcpp)

set(rxcpp_INCLUDE_DIR ${THIRD_PARTY_INCLUDE_DIR})
