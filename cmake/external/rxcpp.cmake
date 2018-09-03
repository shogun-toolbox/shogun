include(ExternalProject)
ExternalProject_Add(
        rxcpp
        PREFIX ${CMAKE_BINARY_DIR}/rxcpp
        DOWNLOAD_DIR ${THIRD_PARTY_DIR}/rxcpp
        URL https://github.com/ReactiveX/RxCpp/archive/4.1.0.tar.gz
        URL_MD5 6c283f36ce251f45146f7099aa9ef19a
        CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo
        -DCMAKE_INSTALL_PREFIX:STRING=${THIRD_PARTY_INCLUDE_DIR}
        -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
        BUILD_COMMAND ""
        )

add_dependencies(libshogun rxcpp)

set(rxcpp_INCLUDE_DIR ${THIRD_PARTY_INCLUDE_DIR}/include)

UNSET(C_COMPILER)
UNSET(CXX_COMPILER)
