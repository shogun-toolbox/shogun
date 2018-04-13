include(ExternalProject)
ExternalProject_Add(
        rxcpp
        PREFIX ${CMAKE_BINARY_DIR}/rxcpp
        DOWNLOAD_DIR ${THIRD_PARTY_DIR}/rxcpp
        URL https://github.com/Reactive-Extensions/RxCpp/archive/v4.0.0.tar.gz
        URL_MD5 feb89934f465bb5ac513c9adce8d3b1b
        CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo
        -DCMAKE_INSTALL_PREFIX:STRING=${THIRD_PARTY_INCLUDE_DIR}
        -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
        BUILD_COMMAND ""
        )

add_dependencies(libshogun rxcpp)

set(rxcpp_INCLUDE_DIR ${THIRD_PARTY_INCLUDE_DIR})

UNSET(C_COMPILER)
UNSET(CXX_COMPILER)