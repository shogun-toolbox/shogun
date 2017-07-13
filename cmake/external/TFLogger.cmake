GetCompilers()

include(ExternalProject)
ExternalProject_Add(
        rxcpp
        PREFIX ${CMAKE_BINARY_DIR}/tflogger
        DOWNLOAD_DIR ${THIRD_PARTY_DIR}/tflogger
        URL https://github.com/shogun-toolbox/tflogger/archive/master.zip
        CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_BINARY_DIR}/src/shogun/lib/external
        -DCMAKE_C_COMPILER:STRING=${C_COMPILER}
        -DCMAKE_CXX_COMPILER:STRING=${CXX_COMPILER}
        BUILD_COMMAND ""
        )

add_dependencies(libshogun tflogger)

set(TFLogger_INCLUDE_DIR ${THIRD_PARTY_INCLUDE_DIR})

UNSET(C_COMPILER)
UNSET(CXX_COMPILER)
