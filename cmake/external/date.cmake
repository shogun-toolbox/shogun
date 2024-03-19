include(ExternalProject)
set(DATE_PREFIX ${CMAKE_BINARY_DIR}/date)
set(DATE_SOURCE_DIR "${THIRD_PARTY_DIR}/date")
set(DATE_INCLUDE_DIR "${DATE_SOURCE_DIR}/include")
ExternalProject_Add(
        date
        PREFIX ${DATE_PREFIX}
        SOURCE_DIR ${DATE_SOURCE_DIR}
        GIT_REPOSITORY https://github.com/HowardHinnant/date.git
        GIT_TAG e7e1482087f58913b80a20b04d5c58d9d6d90155
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${DATE_INCLUDE_DIR}/ ${THIRD_PARTY_INCLUDE_DIR}/
)

add_dependencies(libshogun date)