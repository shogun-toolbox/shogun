# The module defines the following variables:
#   CTAGS_EXECUTABLE - path to ctags command line client
#   CTAGS_FOUND - true if the command line client was found
#   CTAGS_VERSION_STRING - the version of ctags found (since CMake 2.8.8)
# Example usage:
#   find_package( Ctags )
#   if( CTAGS_FOUND )
#     message("ctags found: ${CTAGS_EXECUTABLE}")
#   endif()

find_program( CTAGS_EXECUTABLE
    NAMES ctags
    DOC "ctags executable"
)

if( CTAGS_EXECUTABLE )
    execute_process(COMMAND ${CTAGS_EXECUTABLE} --version
        OUTPUT_VARIABLE ctags_version
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (ctags_version STREQUAL "")
        unset(CTAGS_EXECUTABLE CACHE)
        MESSAGE(STATUS "The ctags found is not suitable for meta examples.")
    else()
        string(TOLOWER ${ctags_version} ctags_version)
        if(NOT ctags_version MATCHES exuberant)
            set(CTAGS_FLAVOR "GNU")
        else()
            set(CTAGS_FLAVOR "Exuberant")
            string( REPLACE "Exuberant Ctags " "" CTAGS_VERSION_STRING "${ctags_version}" )
            string( REGEX REPLACE ",.*$" "" CTAGS_VERSION_STRING ${CTAGS_VERSION_STRING} )
        endif()
        set(CTAGS_FLAVOR ${CTAGS_FLAVOR} CACHE STRING "Ctags executable flavour" FORCE)
        message(STATUS "Ctags flavour: ${CTAGS_FLAVOR}")

        unset( ctags_version )
        mark_as_advanced( CTAGS_EXECUTABLE )
    endif()
endif()

# Handle the QUIETLY and REQUIRED arguments and set CTAGS_FOUND to TRUE if
# all listed variables are TRUE

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Ctags
    REQUIRED_VARS CTAGS_EXECUTABLE
    VERSION_VAR CTAGS_VERSION_STRING
)
