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
mark_as_advanced( CTAGS_EXECUTABLE )

if( CTAGS_EXECUTABLE )
    execute_process(COMMAND ${CTAGS_EXECUTABLE} --version
        OUTPUT_VARIABLE ctags_version
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if( ctags_version MATCHES "^Exuberant Ctags [0-9]" )
        string( REPLACE "Exuberant Ctags " "" CTAGS_VERSION_STRING "${ctags_version}" )
        string( REGEX REPLACE ",.*$" "" CTAGS_VERSION_STRING ${CTAGS_VERSION_STRING} )
    endif()

    unset( ctags_version )
endif()

# Handle the QUIETLY and REQUIRED arguments and set CTAGS_FOUND to TRUE if
# all listed variables are TRUE

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Ctags
    REQUIRED_VARS CTAGS_EXECUTABLE
    VERSION_VAR CTAGS_VERSION_STRING
)
