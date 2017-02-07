find_program(CCACHE
             NAMES ccache)

find_program(CCACHE_SWIG
             NAMES ccache-swig)

if (CCACHE)
    execute_process(COMMAND ${CCACHE} --version ERROR_QUIET OUTPUT_VARIABLE CCACHE_VERSION_OUT)
    string(REGEX MATCH "ccache[ \t]+version[ \t]+([0-9.]+)" _ccache_version "${CCACHE_VERSION_OUT}")
    SET(CCACHE_VERSION "${CMAKE_MATCH_1}")
endif()

# handle REQUIRED and QUIET options
include(FindPackageHandleStandardArgs)
if (CMAKE_VERSION LESS 2.8.3)
  find_package_handle_standard_args(CCache DEFAULT_MSG CCACHE CCACHE_VERSION)
else ()
  find_package_handle_standard_args(CCache REQUIRED_VARS CCACHE CCACHE_VERSION)
endif ()

if (CCACHE_FOUND)
	if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
		SET(CCACHE_FLAGS "-Qunused-arguments -fcolor-diagnostics")
		#set env CCACHE_CPP2=yes
		SET(ENV{CCACHE_CPP} YES)
	endif()
endif()

mark_as_advanced (
	CCACHE
	CCACHE_FLAGS
	CCACHE_SWIG
)
