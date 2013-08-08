find_program(CCACHE
             NAMES ccache)

# handle REQUIRED and QUIET options
include(FindPackageHandleStandardArgs)
if (CMAKE_VERSION LESS 2.8.3)
  find_package_handle_standard_args(CCache DEFAULT_MSG CCACHE)
else ()
  find_package_handle_standard_args(CCache REQUIRED_VARS CCACHE)
endif ()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	SET(CCACHE_FLAGS "-Qunused-arguments -fcolor-diagnostics")
	#set env CCACHE_CPP2=yes
	SET(ENV{CCACHE_CPP} YES)
endif()

mark_as_advanced (
	CCACHE
	CCACHE_FLAGS
)
