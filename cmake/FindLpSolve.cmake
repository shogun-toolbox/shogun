find_path(
	LPSOLVE_INCLUDE_DIR
	NAMES lpsolve/lp_lib.h
	HINTS
	PATH_SUFFIXES include
	PATHS
	/usr/local
	/usr
	/opt/local
)
find_library(
	LPSOLVE_LIBRARIES
	NAMES lpsolve55
	HINTS
	PATH_SUFFIXES lib64 lib
	PATHS
	/usr/local
	/usr
	/opt/local
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LPSOLVE DEFAULT_MSG LPSOLVE_LIBRARIES LPSOLVE_INCLUDE_DIR)
mark_as_advanced(LPSOLVE_INCLUDE_DIR LPSOLVE_LIBRARIES)