#
# Try to find ViennaCL library and include path.
# Once done this will define
#
# VIENNACL_FOUND
# VIENNACL_INCLUDE_DIRS
# VIENNACL_LIBRARIES
# VIENNACL_WITH_OPENCL
#


option(VIENNACL_WITH_OPENCL "Use ViennaCL with OpenCL" YES)

IF(VIENNACL_WITH_OPENCL)
	find_package(OpenCL REQUIRED)
ENDIF(VIENNACL_WITH_OPENCL)

IF (WIN32)
	set(VIENNACL_PATH_WIN32 $ENV{PROGRAMFILES}/ViennaCL CACHE PATH "ViennaCL root directory.")

	find_path(VIENNACL_INCLUDE_DIR viennacl/forwards.h
			PATHS
			${VIENNACL_PATH_WIN32}/include
			DOC "The ViennaCL include path")

	if(VIENNACL_INCLUDE_DIR)
		mark_as_advanced(FORCE VIENNACL_PATH_WIN32)
	else(VIENNACL_INCLUDE_DIR)
		mark_as_advanced(CLEAR VIENNACL_PATH_WIN32)
	endif(VIENNACL_INCLUDE_DIR)
ELSE (WIN32) #Linux
	find_path(VIENNACL_INCLUDE_DIR viennacl/forwards.h
			PATHS
			/usr/local/include
			DOC "The ViennaCL include path")
ENDIF (WIN32)

# TODO: viennacl currently doesn't store the version of itself in the header
IF (VIENNACL_INCLUDE_DIR)
	SET(CMAKE_REQUIRED_INCLUDES ${VIENNACL_INCLUDE_DIR})

	include (CheckCXXSymbolExists)
	IF (VIENNACL_WITH_OPENCL)
		IF (NOT APPLE)
			SET(CMAKE_REQUIRED_LIBRARIES OpenCL)
		ENDIF ()
	ENDIF()

	CHECK_CXX_SYMBOL_EXISTS("viennacl::ocl::type_to_string<char>::apply" "viennacl/ocl/utils.hpp" HAVE_VIENNACL_TYPE_TO_STRING)
	if (EXISTS "${VIENNACL_INCLUDE_DIR}/viennacl/version.hpp")
		# try to read version.hpp
		file(READ "${VIENNACL_INCLUDE_DIR}/viennacl/version.hpp" _viennacl_version_header)
		string(REGEX MATCH "define[ \t]+VIENNACL_MAJOR_VERSION[ \t]+([0-9]+)" _viennacl_major_version_match "${_viennacl_version_header}")
		SET(VIENNACL_MAJOR_VERSION "${CMAKE_MATCH_1}")
		string(REGEX MATCH "define[ \t]+VIENNACL_MINOR_VERSION[ \t]+([0-9]+)" _viennacl_minor_version_match "${_viennacl_version_header}")
		SET(VIENNACL_MINOR_VERSION "${CMAKE_MATCH_1}")
		string(REGEX MATCH "define[ \t]+VIENNACL_PATCH_VERSION[ \t]+([0-9]+)" _viennacl_patch_version_match "${_viennacl_version_header}")
		SET(VIENNACL_PATCH_VERSION "${CMAKE_MATCH_1}")

		SET(VIENNACL_VERSION ${VIENNACL_MAJOR_VERSION}.${VIENNACL_MINOR_VERSION}.${VIENNACL_PATCH_VERSION}
			CACHE STRING "ViennaCL version" FORCE)
		MATH(EXPR VIENNACL_ENCODED_VERSION ${VIENNACL_MAJOR_VERSION}*10000+${VIENNACL_MINOR_VERSION}*100+${VIENNACL_PATCH_VERSION})
	elseif (HAVE_VIENNACL_TYPE_TO_STRING)
		SET(VIENNACL_VERSION "1.5.0" CACHE STRING "ViennaCL version" FORCE)
		SET(VIENNACL_ENCODED_VERSION 10500)
	else ()
		SET(VIENNACL_VERSION "1.4.2" CACHE STRING "ViennaCL version" FORCE)
		SET(VIENNACL_ENCODED_VERSION 10402)
	endif ()
ENDIF()

include(FindPackageHandleStandardArgs)
if(VIENNACL_WITH_OPENCL)
	set(VIENNACL_INCLUDE_DIRS ${VIENNACL_INCLUDE_DIR} ${OPENCL_INCLUDE_DIRS})
	set(VIENNACL_LIBRARIES ${OPENCL_LIBRARIES})
	find_package_handle_standard_args(ViennaCL REQUIRED_VARS
			VIENNACL_INCLUDE_DIR OPENCL_INCLUDE_DIRS OPENCL_LIBRARIES
			VERSION_VAR VIENNACL_VERSION)
else(VIENNACL_WITH_OPENCL)
	set(VIENNACL_INCLUDE_DIRS ${VIENNACL_INCLUDE_DIR})
	set(VIENNACL_LIBRARIES "")
	find_package_handle_standard_args(ViennaCL REQUIRED_VARS VIENNACL_INCLUDE_DIR VERSION_VAR VIENNACL_VERSION)
endif(VIENNACL_WITH_OPENCL)

MARK_AS_ADVANCED(VIENNACL_INCLUDE_DIR)
