# - Try to find Eigen3 lib
#
# This module supports requiring a minimum version, e.g. you can do
#   find_package(Eigen3 3.1.2)
# to require version 3.1.2 or newer of Eigen3.
#
# Once done this will define
#
#  EIGEN_FOUND - system has eigen lib with correct version
#  EIGEN_INCLUDE_DIR - the eigen include directory
#  EIGEN_VERSION - eigen version

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Copyright (c) 2008, 2009 Gael Guennebaud, <g.gael@free.fr>
# Copyright (c) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
# Redistribution and use is allowed according to the terms of the 2-clause BSD license.

macro(_eigen3_get_version)
  file(READ "${EIGEN_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" _eigen3_version_header)

  string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen3_world_version_match "${_eigen3_version_header}")
  set(EIGEN_WORLD_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen3_major_version_match "${_eigen3_version_header}")
  set(EIGEN_MAJOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen3_minor_version_match "${_eigen3_version_header}")
  set(EIGEN_MINOR_VERSION "${CMAKE_MATCH_1}")

  set(EIGEN_VERSION ${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION})

endmacro(_eigen3_get_version)

find_path(EIGEN_INCLUDE_DIR NAMES signature_of_eigen3_matrix_library
    PATHS
    ${CMAKE_INSTALL_PREFIX}/include
    PATH_SUFFIXES eigen3 eigen
  )

if(EIGEN_INCLUDE_DIR)
  _eigen3_get_version()
endif(EIGEN_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
if ( CMAKE_VERSION LESS 2.8.3 )
  find_package_handle_standard_args (Eigen3 DEFAULT_MSG EIGEN_INCLUDE_DIR VERSION_VAR EIGEN_VERSION)
else ()
  find_package_handle_standard_args (Eigen3 REQUIRED_VARS EIGEN_INCLUDE_DIR VERSION_VAR EIGEN_VERSION)
endif ()

mark_as_advanced(EIGEN_INCLUDE_DIR)
SET(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIR} CACHE PATH "The Eigen include path.")

