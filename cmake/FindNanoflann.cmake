# - Try to find Nanoflann lib
#
# This module supports requiring a minimum version, e.g. you can do
#   find_package(Nanoflann 1.1.7)
# to require version 1.1.7 or newer of Nanoflann.
#
# Once done this will define
#
#  NANOFLANN_FOUND - system has Nanoflann lib with correct version
#  NANOFLANN_INCLUDE_DIR - the Nanoflann include directory
#  NANOFLANN_VERSION - Nanoflann version

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Copyright (c) 2008, 2009 Gael Guennebaud, <g.gael@free.fr>
# Copyright (c) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
# Redistribution and use is allowed according to the terms of the 2-clause BSD license.

macro(_nanoflann_get_version)
  file(READ "${NANOFLANN_INCLUDE_DIR}/nanoflann.hpp" _nanoflann_version_header)
  string(REGEX REPLACE ".*[ \t]+.define[ \t]+NANOFLANN_VERSION[ \t]+0x(.).*" "\\1" _nanoflann_major "${_nanoflann_version_header}")
  string(REGEX REPLACE ".*[ \t]+.define[ \t]+NANOFLANN_VERSION[ \t]+0x.(.).*" "\\1" _nanoflann_minor "${_nanoflann_version_header}")
  string(REGEX REPLACE ".*[ \t]+.define[ \t]+NANOFLANN_VERSION[ \t]+0x..(.).*" "\\1" _nanoflann_path "${_nanoflann_version_header}")
  
  set(NANOFLANN_VERSION ${_nanoflann_major}.${_nanoflann_minor}.${_nanoflann_path})

endmacro(_nanoflann_get_version)

find_path(NANOFLANN_INCLUDE_DIR NAMES nanoflann.hpp
    PATHS
    ${CMAKE_INSTALL_PREFIX}/include
  )

if(NANOFLANN_INCLUDE_DIR)
  _nanoflann_get_version()
endif(NANOFLANN_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
if ( CMAKE_VERSION LESS 2.8.3 )
  find_package_handle_standard_args (Nanoflann DEFAULT_MSG NANOFLANN_INCLUDE_DIR VERSION_VAR NANOFLANN_VERSION)
else ()
  find_package_handle_standard_args (Nanoflann REQUIRED_VARS NANOFLANN_INCLUDE_DIR VERSION_VAR NANOFLANN_VERSION)
endif ()

mark_as_advanced(NANOFLANN_INCLUDE_DIR)
SET(NANOFLANN_INCLUDE_DIRS ${NANOFLANN_INCLUDE_DIR} CACHE PATH "The Nanoflann include path.")

