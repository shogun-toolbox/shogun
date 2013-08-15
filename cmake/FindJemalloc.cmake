# Copyright (C) 2007-2012 Hypertable, Inc.
#
# This file is part of Hypertable.
#
# Hypertable is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or any later version.
#
# Hypertable is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Hypertable. If not, see <http://www.gnu.org/licenses/>
#

# - Find Jemalloc
# Find the native Jemalloc includes and library
#
# Jemalloc_INCLUDE_DIR - where to find Jemalloc.h, etc.
# Jemalloc_LIBRARIES - List of libraries when using Jemalloc.
# Jemalloc_FOUND - True if Jemalloc found.

find_path(Jemalloc_INCLUDE_DIR jemalloc/jemalloc.h 
  PATHS /usr/include
  /opt/local/include
  /usr/local/include
)

set(Jemalloc_NAMES jemalloc)

find_library(Jemalloc_LIBRARY NO_DEFAULT_PATH
  NAMES ${Jemalloc_NAMES}
  PATHS /usr/lib /usr/local/lib /opt/local/lib
)

if (Jemalloc_INCLUDE_DIR AND Jemalloc_LIBRARY)
  set(Jemalloc_FOUND TRUE)
  set( Jemalloc_LIBRARIES ${Jemalloc_LIBRARY} )
else ()
  set(Jemalloc_FOUND FALSE)
  set( Jemalloc_LIBRARIES )
endif ()

if (Jemalloc_FOUND)
  message(STATUS "Found Jemalloc: ${Jemalloc_LIBRARY}")
  try_run(TC_CHECK TC_CHECK_BUILD
          ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp
          ${CMAKE_SOURCE_DIR}/cmake/CheckJemalloc.cc
          CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${Jemalloc_INCLUDE_DIR}
                      -DLINK_LIBRARIES=${Jemalloc_LIBRARIES}
          OUTPUT_VARIABLE TC_TRY_OUT)

  if (TC_CHECK_BUILD AND NOT TC_CHECK STREQUAL "0")
    string(REGEX REPLACE ".*\n(Jemalloc .*)" "\\1" TC_TRY_OUT ${TC_TRY_OUT})
    message(STATUS "${TC_TRY_OUT}")
    message(FATAL_ERROR "Please fix the jemalloc installation and try again.")
    set(Jemalloc_LIBRARIES)
  endif ()
  string(REGEX REPLACE ".*\n([0-9]+[^\n]+).*" "\\1" TC_VERSION ${TC_TRY_OUT})
  if (NOT TC_VERSION MATCHES "^[0-9]+.*")
    set(TC_VERSION "unknown -- make sure it's 1.1+")
  endif ()
  message(STATUS " version: ${TC_VERSION}")
else ()
  message(STATUS "Not Found Jemalloc: ${Jemalloc_LIBRARY}")
  if (Jemalloc_FIND_REQUIRED)
    message(STATUS "Looked for Jemalloc libraries named ${Jemalloc_NAMES}.")
    message(FATAL_ERROR "Could NOT find Jemalloc library")
  endif ()
endif ()

mark_as_advanced(
  Jemalloc_LIBRARY
  Jemalloc_INCLUDE_DIR
  )