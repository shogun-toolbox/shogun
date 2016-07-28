# Author: Matt Langston
# Created: 2014.09.03
#
# Try to find JavaScriptCore. Once done this will define:
#
#  JavaScriptCore_FOUND       - system has JavaScriptCore
#  JavaScriptCore_INCLUDE_DIRS - the include directory
#  JavaScriptCore_LIBRARY_DIR - the directory containing the library
#  JavaScriptCore_LIBRARIES   - link these to use JavaScriptCore


find_package(PkgConfig)

pkg_check_modules(PC_JavaScriptCore QUIET JavaScriptCore)

find_path(JavaScriptCore_INCLUDE_DIRS
  NAMES JavaScriptCore/JavaScript.h
  HINTS ${PC_JavaScriptCore_INCLUDE_DIRS} ${PC_JavaScriptCore_INCLUDEDIR}
  PATHS ENV JavaScriptCore_HOME
  PATH_SUFFIXES includes
  )

set(JavaScriptCore_ARCH "x86")
if(CMAKE_GENERATOR MATCHES "^Visual Studio .+ ARM$")
  set(JavaScriptCore_ARCH "arm")
endif()

find_library(JavaScriptCore_LIBRARIES
  NAMES JavaScriptCore JavaScriptCore-Debug JavaScriptCore-Release
  HINTS ${PC_JavaScriptCore_LIBRARY_DIRS} ${PC_JavaScriptCore_LIBDIR}
  PATHS ENV JavaScriptCore_HOME
  PATH_SUFFIXES ${JavaScriptCore_ARCH}
  )

if(NOT JavaScriptCore_LIBRARIES MATCHES ".+-NOTFOUND")
  get_filename_component(JavaScriptCore_LIBRARY_DIR ${JavaScriptCore_LIBRARIES} DIRECTORY)

  # If we found the JavaScriptCore library and we're using a Visual
  # Studio generator and we're targeting either WindowsStore or
  # WindowsPhone, then allow Visual Studio to use both the
  # JavaScriptCore-Debug.lib and JavaScriptCore-Release.lib if they
  # exist.
  if(CMAKE_GENERATOR MATCHES "^Visual Studio .+" AND CMAKE_SYSTEM_NAME MATCHES "^Windows(Store|Phone)")
    string(REGEX REPLACE "-(Debug|Release)" "-$(Configuration)" JavaScriptCore_LIBRARIES ${JavaScriptCore_LIBRARIES})
  endif()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JavaScriptCore DEFAULT_MSG JavaScriptCore_INCLUDE_DIRS JavaScriptCore_LIBRARIES)
