#.rst:
# FindArrow.cmake
# -------------
#
# Find a Arrow installation.
#
# This module finds if Arrow is installed and selects a default
# configuration to use.
#
# find_package(Arrow ...)
#
#
# The following variables control which libraries are found::
#
#   Arrow_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   Arrow_FOUND            - Set to TRUE if Arrow was found.
#   Arrow_LIBRARIES        - Path to the Arrow libraries.
#   Arrow_LIBRARY_DIR      - compile time link directories
#   Arrow_INCLUDE_DIR      - compile time include directories
#
#
# Sample usage:
#
# ::
#
#    find_package(Arrow)
#    if(Arrow_FOUND)
#      target_link_libraries(<YourTarget> ${Arrow_LIBRARIES})
#    endif()

if(Arrow_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()


find_library(Arrow_LIBRARY
  NAMES arrow
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(Arrow_GPU_LIBRARY
  NAMES arrow_gpu
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

get_filename_component(Arrow_LIBRARY_DIR ${Arrow_LIBRARY} DIRECTORY)

if(Arrow_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

# Set standard CMake FindPackage variables if found.
set(ARROW_LIBRARIES ${Arrow_LIBRARY})
set(ARROW_GPU_LIBRARIES ${Arrow_GPU_LIBRARY})
set(ARROW_LIBRARY_DIR ${Arrow_LIBRARY_DIR})
set(ARROW_INCLUDE_DIR ${Arrow_LIBRARY_DIR}/../include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Arrow REQUIRED_VARS Arrow_LIBRARY)
