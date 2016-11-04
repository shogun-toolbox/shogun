# Midified from https://llvm.org/svn/llvm-project/llvm/trunk/cmake/modules/FindSphinx.cmake
# under "UIUC" BSD-Style license.
# CMake find_package() Module for pandoc markdown converter, http://pandoc.org/
#
# Example usage:
#
# find_package(Pandoc)
#
# If successful the following variables will be defined
# PANDOC_FOUND
# PANDOC_EXECUTABLE

find_program(PANDOC_EXECUTABLE
             NAMES pandoc
             PATH /usr/local/bin
             DOC "Path to pandoc executable")

# Handle REQUIRED and QUIET arguments
# this will also set PANDOC_FOUND to true if PANDOC_EXECUTABLE exists
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pandoc
                                  "Failed to locate pandoc executable"
                                  PANDOC_EXECUTABLE)

