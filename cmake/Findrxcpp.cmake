# Copyright Gonzalo Brito Gadeschi 2015
# Copyright Kirk Shoop 2016
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)
#
# Find the rxcpp include directory
# The following variables are set if rxcpp is found.
#  rxcpp_FOUND        - True when the rxcpp include directory is found.
#  rxcpp_INCLUDE_DIR  - The path to where the meta include files are.
# If rxcpp is not found, rxcpp_FOUND is set to false.

# https://github.com/gnzlbg/ndtree/blob/master/cmake/Findrange-v3.cmake

find_package(PkgConfig)

if(NOT EXISTS "${rxcpp_INCLUDE_DIR}")
    find_path(rxcpp_INCLUDE_DIR
            NAMES rxcpp/rx.hpp
            DOC "rxcpp library header files"
            )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rxcpp REQUIRED_VARS rxcpp_INCLUDE_DIR)
