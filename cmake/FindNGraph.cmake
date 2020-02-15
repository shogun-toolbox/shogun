# - Find NGraph
# Find the native NGraph headers and libraries.
#
#  NGraph_INCLUDE_DIRS - Where to find <NGraph/NGraphHeaders.h>, etc.
#  NGraph_FOUND        - True if NGraph found.
#  NGraph_LIBRARY_DIRS - Where to find the library files

# Look for the header files
FIND_PATH(NGraph_INCLUDE_DIRS NAMES ngraph/ngraph.hpp
          PATHS /usr/include
               /usr/local/include
               /usr/local/include
               /opt/local/include
	  HINTS $ENV{NGRAPH_DIR}/include
          PATH_SUFFIXES ngraph)

# Look for the libraries
find_library(NGraph_LIBRARIES NAMES ngraph
            PATHS
                   /usr/local/lib
                   /opt/local/lib
				   /usr/lib64
                   /usr/local/lib64
                   /usr/local/lib64
                   /opt/local/lib64
	    HINTS $ENV{NGRAPH_DIR}/lib
            PATH_SUFFIXES ngraph)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (NGraph REQUIRED_VARS NGraph_LIBRARIES NGraph_INCLUDE_DIRS)

