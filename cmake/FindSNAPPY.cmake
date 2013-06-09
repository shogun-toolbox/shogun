# Find SNAPPY - A fast compressor/decompressor
#
# This module defines
#  SNAPPY_FOUND - whether the qsjon library was found
#  SNAPPY_LIBRARIES - the snappy library
#  SNAPPY_INCLUDE_DIR - the include path of the snappy library
#

if (SNAPPY_INCLUDE_DIR AND SNAPPY_LIBRARIES)

  # Already in cache
  set (SNAPPY_FOUND TRUE)

else (SNAPPY_INCLUDE_DIR AND SNAPPY_LIBRARIES)

  find_library (SNAPPY_LIBRARIES
    NAMES
    snappy
    PATHS
  )

  find_path (SNAPPY_INCLUDE_DIR
    NAMES
    snappy.h
    PATHS
  )

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(SNAPPY DEFAULT_MSG SNAPPY_LIBRARIES SNAPPY_INCLUDE_DIR)

endif (SNAPPY_INCLUDE_DIR AND SNAPPY_LIBRARIES)