# - Find LZO (lzo.h, liblzo2.a)
# This module defines
# LZO_INCLUDE_DIR, directory containing headers
# LZO_STATIC_LIB, path to libslzo2.a
# LZO_FOUND, whether lzo has been found

find_path(LZO_INCLUDE_DIR NAMES lzo/lzoconf.h)

find_library(LZO_LIB NAMES liblzo2.so)

if (LZO_LIB)
  if (LZO_INCLUDE_DIR)
    set(LZO_FOUND TRUE)
  else ()
    set(LZO_FOUND FALSE)
  endif()
else ()
  set(LZO_FOUND FALSE)
endif ()

if (LZO_FOUND)
  if (NOT LZO_FIND_QUIETLY)
    message(STATUS "Lzo Library ${LZO_LIB}")
    message(STATUS "Lzo Include Found in ${LZO_INCLUDE_DIR}")
  endif ()
else ()
  message(STATUS "Lzo includes and libraries NOT found. ")
endif ()

mark_as_advanced(
  LZO_INCLUDE_DIR
  LZO_LIBS
)