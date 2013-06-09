# - Try to find a version of Octave and headers/library required by the
# used compiler. It determines the right MEX-File extension and add
# a macro to help the build of MEX-functions.
#
# This module defines:
# OCTAVE_INCLUDE_DIR: include path for mex.h, mexproto.h
# OCTAVE_OCTINTERP_LIBRARY: path to the library octinterp
# OCTAVE_OCTAVE_LIBRARY: path to the library octave
# OCTAVE_CRUFT_LIBRARY: path to the library cruft
# OCTAVE_LIBRARIES: required libraries: octinterp, octave, cruft
# OCTAVE_CREATE_MEX: macro to build a MEX-file
#
# The macro OCTAVE_CREATE_MEX requires in this order:
# - function's name which will be called in Octave;
# - C/C++ source files;
# - third libraries required.

# Copyright (c) 2009-2011 Arnaud Barré <arnaud.barre@gmail.com>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

IF(OCTAVE_ROOT AND OCTAVE_INCLUDE_DIR AND OCTAVE_LIBRARIES)
# in cache already
SET(Octave_FIND_QUIETLY TRUE)
ENDIF(OCTAVE_ROOT AND OCTAVE_INCLUDE_DIR AND OCTAVE_LIBRARIES)

SET(OCTAVE_MEXFILE_EXT mex)

IF(WIN32)
FILE(GLOB OCTAVE_PATHS "c:/Octave/*")
FIND_PATH(OCTAVE_ROOT "bin/octave.exe" ${OCTAVE_PATHS})

FILE(GLOB OCTAVE_INCLUDE_PATHS "${OCTAVE_ROOT}/include/octave-*/octave")
FILE(GLOB OCTAVE_LIBRARIES_PATHS "${OCTAVE_ROOT}/lib/octave-*")

# LIBOCTINTERP, LIBOCTAVE, LIBCRUFT names
SET(LIBOCTINTERP "liboctinterp")
SET(LIBOCTAVE "liboctave")
SET(LIBCRUFT "libcruft")
ELSE(WIN32)
# MEX files extension
IF(APPLE)
FILE(GLOB OCTAVE_PATHS "/Applications/Octave*")
FIND_PATH(OCTAVE_ROOT "Contents/Resources/bin/octave" ${OCTAVE_PATHS})

FILE(GLOB OCTAVE_INCLUDE_PATHS "${OCTAVE_ROOT}/Contents/Resources/include/octave-*/octave")
FILE(GLOB OCTAVE_LIBRARIES_PATHS "${OCTAVE_ROOT}/Contents/Resources/lib/octave-*")

SET(LIBOCTINTERP "liboctinterp.dylib")
SET(LIBOCTAVE "liboctave.dylib")
SET(LIBCRUFT "libcruft.dylib")
ELSE(APPLE)
SET(OCTAVE_ROOT "")
FILE(GLOB OCTAVE_LOCAL_PATHS "/usr/local/lib/octave-*")
FILE(GLOB OCTAVE_USR_PATHS "/usr/lib/octave-*")

SET (OCTAVE_INCLUDE_PATHS
"/usr/local/include"
"/usr/local/include/octave"
"/usr/include"
"/usr/include/octave")
SET (OCTAVE_LIBRARIES_PATHS
"/usr/local/lib"
"/usr/local/lib/octave"
${OCTAVE_LOCAL_PATHS}
"/usr/lib"
"/usr/lib/octave"
${OCTAVE_USR_PATHS})

SET (LIBOCTINTERP "octinterp")
SET (LIBOCTAVE "octave")
SET (LIBCRUFT "cruft")
ENDIF(APPLE)
ENDIF(WIN32)

FIND_LIBRARY(OCTAVE_OCTINTERP_LIBRARY
${LIBOCTINTERP}
${OCTAVE_LIBRARIES_PATHS} NO_DEFAULT_PATH
)
FIND_LIBRARY(OCTAVE_OCTAVE_LIBRARY
${LIBOCTAVE}
${OCTAVE_LIBRARIES_PATHS} NO_DEFAULT_PATH
)
FIND_LIBRARY(OCTAVE_CRUFT_LIBRARY
${LIBCRUFT}
${OCTAVE_LIBRARIES_PATHS} NO_DEFAULT_PATH
)
FIND_PATH(OCTAVE_INCLUDE_DIR
"mex.h"
${OCTAVE_INCLUDE_PATHS} NO_DEFAULT_PATH
)

# This is common to UNIX and Win32:
SET(OCTAVE_LIBRARIES
${OCTAVE_OCTINTERP_LIBRARY}
${OCTAVE_OCTAVE_LIBRARY}
${OCTAVE_CRUFT_LIBRARY}
)

# Macros for building MEX-files
MACRO(OCTAVE_EXTRACT_SOURCES_LIBRARIES sources thirdlibraries)
SET(${sources})
SET(${thirdlibraries})
FOREACH(_arg ${ARGN})
GET_FILENAME_COMPONENT(_ext ${_arg} EXT)
IF("${_ext}" STREQUAL "")
LIST(APPEND ${thirdlibraries} "${_arg}")
ELSE("${_ext}" STREQUAL "")
LIST(APPEND ${sources} "${_arg}")
ENDIF ("${_ext}" STREQUAL "")
ENDFOREACH(_arg)
ENDMACRO(OCTAVE_EXTRACT_SOURCES_LIBRARIES)

# OCTAVE_MEX_CREATE(functionname inputfiles thridlibraries)
MACRO(OCTAVE_MEX_CREATE functionname)
OCTAVE_EXTRACT_SOURCES_LIBRARIES(sources thirdlibraries ${ARGN})
ADD_LIBRARY(${functionname} SHARED ${sources})
TARGET_LINK_LIBRARIES(${functionname} ${OCTAVE_LIBRARIES} ${thirdlibraries})
SET_TARGET_PROPERTIES(${functionname} PROPERTIES
PREFIX ""
SUFFIX ".${OCTAVE_MEXFILE_EXT}"
)
ENDMACRO(OCTAVE_MEX_CREATE)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Octave DEFAULT_MSG OCTAVE_ROOT OCTAVE_INCLUDE_DIR OCTAVE_OCTINTERP_LIBRARY OCTAVE_OCTAVE_LIBRARY OCTAVE_CRUFT_LIBRARY)

MARK_AS_ADVANCED(
OCTAVE_OCTINTERP_LIBRARY
OCTAVE_OCTAVE_LIBRARY
OCTAVE_CRUFT_LIBRARY
OCTAVE_LIBRARIES
OCTAVE_INCLUDE_DIR
)