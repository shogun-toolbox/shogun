# - Find NLopt
# Find the native NLopt includes and library
#
#  NLOPT_INCLUDE_DIR - where to find nlopt.h, etc.
#  NLOPT_LIBRARIES   - List of libraries when using nlopt.
#  NLOPT_FOUND       - True if nlopt found.


IF (NLOPT_INCLUDE_DIR)
  # Already in cache, be silent
  SET (nlopt_FIND_QUIETLY TRUE)
ENDIF (NLOPT_INCLUDE_DIR)

FIND_PATH(NLOPT_INCLUDE_DIR nlopt.h)

SET (NLOPT_NAMES nlopt nlopt_cxx)
FIND_LIBRARY (NLOPT_LIBRARY NAMES ${NLOPT_NAMES})

# handle the QUIETLY and REQUIRED arguments and set NLOPT_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (NLOPT DEFAULT_MSG 
  NLOPT_LIBRARY 
  NLOPT_INCLUDE_DIR)

IF(NLOPT_FOUND)
  SET (NLOPT_LIBRARIES ${NLOPT_LIBRARY})
ELSE (NLOPT_FOUND)
  SET (NLOPT_LIBRARIES)
ENDIF (NLOPT_FOUND)

MARK_AS_ADVANCED (NLOPT_LIBRARY NLOPT_INCLUDE_DIR)