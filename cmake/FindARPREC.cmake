
find_path(ARPREC_INCLUDE_DIR arprec/mp_real.h HINTS $ENV{ARPREC_DIR}/include)
find_library(ARPREC_LIBRARIES arprec HINTS $ENV{ARPREC_DIR}/lib)

# handle REQUIRED and QUIET options
include (FindPackageHandleStandardArgs)
if (CMAKE_VERSION LESS 2.8.3)
  find_package_handle_standard_args (ARPREC DEFAULT_MSG ARPREC_LIBRARIES ARPREC_INCLUDE_DIR)
else ()
  find_package_handle_standard_args (ARPREC REQUIRED_VARS ARPREC_LIBRARIES ARPREC_INCLUDE_DIR)
endif ()

mark_as_advanced(ARPREC_INCLUDE_DIR ARPREC_LIBRARIES)
