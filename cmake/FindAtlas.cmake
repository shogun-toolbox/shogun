if (ATLAS_LIBRARIES)
  set(ATLAS_FIND_QUIETLY TRUE)
endif (ATLAS_LIBRARIES)

find_file(ATLAS_LIB libatlas.so.3 PATHS /usr/lib /usr/lib64 $ENV{ATLASDIR} PATH_SUFFIXES atlas atlas-base)
find_library(ATLAS_LIB atlas PATHS $ENV{ATLASDIR})

find_file(ATLAS_CBLAS libcblas.so.3 PATHS /usr/lib /usr/lib64 $ENV{ATLASDIR} PATH_SUFFIXES atlas atlas-base)
find_library(ATLAS_CBLAS cblas PATHS $ENV{ATLASDIR})

find_file(ATLAS_LAPACK NAMES liblapack_atlas.so.3 PATHS /usr/lib /usr/lib64 $ENV{ATLASDIR} PATH_SUFFIXES atlas atlas-base)
find_library(ATLAS_LAPACK NAMES lapack_atlas alapack PATHS $ENV{ATLASDIR})

if(ATLAS_LAPACK)
  include(CheckLibraryExists)
  set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES} ${ATLAS_CBLAS})
  check_library_exists("${ATLAS_LAPACK}" clapack_dpotrf "" FOUND_CLAPACK)
  if(NOT FOUND_CLAPACK)
    unset(ATLAS_LAPACK CACHE)
  endif()
  unset(CMAKE_REQUIRED_LIBRARIES CACHE)
else()
  find_file(ATLAS_LAPACK liblapack.so.3 PATHS /usr/lib/atlas /usr/lib64/atlas)
  find_library(ATLAS_LAPACK NAMES lapack)
  set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES} ${ATLAS_CBLAS})
  check_library_exists("${ATLAS_LAPACK}" clapack_dpotrf "" FOUND_CLAPACK)
  if(NOT FOUND_CLAPACK)
    unset(ATLAS_LAPACK CACHE)
  endif()
  unset(CMAKE_REQUIRED_LIBRARIES CACHE)
endif()

find_file(ATLAS_F77BLAS libf77blas.so.3 PATHS /usr/lib /usr/lib64 $ENV{ATLASDIR} PATH_SUFFIXES atlas atlas-base)
find_library(ATLAS_F77BLAS f77blas PATHS $ENV{ATLASDIR} PATH_SUFFIXES atlas atlas-base)

if(ATLAS_LIB AND ATLAS_CBLAS AND ATLAS_LAPACK AND ATLAS_F77BLAS)

  set(ATLAS_LIBRARIES ${ATLAS_LAPACK} ${ATLAS_CBLAS} ${ATLAS_F77BLAS} ${ATLAS_LIB})
  
  # search the default lapack lib link to it
  find_file(ATLAS_REFERENCE_LAPACK liblapack.so.3 PATHS /usr/lib /usr/lib64)
  find_library(ATLAS_REFERENCE_LAPACK NAMES lapack)
  if(ATLAS_REFERENCE_LAPACK)
    set(ATLAS_LIBRARIES ${ATLAS_LIBRARIES} ${ATLAS_REFERENCE_LAPACK})
  endif()
  
endif(ATLAS_LIB AND ATLAS_CBLAS AND ATLAS_LAPACK AND ATLAS_F77BLAS)

find_path(ATLAS_INCLUDES clapack.h PATHS /usr/include/atlas /usr/local/include/atlas /opt/local/include/atlas $ENV{ATLASDIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ATLAS DEFAULT_MSG ATLAS_LIBRARIES ATLAS_INCLUDES)

mark_as_advanced(ATLAS_LIBRARIES ATLAS_INCLUDES)
