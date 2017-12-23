# Find the Atlas (and Lapack) libraries
#
# The following variables are optionally searched for defaults
#  Atlas_ROOT_DIR:            Base directory where all Atlas components are found
#
# The following are set after configuration is done:
#  Atlas_FOUND
#  Atlas_INCLUDE_DIRS
#  Atlas_LIBRARIES

set(Atlas_INCLUDE_SEARCH_PATHS
  /usr/include/atlas
  /usr/include/atlas-base
  $ENV{Atlas_ROOT_DIR}
  $ENV{Atlas_ROOT_DIR}/include
)

set(Atlas_LIB_SEARCH_PATHS
  /usr/lib/atlas
  /usr/lib/atlas-base
  /usr/lib64/atlas
  /usr/lib64/atlas-base
  $ENV{Atlas_ROOT_DIR}
  $ENV{Atlas_ROOT_DIR}/lib
)
find_path(Atlas_CBLAS_INCLUDE_DIR   NAMES cblas.h   PATHS ${Atlas_INCLUDE_SEARCH_PATHS})
find_path(Atlas_CLAPACK_INCLUDE_DIR NAMES clapack.h PATHS ${Atlas_INCLUDE_SEARCH_PATHS})

find_library(Atlas_BLAS_LIBRARY NAMES atlas_r atlas tatlas satlas PATHS ${Atlas_LIB_SEARCH_PATHS})
set(ATLAS_LIBS_VAR Atlas_BLAS_LIBRARY)
if (Atlas_BLAS_LIBRARY)
  include(CheckLibraryExists)
  # atlas 3.10+ contains all the function in one shared lib so dont try to find other parts of atlas
  check_library_exists("${Atlas_BLAS_LIBRARY}" cblas_dgemv "" FOUND_ATLAS_CBLAS_DGEMV)
  if (NOT FOUND_ATLAS_CBLAS_DGEMV)
    find_library(Atlas_CBLAS_LIBRARY NAMES ptcblas_r ptcblas cblas_r cblas PATHS ${Atlas_LIB_SEARCH_PATHS})
    list(APPEND ATLAS_LIBS_VAR Atlas_CBLAS_LIBRARY)
  endif()
  check_library_exists("${Atlas_BLAS_LIBRARY}" clapack_dpotrf "" FOUND_ATLAS_CLAPACK_DPOTRF)
  if (NOT FOUND_ATLAS_CLAPACK_DPOTRF)
    find_library(Atlas_LAPACK_LIBRARY NAMES alapack_r alapack lapack_atlas atllapack PATHS ${Atlas_LIB_SEARCH_PATHS})
    list(APPEND ATLAS_LIBS_VAR Atlas_LAPACK_LIBRARY)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Atlas DEFAULT_MSG Atlas_CBLAS_INCLUDE_DIR Atlas_CLAPACK_INCLUDE_DIR ${ATLAS_LIBS_VAR})

if(ATLAS_FOUND)
  set(Atlas_INCLUDE_DIRS ${Atlas_CBLAS_INCLUDE_DIR} ${Atlas_CLAPACK_INCLUDE_DIR})
  set(ATLAS_LIBS)
  foreach(atlas_lib ${ATLAS_LIBS_VAR})
    list(APPEND ATLAS_LIBS ${${atlas_lib}})
  endforeach()
  set(Atlas_LIBRARIES ${ATLAS_LIBS})
  mark_as_advanced(${Atlas_CBLAS_INCLUDE_DIR} ${Atlas_CLAPACK_INCLUDE_DIR} ${ATLAS_LIBS})

  message(STATUS "Found Atlas (include: ${Atlas_CBLAS_INCLUDE_DIR}, library: ${Atlas_BLAS_LIBRARY})")
endif(ATLAS_FOUND)
