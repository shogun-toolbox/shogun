# Enable Eigen to use Lapack backend
OPTION(ENABLE_EIGEN_LAPACK "Enable Eigen to use detected BLAS and LAPACK backend" ON)

FIND_PACKAGE(LAPACK)
IF (LAPACK_FOUND)
  SET(HAVE_LAPACK 1)

  # find out the type of Lapack/BLAS implementation we are dealing with
  IF("${LAPACK_LIBRARIES}" MATCHES ".*/Accelerate.framework$")
    # Accelerate.framework we found for LaPack/BLAS
    SET(HAVE_MVEC 1)
    SET(HAVE_CATLAS 1)

    if (ENABLE_EIGEN_LAPACK)
      SET(EIGEN_USE_BLAS 1)
      MESSAGE(STATUS "Enabling Accelerate.framework as BLAS backend for Eigen")
      find_library(LAPACKE_LIBRARY
        NAMES lapacke
        PATHS /usr/lib /usr/local/lib $ENV{LAPACKE_PATH})
      if (LAPACKE_LIBRARY)
        MESSAGE(STATUS "Enabling Accelerate.framework as LAPACK backend for Eigen")
        SET(EIGEN_USE_LAPACKE_STRICT 1)
        LIST(APPEND LAPACK_LIBRARIES ${LAPACKE_LIBRARY})
      endif()
    endif()
  ELSEIF("${LAPACK_LIBRARIES}" MATCHES ".*/libmkl_.*")
    # in case MKL is available enable Eigen to use it.
    # for more fine grained control and details see:
    # https://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html
    # this is supported since Eigen version 3.1 and later
    SET(HAVE_MKL 1)
    IF (ENABLE_EIGEN_LAPACK)
      MESSAGE(STATUS "Enabling MKL as BLAS/Lapack backend for Eigen")
      SET(EIGEN_USE_MKL_ALL 1)
    ENDIF()
  ELSE()
    # detect if the detected Lapack is atlas
    # clapack_* functions are implemented in atlas
    include(CheckLibraryExists)
    check_library_exists("${LAPACK_LIBRARIES}" clapack_dpotrf "" FOUND_CLAPACK_DPOTRF)
    IF (FOUND_CLAPACK_DPOTRF)
      FIND_PACKAGE(Atlas)
      IF(Atlas_FOUND)
        SET(HAVE_ATLAS 1)
        SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC ${Atlas_INCLUDE_DIRS})
      ENDIF()
    ENDIF()

    # if LaPack is detected and Eigen is 3.3 or later
    # use the lapack/blas backend in Eigen
    IF(${EIGEN_VERSION} VERSION_GREATER 3.3.0 AND ENABLE_EIGEN_LAPACK)
      SET(EIGEN_USE_BLAS 1)
      MESSAGE(STATUS "Enabling detected BLAS library as backend for Eigen")

      find_library(LAPACKE_LIBRARY NAMES lapacke PATHS /usr/lib /usr/local/lib $ENV{LAPACKE_PATH})
      if (LAPACKE_LIBRARY)
        MESSAGE(STATUS "Enabling detected LAPACK backend for Eigen")
        SET(EIGEN_USE_LAPACKE_STRICT 1)
        LIST(APPEND LAPACK_LIBRARIES ${LAPACKE_LIBRARY})
      endif()
    ENDIF()
  ENDIF()

  IF (ENABLE_EIGEN_LAPACK)
    SET (LAPACK_SCOPE PUBLIC)
  ELSE()
    SET (LAPACK_SCOPE PRIVATE)
  ENDIF()
  target_link_libraries(shogun ${LAPACK_SCOPE} ${LAPACK_LIBRARIES})
  if (LIBSHOGUN_BUILD_STATIC)
    target_link_libraries(shogun-static ${LAPACK_SCOPE} ${LAPACK_LIBRARIES})
  endif()
ENDIF()
