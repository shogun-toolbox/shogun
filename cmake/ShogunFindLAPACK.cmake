# Enable Eigen to use Lapack backend
OPTION(ENABLE_EIGEN_LAPACK "Enable Eigen to use detected BLAS and LAPACK backend" ON)

FIND_PACKAGE(LAPACK QUIET)
IF (LAPACK_FOUND)
  SET(HAVE_LAPACK 1)

  # find out the type of Lapack/BLAS implementation we are dealing with
  IF("${LAPACK_LIBRARIES}" MATCHES ".*/Accelerate.framework$")
    # Accelerate.framework we found for LaPack/BLAS
    SET(HAVE_MVEC 1)
    SET(HAVE_CATLAS 1)
    MESSAGE(STATUS "Found Accelerate.framework using as BLAS/LAPACK backend.")

    if (ENABLE_EIGEN_LAPACK)
      SET(EIGEN_USE_BLAS 1)
      MESSAGE(STATUS "Enabling Accelerate.framework as BLAS backend for Eigen.")
      find_library(LAPACKE_LIBRARY
        NAMES lapacke
        PATHS /usr/lib /usr/local/lib $ENV{LAPACKE_PATH})
      if (LAPACKE_LIBRARY)
        # no comes some magic as on osx el capitalismo and sierra leone
        # versions things are working in some magic way (vecLib that is)
        # see #4136
        include(CheckCXXSourceCompiles)
        CHECK_CXX_SOURCE_COMPILES(
          "#include <vecLib/cblas.h>
          int main(void){
              return 0;
          }" VECLIB_INCLUDE_WORKS)

        if (VECLIB_INCLUDE_WORKS)
          MESSAGE(STATUS "Enabling Accelerate.framework as LAPACK backend for Eigen.")
          SET(EIGEN_USE_LAPACKE_STRICT 1)
          LIST(APPEND LAPACK_LIBRARIES ${LAPACKE_LIBRARY})
        else()
          MESSAGE(STATUS "Could not include <vecLib/cblas.h> hence not enabling LAPACK as an Eigen backend")
        endif()
      endif()
    endif()
  ELSEIF("${LAPACK_LIBRARIES}" MATCHES ".*/.*mkl_.*")
    # in case MKL is available enable Eigen to use it.
    # for more fine grained control and details see:
    # https://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html
    # this is supported since Eigen version 3.1 and later
    MESSAGE(STATUS "Found MKL using as BLAS/LAPACK backend.")
    SET(HAVE_MKL 1)

    # trying to use the new Single Dynamic Library of MKL
    # https://software.intel.com/en-us/articles/a-new-linking-model-single-dynamic-library-mkl_rt-since-intel-mkl-103
    IF (NOT "${LAPACK_LIBRARIES}" MATCHES ".*/.*mkl_rt.*")
      # just use the mkl_rt and let the user specify/decide all the MKL specific
      # optimisation runtime
      SET(MKL_LIBRARIES ${LAPACK_LIBRARIES})
      LIST(FILTER MKL_LIBRARIES INCLUDE REGEX ".*/.*mkl_core.*")
      get_filename_component(MKL_PATH ${MKL_LIBRARIES} DIRECTORY)
      find_library(MKL_RT mkl_rt PATHS ${MKL_PATH})
      IF (MKL_RT)
        IF (MSVC)
          SET(LAPACK_LIBRARIES ${MKL_RT})
        ELSEIF(CMAKE_USE_PTHREADS_INIT)
          SET(LAPACK_LIBRARIES ${MKL_RT})
          LIST(APPEND LAPACK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} -lm)
        ENDIF()
      ENDIF()
    ENDIF()

    IF (ENABLE_EIGEN_LAPACK)
      FIND_PATH(MKL_INCLUDE_DIR mkl.h)
      IF(NOT MKL_INCLUDE_DIR)
        MESSAGE(STATUS "Found MKL, but not mkl.h. Make sure that mkl headers are available in order to use MKL as BLAS/Lapack backend for Eigen.")
        SET(ENABLE_EIGEN_LAPACK OFF CACHE BOOL "Use ${ENABLE_EIGEN_LAPACK}" FORCE)
      ELSE()
        MESSAGE(STATUS "Enabling MKL as BLAS/Lapack backend for Eigen.")
        SET(EIGEN_USE_MKL_ALL 1)
        target_include_directories(shogun PUBLIC ${MKL_INCLUDE_DIR})
        IF (LIBSHOGUN_BUILD_STATIC)
          target_include_directories(shogun-static PUBLIC ${MKL_INCLUDE_DIR})
        ENDIF()
      ENDIF()
    ENDIF()
  ELSE()
    include(CheckLibraryExists)
    # test whether we have cblas.h in the header paths and the detected
    # LAPACK_LIBRARIES contains all the libraries to compile even with cblas_* functions
    check_library_exists("${LAPACK_LIBRARIES}" cblas_dgemv "" FOUND_CBLAS_DGEMV)

    # detect if the detected Lapack is atlas
    # clapack_* functions are implemented in atlas
    check_library_exists("${LAPACK_LIBRARIES}" clapack_dpotrf "" FOUND_CLAPACK_DPOTRF)
    IF (FOUND_CLAPACK_DPOTRF OR NOT FOUND_CBLAS_DGEMV)
      FIND_PACKAGE(Atlas QUIET)
      IF(Atlas_FOUND)
        MESSAGE(STATUS "Found Atlas using as BLAS/LAPACK backend.")
        SET(HAVE_ATLAS 1)
        SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC ${Atlas_INCLUDE_DIRS})
        IF (NOT FOUND_CBLAS_DGEMV)
          # this usually happens on RHEL/CentOS; usually having Atlas is good
          SET(LAPACK_LIBRARIES ${Atlas_LIBRARIES})
        ENDIF()
      ELSEIF(NOT FOUND_CBLAS_DGEMV)
          UNSET(LAPACK_FOUND CACHE)
          UNSET(LAPACK_LIBRARIES)
          UNSET(HAVE_LAPACK)
      ENDIF()
    ENDIF()

    IF(FOUND_CBLAS_DGEMV AND NOT HAVE_ATLAS)
      #check is detected BLAS/LAPACK is OpenBLAS by looking for an OpenBLAS specific function
      check_library_exists("${LAPACK_LIBRARIES}" openblas_set_num_threads "" OpenBLAS_FOUND)
      IF(OpenBLAS_FOUND)
        #check if cblas.h exists
        FIND_PATH(CBLAS_INCLUDE_DIR cblas.h)
          IF(NOT CBLAS_INCLUDE_DIR)
	    MESSAGE(STATUS "Make sure that cblas.h header is available within the header search path in order to use OpenBLAS as BLAS/Lapack backend")
	    UNSET(LAPACK_FOUND CACHE)
	    UNSET(LAPACK_LIBRARIES)
	    UNSET(HAVE_LAPACK)
          ELSE()
            MESSAGE("Found OPENBLAS using as BLAS/LAPACK backend.")
          ENDIF()
      ENDIF()
    ENDIF()
    # if LaPack is detected use the lapack/blas backend in Eigen
    IF(ENABLE_EIGEN_LAPACK AND HAVE_LAPACK)
      SET(EIGEN_USE_BLAS 1)
      MESSAGE(STATUS "Enabling detected BLAS library as backend for Eigen")

      find_library(LAPACKE_LIBRARY NAMES lapacke PATHS /usr/lib /usr/local/lib $ENV{LAPACKE_PATH})
      IF (LAPACKE_LIBRARY)
        MESSAGE(STATUS "Enabling detected LAPACK backend for Eigen")
        SET(EIGEN_USE_LAPACKE_STRICT 1)
        LIST(APPEND LAPACK_LIBRARIES ${LAPACKE_LIBRARY})
      ENDIF()
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
