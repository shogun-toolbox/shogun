INCLUDE_DIRECTORIES(
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}/src
)

include(GenerateExportHeader)
include(CMakeDependentOption)
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckFunctionExists)
include(CheckCXXSourceCompiles)
include(CMakePackageConfigHelpers)

FILE(GLOB_RECURSE LIBSHOGUN_SRC *.${EXT_SRC_CPP} *.${EXT_SRC_C})
FILE(GLOB_RECURSE LIBSHOGUN_HEADERS *.${EXT_SRC_HEADER})
FILE(GLOB_RECURSE LIBSHOGUN_SRC_TMP *.${EXT_CPP_TMP})

IF(ENABLE_TESTING AND BUILD_META_EXAMPLES)
  SET(USE_META_INTEGRATION_TESTS 1)
ENDIF()

OPTION(LIBSHOGUN_BUILD_STATIC "Build libshogun static library")
OPTION(DISABLE_SSE "Disable SSE and SSE2 features.")
set(INCLUDE_INSTALL_DIR include)
set(THIRD_PARTY_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib/external)

if (MSVC AND (BUILD_EXAMPLES OR BUILD_META_EXAMPLES))
  SET(LIBSHOGUN_BUILD_STATIC ON
    CACHE BOOL "Build libshogun static library" FORCE)
endif()

# add target to compile the libshogun sources
add_library(libshogun OBJECT ${LIBSHOGUN_SRC} ${LIBSHOGUN_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h)
set_property(TARGET libshogun PROPERTY POSITION_INDEPENDENT_CODE TRUE)
IF (SANITIZER_FLAGS)
  set_property(TARGET libshogun PROPERTY COMPILE_FLAGS ${SANITIZER_FLAGS})
ENDIF()

# Generate versionstring.h
ADD_CUSTOM_TARGET(
    version
    ${CMAKE_COMMAND}
      -D SRC=${CMAKE_CURRENT_SOURCE_DIR}/lib/versionstring.h.in
      -D DST=${CMAKE_CURRENT_BINARY_DIR}/lib/versionstring.h
      -D ROOT_DIR=${CMAKE_SOURCE_DIR}
      -D PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      -P ${CMAKE_MODULE_PATH}/version.cmake
    COMMENT "Generating version header"
)
add_dependencies(libshogun version)

# generate source files from templates
FOREACH(template ${LIBSHOGUN_SRC_TMP})
  STRING(REGEX REPLACE ${EXT_CPP_TMP} ${EXT_SRC_CPP} generated_cpp "${template}")
  STRING(REGEX REPLACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} generated_cpp "${generated_cpp}")
  STRING(REGEX REPLACE ${EXT_CPP_TMP} ${EXT_CPP_PY} generator_script "${template}")
  STRING(REGEX REPLACE ".*/(.*).${EXT_CPP_TMP}" "\\1" generated_target "${template}")

  IF(MSVC)
    FOREACH(h ${LIBSHOGUN_HEADERS})
      FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/headers_list.txt "${h}\n")
    ENDFOREACH()

    ADD_CUSTOM_COMMAND(OUTPUT ${generated_cpp}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${template} ${generated_cpp}
        COMMAND ${PYTHON_EXECUTABLE} ${generator_script} ${generated_cpp} -in ${CMAKE_CURRENT_BINARY_DIR}/headers_list.txt
        DEPENDS ${template} ${generator_script} version ${LIBSHOGUN_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/headers_list.txt
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating ${generated_cpp}"
    )
  ELSE()
    ADD_CUSTOM_COMMAND(OUTPUT ${generated_cpp}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${template} ${generated_cpp}
        COMMAND ${PYTHON_EXECUTABLE} ${generator_script} ${generated_cpp} ${LIBSHOGUN_HEADERS}
        DEPENDS ${template} ${generator_script} version ${LIBSHOGUN_HEADERS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating ${generated_cpp}"
    )
  ENDIF()

  ADD_CUSTOM_TARGET(${generated_target} DEPENDS ${generated_cpp})
  add_dependencies(libshogun ${generated_target})
  set_source_files_properties(${generated_cpp} PROPERTIES GENERATED ON)
  target_sources(libshogun PRIVATE ${generated_cpp})
ENDFOREACH()

add_library(shogun SHARED $<TARGET_OBJECTS:libshogun> ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h)
generate_export_header(shogun)
set_target_properties(
  shogun PROPERTIES
  VERSION ${LIBSHOGUNVER}
  SOVERSION ${LIBSHOGUNSO}
  INTERFACE_shogun_MAJOR_VERSION ${SHOGUN_VERSION_MAJOR})
set_property(TARGET shogun APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${SHOGUN_VERSION_MAJOR})
target_include_directories(shogun PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
  $<INSTALL_INTERFACE:include/shogun>
)

# create shogun_deps meta target to store all the
# dependencies of shogun itself, this is used in unit test
add_library(shogun_deps INTERFACE)
target_link_libraries(shogun_deps INTERFACE shogun)
target_include_directories(shogun_deps INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
)

# add bundled libraries to dependency
if(SHOGUN_DEPENDS)
  add_dependencies(libshogun ${SHOGUN_DEPENDS})
endif()

# add target for static library if enabled
if (LIBSHOGUN_BUILD_STATIC)
  add_library(shogun-static STATIC $<TARGET_OBJECTS:libshogun> ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h)
  set_property(TARGET shogun-static PROPERTY OUTPUT_NAME shogun)
  if(MSVC)
    target_link_libraries(shogun-static PUBLIC winmm.lib)
  endif()
ENDIF()

IF(MSVC)
  IF(MSVC_VERSION VERSION_LESS "1700")
    include(external/MSIntTypes)
    SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC
      $<BUILD_INTERFACE:${MSINTTYPES_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include/shogun/lib/external/MSIntTypes>
    )
  ENDIF()

  # bundle dirent
  include(external/MSDirent)
  SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC
    $<BUILD_INTERFACE:${MSDIRENT_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include/shogun/lib/external/MSDirent>
  )

  target_link_libraries(shogun PUBLIC winmm)
  target_link_libraries(shogun_deps INTERFACE winmm)
ENDIF()

########################### compiler capabilities
FIND_PACKAGE(Threads)
IF (CMAKE_USE_PTHREADS_INIT)
  SET(HAVE_PTHREAD 1)
  SHOGUN_LINK_LIBS(${CMAKE_THREAD_LIBS_INIT})
ENDIF()

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
  SET(HAVE_OPENMP 1)
  SHOGUN_COMPILE_OPTS(${OpenMP_CXX_FLAGS})
  if (CMAKE_COMPILER_IS_GNUCC)
    SHOGUN_LINK_LIBS(gomp)
  endif()
endif()

FIND_PACKAGE(Spinlock)
IF (SPINLOCK_FOUND)
  SET(USE_SPINLOCKS 1)
ENDIF()

FIND_PACKAGE(CxaDemangle)

# check SSE and SSE2 intrinsics header
IF((NOT CYGWIN) AND (NOT DISABLE_SSE))
  CHECK_INCLUDE_FILE(xmmintrin.h HAVE_BUILTIN_VECTOR)
  CHECK_INCLUDE_FILE(emmintrin.h HAVE_SSE2)
ENDIF((NOT CYGWIN) AND (NOT DISABLE_SSE))

FIND_PACKAGE(CxaDemangle)
############################ std lib functions
CHECK_INCLUDE_FILE_CXX("unordered_map" HAVE_STD_UNORDERED_MAP)

include (CheckCXXSymbolExists)
CHECK_CXX_SYMBOL_EXISTS(isfinite "cmath" HAVE_DECL_ISFINITE)
CHECK_CXX_SYMBOL_EXISTS(isinf "cmath" HAVE_DECL_ISINF)
CHECK_CXX_SYMBOL_EXISTS(isnan "cmath" HAVE_DECL_ISNAN)
CHECK_CXX_SYMBOL_EXISTS(signgam "cmath" HAVE_DECL_SIGNGAM)
CHECK_CXX_SYMBOL_EXISTS(fdopen "stdio.h" HAVE_FDOPEN)

# check for math functions
IF(UNIX)
  SET(CMAKE_REQUIRED_LIBRARIES m)
ENDIF()
CHECK_FUNCTION_EXISTS(log2 HAVE_LOG2)
CHECK_FUNCTION_EXISTS(powl HAVE_POWL)
CHECK_FUNCTION_EXISTS(lgammal HAVE_LGAMMAL)
CHECK_FUNCTION_EXISTS(sqrtl HAVE_SQRTL)

CHECK_FUNCTION_EXISTS(finite HAVE_FPCLASS)
CHECK_FUNCTION_EXISTS(fpclass HAVE_FPCLASS)
CHECK_FUNCTION_EXISTS(isfinite HAVE_ISFINITE)
CHECK_FUNCTION_EXISTS(isinf HAVE_ISINF)
CHECK_FUNCTION_EXISTS(isnan HAVE_ISNAN)

CHECK_CXX_SOURCE_COMPILES(
  "#include <cmath>\nint main() { return std::isinf( 0 ); }\n"
  HAVE_STD_ISINF )
CHECK_CXX_SOURCE_COMPILES(
  "#include <cmath>\nint main() { return std::isfinite( 0 ); }\n"
  HAVE_STD_ISFINITE )
CHECK_CXX_SOURCE_COMPILES(
  "#include <cmath>\nint main() { return std::isnan( 0 ); }\n"
  HAVE_STD_ISNAN )

############################ external dependencies

####### LINALG
FIND_PACKAGE(Eigen3 ${EIGEN_VERSION_MINIMUM})
IF(NOT EIGEN3_FOUND)
  include(external/Eigen3)
  SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC SYSTEM
    $<BUILD_INTERFACE:${EIGEN_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include/shogun/lib/external/eigen>
  )
ELSE()
  # eigen3 bug related to aliasing operators and self-storing.
  IF(${EIGEN_VERSION} VERSION_GREATER 3.2.90 AND ${EIGEN_VERSION} VERSION_LESS 3.3.0)
      MESSAGE(WARNING "The system Eigen3 version ${EIGEN_VERSION} contains a lot of bugs, which could cause some models failing!")
  ENDIF()
  SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC SYSTEM ${EIGEN_INCLUDE_DIR})
ENDIF()

# ViennaCL detection
SHOGUN_DEPENDENCIES(
  LIBRARY ViennaCL
  SCOPE PRIVATE
  VERSION ${VIENNACL_VERSION_MINIMUM}
  CONFIG_FLAG HAVE_VIENNACL)

# Linear algebra default global backend setups
SET_LINALG_BACKEND(LINALG_DEFAULT_BACKEND GLOBAL)

# Linear algebra default module specific backend setup
# Linear solver module
SET_LINALG_BACKEND(LINALG_LINEAR_SOLVER_LIB LINSLV)

# Eigen solver module
SET_LINALG_BACKEND(LINALG_EIGENSOLVER_LIB EIGSLV)

####### /LINALG

# prefer original LAPACK, if needed
OPTION(USE_ORIGINAL_LAPACK "Original LAPACK" OFF)
FIND_PACKAGE(LAPACK)
IF (LAPACK_FOUND)
  # find out whether it is Accelerate.framework we found for LaPack/BLAS
  IF("${LAPACK_LIBRARIES}" MATCHES ".*/Accelerate.framework$")
    SET(HAVE_MVEC 1)
    SET(HAVE_CATLAS 1)
    SET(HAVE_LAPACK 1)
    SHOGUN_LINK_LIBS(${LAPACK_LIBRARIES})
  ELSEIF("${LAPACK_LIBRARIES}" MATCHES ".*/mkl_.*")
    SET(HAVE_LAPACK 1)
    SET(HAVE_MKL 1)
    SHOGUN_LINK_LIBS(${LAPACK_LIBRARIES})
  ELSE()
    FIND_PACKAGE(Atlas)
    IF(Atlas_FOUND)
      SET(HAVE_ATLAS 1)
      SET(HAVE_LAPACK 1)
      SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC ${Atlas_INCLUDE_DIRS})
      IF(USE_ORIGINAL_LAPACK)
        SET(ATLAS_CLAPACK_LIBRARY)
        FOREACH(ITR ${Atlas_LIBRARIES})
          IF(ITR MATCHES ".*atlas.*lapack.*" OR ITR MATCHES ".*lapack.*atlas.*")
            STRING(REGEX REPLACE "lapack" "clapack" ITR ${ITR})
            LIST(APPEND ATLAS_CLAPACK_LIBRARY ${ITR})
          ENDIF(ITR MATCHES ".*atlas.*lapack.*" OR ITR MATCHES ".*lapack.*atlas.*")
        ENDFOREACH(ITR ${ATLAS_LIBRARIES})
        MESSAGE(STATUS "using ATLAS-CLAPACK from: ${ATLAS_CLAPACK_LIBRARY}")
        SHOGUN_LINK_LIBS(${LAPACK_LIBRARIES} ${ATLAS_CLAPACK_LIBRARY} ${Atlas_LIBRARIES})
      ELSE(USE_ORIGINAL_LAPACK)
        SHOGUN_LINK_LIBS(${LAPACK_LIBRARIES} ${Atlas_LIBRARIES})
      ENDIF(USE_ORIGINAL_LAPACK)
    ELSE()
      FIND_PACKAGE(CBLAS)
      if(CBLAS_LIBRARY)
        SET(HAVE_LAPACK 1)
        SHOGUN_LINK_LIBS(${LAPACK_LIBRARIES} ${CBLAS_LIBRARY})
      else()
        SET(HAVE_LAPACK 1)
        SHOGUN_LINK_LIBS(${LAPACK_LIBRARIES})
        endif()
    ENDIF()
  ENDIF()
ENDIF()

SHOGUN_DEPENDENCIES(
  LIBRARY GLPK
  SCOPE PRIVATE
  CONFIG_FLAG USE_GLPK)

SHOGUN_DEPENDENCIES(
  LIBRARY CPLEX
  SCOPE PRIVATE
  CONFIG_FLAG USE_CPLEX)

SHOGUN_DEPENDENCIES(
  LIBRARY ARPACK
  SCOPE PRIVATE
  CONFIG_FLAG HAVE_ARPACK)

SHOGUN_DEPENDENCIES(
  LIBRARY Mosek
  SCOPE PRIVATE
  CONFIG_FLAG USE_MOSEK)

SHOGUN_DEPENDENCIES(
  LIBRARY Protobuf
  SCOPE PUBLIC
  CONFIG_FLAG HAVE_PROTOBUF)

IF (PROTOBUF_FOUND AND ENABLE_PROTOBUF)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/io/protobuf)
  FILE(GLOB protobuf_src ${CMAKE_CURRENT_SOURCE_DIR}/io/protobuf/*.proto)
  FOREACH(FIL ${protobuf_src})
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    SET(name "${CMAKE_CURRENT_BINARY_DIR}/io/protobuf/${FIL_WE}")

    ADD_CUSTOM_COMMAND(
       OUTPUT "${name}.pb.cc" "${name}.pb.h"
       COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${FIL}
         --cpp_out ${CMAKE_CURRENT_BINARY_DIR}/io/protobuf
         --proto_path ${CMAKE_CURRENT_SOURCE_DIR}/io/protobuf
       COMMENT "Running C++ protocol buffer compiler on ${FIL_WE}.proto"
      )
    ADD_CUSTOM_TARGET("${FIL_WE}Protobuf" DEPENDS "${name}.pb.cc")
    add_dependencies(libshogun "${FIL_WE}Protobuf")
    set_source_files_properties(${name}.pb.cc PROPERTIES GENERATED ON)
    target_sources(libshogun PRIVATE ${name}.pb.cc)
  ENDFOREACH()
ENDIF()

# JSON
OPTION(BUNDLE_JSON "Bundle JSON" OFF)
IF(BUNDLE_JSON)
  # FIXME: BUNDLING IS NOT PROPER!
  # see how eigen is being done
  include(external/JSON)
  SET(HAVE_JSON 1)
  LIST(APPEND INCLUDES ${JSON_INCLUDE_DIRS})
  SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${JSON_LDFLAGS})
ELSE()
  SHOGUN_DEPENDENCIES(
    LIBRARY JSON
    SCOPE PUBLIC
    CONFIG_FLAG HAVE_JSON
    VERSION 0.11)
ENDIF()

SHOGUN_DEPENDENCIES(
  LIBRARY LibXml2
  SCOPE PUBLIC
  CONFIG_FLAG HAVE_XML)

SHOGUN_DEPENDENCIES(
  LIBRARY HDF5
  SCOPE PUBLIC
  CONFIG_FLAG HAVE_HDF5)

SHOGUN_DEPENDENCIES(
  LIBRARY CURL
  SCOPE PRIVATE
  CONFIG_FLAG HAVE_CURL)

SHOGUN_DEPENDENCIES(
  LIBRARY ZLIB
  SCOPE PRIVATE
  CONFIG_FLAG USE_GZIP)

SHOGUN_DEPENDENCIES(
  LIBRARY BZip2
  SCOPE PRIVATE
  CONFIG_FLAG USE_BZIP2)

SHOGUN_DEPENDENCIES(
  LIBRARY LibLZMA
  SCOPE PRIVATE
  CONFIG_FLAG USE_LZMA)

SHOGUN_DEPENDENCIES(
  LIBRARY SNAPPY
  SCOPE PRIVATE
  CONFIG_FLAG USE_SNAPPY)

SHOGUN_DEPENDENCIES(
  LIBRARY LZO
  SCOPE PRIVATE
  CONFIG_FLAG USE_LZO)

#integration
OPTION(OpenCV "OpenCV Integration" OFF)
IF (OpenCV)
  # find OpenCV library if intended by the user.
  FIND_PACKAGE(OpenCV)
  IF (OpenCV_FOUND)
    SET(HAVE_OPENCV 1)
    target_include_directories(libshogun PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(shogun PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(shogun PUBLIC ${OpenCV_LIBRARIES})
    if (EXISTS shogun-static)
      target_link_libraries(shogun-static PUBLIC ${OpenCV_LIBRARIES})
    endif()
  ENDIF()
ENDIF()

OPTION(BUNDLE_NLOPT "Bundle NLOPT" OFF)
IF(BUNDLE_NLOPT)
  # FIXME: BUNDLING IS NOT PROPER!
  # see how eigen is being done
  include(external/NLopt)
  SET(HAVE_NLOPT 1)
  LIST(APPEND INCLUDES ${NLOPT_INCLUDE_DIR})
  SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${NLOPT_LIBRARIES})
ELSE()
  SHOGUN_DEPENDENCIES(
    LIBRARY NLopt
    SCOPE PRIVATE
    CONFIG_FLAG HAVE_NLOPT)
ENDIF()

SHOGUN_DEPENDENCIES(
  LIBRARY LpSolve
  SCOPE PRIVATE
  CONFIG_FLAG USE_LPSOLVE)

SHOGUN_DEPENDENCIES(
  LIBRARY ColPack
  SCOPE PRIVATE
  CONFIG_FLAG HAVE_COLPACK)

IF(USE_ARPREC AND NOT LICENSE_GPL_SHOGUN)
  MESSAGE(FATAL_ERROR "ARPREC can only be used if USE_GPL_SHOGUN is enabled ")
ENDIF()
SHOGUN_DEPENDENCIES(
  LIBRARY ARPREC
  SCOPE PRIVATE
  CONFIG_FLAG HAVE_ARPREC)

###### checks for random
CHECK_FUNCTION_EXISTS(arc4random HAVE_ARC4RANDOM)
IF(NOT HAVE_ARC4RANDOM)
    # assume that /dev/random is non-blocking if /dev/urandom does not exist
    if(EXISTS /dev/urandom)
      set(DEV_RANDOM "/dev/urandom" CACHE INTERNAL "" FORCE)
    elseif( EXISTS /dev/random )
      set(DEV_RANDOM "/dev/random" CACHE INTERNAL "" FORCE)
    endif()
ENDIF()


################## linker optimisations
OPTION(INCREMENTAL_LINKING "Enable incremantal linking")
SET(INCREMENTAL_LINKING_DIR ${CMAKE_BINARY_DIR}/linker_cache
  CACHE PATH "Incremental linking cache directory")
OPTION(ENABLE_LTO "Enable link-time optimisation")

include(LDGOLD)
if (LDGOLD_FOUND)
  SET_LDGOLD(shogun shogun-static libshogun shogun_deps)
endif()

include(LTO)
if (ENABLE_LTO AND LTO_FOUND)
  MESSAGE(STATUS "Link-time optimisation enabled")
  SET_LTO(shogun shogun-static shogun_deps libshogun)
endif()

# FIXME: remove all the INCLUDES/SYSTEM_INCLUDES/POSTLINKFLAGS
include_directories(${INCLUDES})
target_link_libraries(shogun PUBLIC ${POSTLINKFLAGS})

#############################################
FIND_PACKAGE(Ctags)
IF (CTAGS_FOUND)
    SET(CTAGS_FILE ${CMAKE_CURRENT_BINARY_DIR}/tags CACHE INTERNAL "" FORCE)
    ADD_CUSTOM_COMMAND(OUTPUT ${CTAGS_FILE}
        COMMAND ${CTAGS_EXECUTABLE} -f ${CTAGS_FILE}
        # functions, classes, macroses, enumerations, enumerators, typedefs
        --c++-kinds=fcdget
        -R ${CMAKE_CURRENT_SOURCE_DIR})
    ADD_CUSTOM_TARGET(ctags DEPENDS ${CTAGS_FILE})
    SET_SOURCE_FILES_PROPERTIES(${CTAGS_FILE} PROPERTIES GENERATED 1)
ENDIF()

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/lib/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h @ONLY)

INSTALL(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
  DESTINATION ${INCLUDE_INSTALL_DIR}
  COMPONENT headers
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  PATTERN "CMakeFiles*" EXCLUDE
  PATTERN ".settings" EXCLUDE)

IF (LIB_INSTALL_DIR)
  SET(SHOGUN_LIB_INSTALL ${LIB_INSTALL_DIR})
ELSE()
  SET(SHOGUN_LIB_INSTALL lib)
ENDIF()

# set the desidered targets to be installed
set(INSTALL_TARGETS shogun)
if (LIBSHOGUN_BUILD_STATIC)
  LIST(APPEND INSTALL_TARGETS shogun-static)
endif()

INSTALL(
  TARGETS ${INSTALL_TARGETS}
  EXPORT ShogunTargets
  ARCHIVE DESTINATION ${SHOGUN_LIB_INSTALL}
  LIBRARY DESTINATION ${SHOGUN_LIB_INSTALL}
  INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
)

set(CONFIG_PACKAGE_DIR ${SHOGUN_LIB_INSTALL}/cmake/shogun)
configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake/ShogunConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/ShogunConfig.cmake
  INSTALL_DESTINATION ${SHOGUN_LIB_INSTALL}/cmake/shogun
  PATH_VARS INCLUDE_INSTALL_DIR CONFIG_PACKAGE_DIR)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ShogunConfigVersion.cmake"
  VERSION ${VERSION}
  COMPATIBILITY SameMajorVersion
)

export(EXPORT ShogunTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/ShogunTargets.cmake"
  NAMESPACE shogun::
)

install(
  EXPORT ShogunTargets
  FILE ShogunTargets.cmake
  NAMESPACE shogun::
  DESTINATION ${CONFIG_PACKAGE_DIR}
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ShogunConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ShogunConfigVersion.cmake"
  DESTINATION ${CONFIG_PACKAGE_DIR}
)
