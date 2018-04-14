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
set(THIRD_PARTY_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party)

if (MSVC)
  SET(LIBSHOGUN_BUILD_STATIC ON
    CACHE BOOL "Build libshogun static library" FORCE)
endif()

# Allow to hide non-bsd compatible codes
OPTION(LICENSE_GPL_SHOGUN "Include GPL codes of Shogun (non-BSD compatible) in build" ON)
SET(USE_GPL_SHOGUN 0)
IF (LICENSE_GPL_SHOGUN)
        SET(SHOGUN_GPL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/gpl)
        IF(EXISTS "${SHOGUN_GPL_INCLUDE_DIR}/shogun")
            SET(USE_GPL_SHOGUN 1)

            FILE(GLOB_RECURSE GPL_LIBSHOGUN_SRC ${SHOGUN_GPL_INCLUDE_DIR}/*.${EXT_SRC_CPP} ${SHOGUN_GPL_INCLUDE_DIR}/*.${EXT_SRC_C})
            FILE(GLOB_RECURSE GPL_LIBSHOGUN_HEADERS ${SHOGUN_GPL_INCLUDE_DIR}/*.${EXT_SRC_HEADER})

            LIST(APPEND LIBSHOGUN_SRC ${GPL_LIBSHOGUN_SRC})
            LIST(APPEND LIBSHOGUN_HEADERS ${GPL_LIBSHOGUN_HEADERS})
            INCLUDE_DIRECTORIES(${SHOGUN_GPL_INCLUDE_DIR})

            SET(USE_GPL_SHOGUN 1)
        ELSE()
            MESSAGE(FATAL_ERROR "Shogun can only be built with GPL codes if the source files are in ${SHOGUN_GPL_INCLUDE_DIR}. Please download or disable with LICENSE_GPL_SHOGUN=OFF.\n")
        ENDIF()
ENDIF()

OPTION(USE_SVMLIGHT "SVMLight" ON)
IF(USE_SVMLIGHT AND NOT USE_GPL_SHOGUN)
    MESSAGE(FATAL_ERROR "Can only use SVMLight when GPL codes are included. Disable by USE_SVMLIGHT=OFF")
ENDIF()

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
      -P ${CMAKE_SOURCE_DIR}/cmake/version.cmake
    COMMENT "Generating version header"
)
add_dependencies(libshogun version)

# generate source files from templates
FOREACH(template ${LIBSHOGUN_SRC_TMP})
  STRING(REGEX REPLACE ${EXT_CPP_TMP} ${EXT_SRC_CPP} generated_cpp "${template}")
  STRING(REGEX REPLACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} generated_cpp "${generated_cpp}")
  STRING(REGEX REPLACE ${EXT_CPP_TMP} ${EXT_CPP_PY} generator_script "${template}")
  STRING(REGEX REPLACE ".*/(.*).${EXT_CPP_TMP}" "\\1" generated_target "${template}")

  IF (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/headers_list.txt)
    FILE(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/headers_list.txt)
  ENDIF()
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
IF (${USE_GPL_SHOGUN})
    target_include_directories(shogun PUBLIC $<BUILD_INTERFACE:${SHOGUN_GPL_INCLUDE_DIR}>)
ENDIF()

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

# create shogun_deps meta target to store all the
# dependencies of shogun itself, this is used in unit test
add_library(shogun_deps INTERFACE)
target_link_libraries(shogun_deps INTERFACE shogun)
target_include_directories(shogun_deps INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
)
IF (${USE_GPL_SHOGUN})
    target_include_directories(shogun_deps INTERFACE $<BUILD_INTERFACE:${SHOGUN_GPL_INCLUDE_DIR}>)
ENDIF()

# add bundled libraries to dependency
if(SHOGUN_DEPENDS)
  add_dependencies(libshogun ${SHOGUN_DEPENDS})
endif()

# add target for static library if enabled
if (LIBSHOGUN_BUILD_STATIC)
  add_library(shogun-static STATIC $<TARGET_OBJECTS:libshogun> ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h)
  set_property(TARGET shogun-static PROPERTY OUTPUT_NAME shogun)
  target_include_directories(shogun-static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
    $<INSTALL_INTERFACE:include/shogun>
  )
  IF (${USE_GPL_SHOGUN})
    target_include_directories(shogun-static PUBLIC $<BUILD_INTERFACE:${SHOGUN_GPL_INCLUDE_DIR}>)
  ENDIF()

  if(MSVC)
    target_link_libraries(shogun-static PUBLIC winmm.lib)
  endif()
ENDIF()

IF(MSVC)
  IF(MSVC_VERSION VERSION_LESS "1700")
    include(external/MSIntTypes)
    SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC
      $<BUILD_INTERFACE:${MSINTTYPES_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include/shogun/third_party/MSIntTypes>
    )
  ENDIF()

  # bundle dirent
  include(external/MSDirent)
  SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC
    $<BUILD_INTERFACE:${MSDIRENT_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include/shogun/third_party/MSDirent>
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

FIND_PACKAGE(CxaDemangle)

# check SSE and SSE2 intrinsics header
IF((NOT CYGWIN) AND (NOT DISABLE_SSE))
  CHECK_INCLUDE_FILE(xmmintrin.h HAVE_BUILTIN_VECTOR)
  CHECK_INCLUDE_FILE(emmintrin.h HAVE_SSE2)
ENDIF((NOT CYGWIN) AND (NOT DISABLE_SSE))

FIND_PACKAGE(CxaDemangle)
############################ std lib functions
include (CheckCXXSymbolExists)
CHECK_CXX_SYMBOL_EXISTS(signgam "cmath" HAVE_DECL_SIGNGAM)
CHECK_CXX_SYMBOL_EXISTS(fdopen "stdio.h" HAVE_FDOPEN)

# check for math functions
IF(UNIX)
  SET(CMAKE_REQUIRED_LIBRARIES m)
ENDIF()
CHECK_FUNCTION_EXISTS(lgammal HAVE_LGAMMAL)

############################ external dependencies

####### LINALG
FIND_PACKAGE(Eigen3 ${EIGEN_VERSION_MINIMUM})
IF(NOT EIGEN3_FOUND)
  include(external/Eigen3)
  SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC SYSTEM
    $<BUILD_INTERFACE:${EIGEN_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include/shogun/third_party/eigen>
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

####### /LINALG

FIND_PACKAGE(rxcpp)
IF(NOT rxcpp_FOUND)
    include(external/rxcpp)
    SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC SYSTEM
            $<BUILD_INTERFACE:${rxcpp_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include/shogun/third_party/rxcpp>
            )
ELSE()
    SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC SYSTEM ${rxcpp_INCLUDE_DIR})
ENDIF()

# TFLogger package
FIND_PACKAGE(TFLogger 0.1.0 CONFIG)
IF (TFLogger_FOUND)
    SET(HAVE_TFLOGGER 1)
    SHOGUN_INCLUDE_DIRS(SCOPE PRIVATE SYSTEM ${TFLogger_INCLUDE_DIR})
    target_link_libraries(shogun PRIVATE tflogger::tflogger)
ENDIF()

#### LAPACK
include(ShogunFindLAPACK)

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

# RapidJSON
include(external/RapidJSON)
SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC SYSTEM
  $<BUILD_INTERFACE:${RAPIDJSON_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include/shogun/lib/external/eigen>
)

SHOGUN_DEPENDENCIES(
  LIBRARY LibXml2
  SCOPE PUBLIC
  CONFIG_FLAG HAVE_XML)

if (NOT WIN32)
  # FIXME: HDF5 linking on WIN32 is broken.
  # at least with the hdf5 supplied in anaconda
  SHOGUN_DEPENDENCIES(
    LIBRARY HDF5
    SCOPE PUBLIC
    CONFIG_FLAG HAVE_HDF5)
endif ()

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
        # classes, enums, functions
        --c++-kinds=cfgp
        --fields=+im
        --exclude=*.cpp
        --languages=c++
        -R ${CMAKE_SOURCE_DIR})

    ADD_CUSTOM_TARGET(ctags DEPENDS ${CTAGS_FILE})
    SET_SOURCE_FILES_PROPERTIES(${CTAGS_FILE} PROPERTIES GENERATED 1)
ENDIF()

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/lib/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h @ONLY)

LIST(APPEND INCLUDE_HEADERS_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
IF (LICENSE_GPL_SHOGUN)
  LIST(APPEND INCLUDE_HEADERS_DIR_LIST ${SHOGUN_GPL_INCLUDE_DIR}/shogun)
ENDIF()

INSTALL(
  DIRECTORY ${INCLUDE_HEADERS_DIR_LIST}
  DESTINATION ${INCLUDE_INSTALL_DIR}
  COMPONENT headers
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  PATTERN "CMakeFiles*" EXCLUDE
  PATTERN "*.dir" EXCLUDE
  PATTERN "*Debug*" EXCLUDE
  PATTERN "*Release*" EXCLUDE
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
  RUNTIME DESTINATION ${SHOGUN_LIB_INSTALL}
  LIBRARY DESTINATION ${SHOGUN_LIB_INSTALL}
  INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
)

file(TO_CMAKE_PATH ${SHOGUN_LIB_INSTALL}/cmake/shogun CONFIG_PACKAGE_DIR)
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
