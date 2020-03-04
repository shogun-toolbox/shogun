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

# TODO: use explicit export instead of this
OPTION(LIBSHOGUN_BUILD_STATIC "Build libshogun static library")
OPTION(DISABLE_SSE "Disable SSE and SSE2 features.")
OPTION(BUILD_BENCHMARKS "Build benchmarks" OFF)
SET(PARAMETER_BACKEND "VECTOR" CACHE STRING "Parameter map backend type (MAP, VECTOR, SORTED_VECTOR)")

IF (PARAMETER_BACKEND STREQUAL "VECTOR")
  SET(USE_VECTOR_BACKEND 1)
ELSEIF(PARAMETER_BACKEND STREQUAL "MAP")
  SET(USE_MAP_BACKEND 1)
ELSEIF(PARAMETER_BACKEND STREQUAL "SORTED_VECTOR")
  SET(USE_SORTED_VECTOR_BACKEND 1)
ELSE()
  MESSAGE(FATAL_ERROR "Unknown parameter backend type!")
ENDIF()

SET(SHOGUN_CORE_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/base/base_types.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/class_list.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/constraint.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/library.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/macros.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/manifest.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/metaclass.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/Parallel.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/progress.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/range.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/SGObject.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/ShogunEnv.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/unique.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/variant.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/Version.h
  ${CMAKE_CURRENT_SOURCE_DIR}/distance/Distance.h
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluation/Evaluation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/evaluation/EvaluationResult.h
  ${CMAKE_CURRENT_SOURCE_DIR}/features/Features.h
  ${CMAKE_CURRENT_SOURCE_DIR}/features/Subset.h
  ${CMAKE_CURRENT_SOURCE_DIR}/features/SubsetStack.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/Serializable.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/File.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/ShogunErrc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/fmt/fmt.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/fs/FileSystem.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/fs/FileSystemRegistry.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/fs/Path.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/fs/PosixFileSystem.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/fs/WindowsFileSystem.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/serialization/Deserializer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/serialization/Serializer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/stream/InputStream.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/stream/OutputStream.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/stream/FileInputStream.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/stream/FileOutputStream.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/stream/BufferedInputStream.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/stream/BufferedOutputStream.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/SGIO.h
  ${CMAKE_CURRENT_SOURCE_DIR}/io/visitors/ToStringVisitor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/Kernel.h
  ${CMAKE_CURRENT_SOURCE_DIR}/labels/Labels.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/exception/ShogunException.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/any.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/basetag.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/cpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/DataType.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/DynamicArray.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/DynamicObjectArray.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/GPUMatrix.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/GPUVector.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/memory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/RefCount.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/RxCppHeader.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/sg_types.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/SGCachedVector.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/SGMatrix.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/SGMatrixList.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/SGNDArray.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/SGReferencedData.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/SGSparseMatrix.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/SGSparseVector.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/SGVector.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Signal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/StoppableSGObject.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/tag.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Time.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Tokenizer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/type_case.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/View.h
  ${CMAKE_CURRENT_SOURCE_DIR}/loss/LossFunction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/machine/Machine.h
  ${CMAKE_CURRENT_SOURCE_DIR}/machine/gp/LikelihoodModel.h
  ${CMAKE_CURRENT_SOURCE_DIR}/machine/Pipeline.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mathematics/linalg/internal/Block.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mathematics/linalg/LinalgBackendBase.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mathematics/linalg/LinalgNamespace.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mathematics/linalg/LinalgEnums.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mathematics/linalg/LinalgMacros.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mathematics/linalg/SGLinalg.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mathematics/Math.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mathematics/RandomMixin.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mathematics/RandomNamespace.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mathematics/Seedable.h
  ${CMAKE_CURRENT_SOURCE_DIR}/multiclass/MulticlassStrategy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/transformer/Transformer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/visitors/FilterVisitor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/clone.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/converters.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/enumerate.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/factory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/hash.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/iterators.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/system.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/traits.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/zip_iterator.h
)

IF(WIN32)
  LIST(APPEND SHOGUN_CORE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/util/windows_wchar.h)
ENDIF()

SET(SHOGUN_CORE_SOURCES
  base/class_list.cpp
  base/library.cpp
  base/manifest.cpp
  base/Parallel.cpp
  base/SGObject.cpp
  base/ShogunEnv.cpp
  base/Version.cpp
  distance/Distance.cpp
  features/Features.cpp
  features/Subset.cpp
  features/SubsetStack.cpp
  io/fs/FileSystemRegistry.cpp
  io/fs/Path.cpp
  io/fs/PosixFileSystem.cpp
  io/fs/WindowsFileSystem.cpp
  io/File.cpp
  io/ShogunErrc.cpp
  io/serialization/Deserializer.cpp
  io/serialization/Serializer.cpp
  io/stream/InputStream.cpp
  io/stream/BufferedInputStream.cpp
  io/stream/FileInputStream.cpp
  io/stream/OutputStream.cpp
  io/SGIO.cpp
  io/visitors/ToStringVisitor.cpp
  kernel/Kernel.cpp
  labels/Labels.cpp
  lib/exception/ShogunException.cpp
  lib/any.cpp
  lib/DataType.cpp
  lib/GPUMatrix.cpp
  lib/GPUVector.cpp
  lib/memory.cpp
  lib/SGMatrix.cpp
  lib/SGMatrixList.cpp
  lib/SGNDArray.cpp
  lib/SGReferencedData.cpp
  lib/SGSparseMatrix.cpp
  lib/SGSparseVector.cpp
  lib/Signal.cpp
  lib/StoppableSGObject.cpp
  lib/SGVector.cpp
  lib/Time.cpp
  lib/Tokenizer.cpp
  machine/Machine.cpp
  machine/gp/LikelihoodModel.cpp
  machine/Pipeline.cpp
  mathematics/Math.cpp
  multiclass/MulticlassStrategy.cpp
  transformer/Transformer.cpp
)

IF (LIB_INSTALL_DIR)
  SET(SHOGUN_LIB_INSTALL ${LIB_INSTALL_DIR})
ELSE()
  SET(SHOGUN_LIB_INSTALL lib)
ENDIF()
set(INCLUDE_INSTALL_DIR include)
set(THIRD_PARTY_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party)
set(SHOGUN_CLING_LIBRARY_DIR "\"${CMAKE_INSTALL_PREFIX}/${SHOGUN_LIB_INSTALL}\"")

if (MSVC OR BUILD_BENCHMARKS)
  SET(LIBSHOGUN_BUILD_STATIC ON
    CACHE BOOL "Build libshogun static library" FORCE)
endif()

# Allow to hide non-bsd compatible codes
OPTION(LICENSE_GPL_SHOGUN "Include GPL codes of Shogun (non-BSD compatible) in build" OFF)
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
            MESSAGE(FATAL_ERROR "Shogun can only be built with GPL codes if the source files are in ${SHOGUN_GPL_INCLUDE_DIR}. Please download the GPL codes or checkout the Git Submodule with 'git submodule update --init'. To exclude GPL codes of Shogun in build set the switch LICENSE_GPL_SHOGUN=OFF.\n")
        ENDIF()
ENDIF()

OPTION(USE_SVMLIGHT "SVMLight" OFF)
IF(USE_SVMLIGHT AND NOT USE_GPL_SHOGUN)
    MESSAGE(FATAL_ERROR "Can only use SVMLight when GPL codes are included. Disable by USE_SVMLIGHT=OFF")
ENDIF()

add_library(shogun-core SHARED ${SHOGUN_CORE_HEADERS} ${SHOGUN_CORE_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h)
generate_export_header(shogun-core BASE_NAME SHOGUN)
set_property(TARGET shogun-core PROPERTY POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(
  shogun-core PROPERTIES
  VERSION ${LIBSHOGUNVER}
  SOVERSION ${LIBSHOGUNSO}
  INTERFACE_shogun_MAJOR_VERSION ${SHOGUN_VERSION_MAJOR})
set_property(TARGET shogun-core APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${SHOGUN_VERSION_MAJOR})
CommonOutputDir(shogun-core)

add_subdirectory(mathematics/graph)

# add target to compile the libshogun sources
add_library(libshogun OBJECT EXCLUDE_FROM_ALL ${LIBSHOGUN_SRC} ${LIBSHOGUN_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h)
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
add_dependencies(shogun-core version)

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

add_library(shogun SHARED EXCLUDE_FROM_ALL $<TARGET_OBJECTS:libshogun> ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h)
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

# add target for static library if enabled
if (LIBSHOGUN_BUILD_STATIC)
  add_library(shogun-static STATIC EXCLUDE_FROM_ALL $<TARGET_OBJECTS:libshogun> ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h)
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
    target_link_libraries(shogun-static PUBLIC winmm.lib Shlwapi.lib)
    target_compile_definitions(shogun-static PUBLIC _ENABLE_EXTENDED_ALIGNED_STORAGE)
  endif()
ENDIF()

IF(MSVC)
  target_compile_definitions(shogun PUBLIC _ENABLE_EXTENDED_ALIGNED_STORAGE)
  target_compile_definitions(shogun_deps INTERFACE _ENABLE_EXTENDED_ALIGNED_STORAGE)
  IF(MSVC_VERSION VERSION_LESS "1700")
    include(external/MSIntTypes)
    SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC
      $<BUILD_INTERFACE:${MSINTTYPES_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include/shogun/third_party/MSIntTypes>
    )
  ENDIF()

  target_link_libraries(shogun PUBLIC winmm Shlwapi)
  target_link_libraries(shogun-core PUBLIC winmm Shlwapi)
  target_link_libraries(shogun_deps INTERFACE winmm Shlwapi)
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
  else ()
    SHOGUN_LINK_LIBS(OpenMP::OpenMP_CXX)
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

###### MALLOC LIBRARY ###########
SET(EXTERNAL_MALLOC_CFLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
if (MALLOC_REPLACEMENT MATCHES "Jemalloc")
  include(external/Jemalloc)

  SET(USE_JEMALLOC 1)
  SET(HAVE_ALIGNED_MALLOC 1)
  SET(CMAKE_CXX_FLAGS "${EXTERNAL_MALLOC_CFLAGS} ${CMAKE_CXX_FLAGS}")
  target_include_directories(libshogun PRIVATE ${Jemalloc_INCLUDE_DIR})
  target_include_directories(shogun PRIVATE ${Jemalloc_INCLUDE_DIR})
  target_link_libraries(shogun PRIVATE ${Jemalloc_LIBRARIES})
  if (EXISTS shogun-static)
    target_link_libraries(shogun-static PRIVATE ${Jemalloc_LIBRARIES})
  endif()
elseif(MALLOC_REPLACEMENT MATCHES "TCMalloc")
  include(external/TCMalloc)

  SET(USE_TCMALLOC 1)
  SET(HAVE_ALIGNED_MALLOC 1)
  SET(CMAKE_CXX_FLAGS "${EXTERNAL_MALLOC_CFLAGS} ${CMAKE_CXX_FLAGS}")
  target_include_directories(libshogun PRIVATE ${TCMalloc_INCLUDE_DIR})
  target_include_directories(shogun PRIVATE ${TCMalloc_INCLUDE_DIR})
  target_link_libraries(shogun PRIVATE ${TCMalloc_LIBRARIES})
  if (EXISTS shogun-static)
    target_link_libraries(shogun-static PRIVATE ${TCMalloc_LIBRARIES})
  endif()
elseif(MALLOC_REPLACEMENT MATCHES "Hoard")
  find_package(Hoard)
  if (Hoard_FOUND)
    SET(USE_HOARD 1)
    SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${Hoard_LIBRARIES})
  else ()
  message(FATAL_ERROR "Unable to use hoard malloc: library not found")
  endif ()
else()
  message(STATUS "Using system's malloc")
  include(CheckCXXCompilerFlag)

  # FIXME: add support for aligned new
  # this is c++17
  #CHECK_CXX_COMPILER_FLAG("-faligned-new" HAVE_ALIGNED_NEW)
  CHECK_CXX_SOURCE_COMPILES("
  #include <cstdio>
  #include <cstdlib>

  int main()
  {
      int* p2 = static_cast<int*>(std::aligned_alloc(1024, 1024));
      std::free(p2);
      return 0;
  }
  " HAVE_STD_ALIGNED_ALLOC)

  if (NOT HAVE_STD_ALIGNED_ALLOC)
    if (MSVC)
      SET(HAVE_ALIGNED_MALLOC 1)
    else()
      check_function_exists("posix_memalign" HAVE_POSIX_MEMALIGN)
      if (HAVE_POSIX_MEMALIGN)
        SET(HAVE_ALIGNED_MALLOC 1)
      endif()
    endif()
  else()
    SET(HAVE_ALIGNED_MALLOC 1)
  endif()
endif()

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
    # TODO: the dependencies should be refactored
    add_dependencies(shogun-core rxcpp)
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
#include(ShogunFindLAPACK)

CHECK_CXX_SOURCE_COMPILES("#include <variant>\n int main(int argc, char** argv) { std::variant<int, float> v; return 0; }" HAVE_STD_VARIANT)
# variant
IF (NOT HAVE_STD_VARIANT)
  include(external/variant)
  SHOGUN_INCLUDE_DIRS(SCOPE PUBLIC SYSTEM
	$<BUILD_INTERFACE:${VARIANT_INCLUDE_DIR}>
	$<INSTALL_INTERFACE:include/shogun/third_party/variant>
  )
ENDIF()

# spdlog
FIND_PACKAGE(spdlog CONFIG)
IF (NOT spdlog_FOUND)
    include(external/spdlog)
    target_link_libraries(shogun-core PUBLIC spdlog)
ELSE()
    SET(USE_EXTERNAL_SPDLOG 1)
    target_link_libraries(shogun PRIVATE spdlog::spdlog)
    if (EXISTS shogun-static)
	    target_link_libraries(shogun-static PRIVATE spdlog::spdlog)
    endif()
    # TODO: the dependencies should be refactored
    target_link_libraries(shogun-core PUBLIC spdlog::spdlog)
ENDIF()

SHOGUN_DEPENDENCIES(
  LIBRARY GLPK
  SCOPE PRIVATE
  CONFIG_FLAG USE_GLPK)

SHOGUN_DEPENDENCIES(
  LIBRARY LibArchive
  SCOPE PRIVATE
  CONFIG_FLAG HAVE_LIBARCHIVE)

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

# RapidJSON
find_package(RapidJSON)
if (NOT RapidJSON_FOUND)
  include(external/RapidJSON)
endif()
SHOGUN_INCLUDE_DIRS(SCOPE PRIVATE ${RAPIDJSON_INCLUDE_DIRS})

include(external/bitsery)
SHOGUN_INCLUDE_DIRS(SCOPE PRIVATE ${BITSERY_INCLUDE_DIR})

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

############################ HMM
OPTION(USE_HMMDEBUG "HMM debug mode" OFF)
OPTION(USE_HMMCACHE "HMM cache" ON)
OPTION(USE_HMMPARALLEL "Parallel structures in hmm training" OFF)
IF(USE_HMMPARALLEL)
  SET(USE_HMMPARALLEL_STRUCTURES 1)
ENDIF()
# Viterbi path debug
OPTION(USE_PATHDEBUG "Viterbi path debugging" OFF)
# big states
OPTION(USE_BIGSTATES "Big (16bit) state" ON)
OPTION(USE_LOGSUMARRAY "Use sum array, supposed to be a bit more accurate" OFF)

#kernelcache to use 4-byte-floating-point values instead of 8-byte-doubles
OPTION(USE_SHORTREAL_KERNELCACHE "Kernelcache to use 4-byte-floating-point values instead of 8-byte-doubles" ON)

# add bundled libraries to dependency
if(SHOGUN_DEPENDS)
  add_dependencies(libshogun ${SHOGUN_DEPENDS})
  add_dependencies(shogun-core ${SHOGUN_DEPENDS})
endif()

OPTION(USE_LOGCACHE "Use (1+exp(x)) log cache (is much faster but less accurate)" OFF)
################## linker optimisations
OPTION(INCREMENTAL_LINKING "Enable incremantal linking")
SET(INCREMENTAL_LINKING_DIR ${CMAKE_BINARY_DIR}/linker_cache
  CACHE PATH "Incremental linking cache directory")
OPTION(ENABLE_LTO "Enable link-time optimisation")

include(LDGOLD)
if (LDGOLD_FOUND)
  SET_LDGOLD(shogun shogun-static libshogun shogun_deps shogun-core)
endif()

include(LTO)
if (ENABLE_LTO AND LTO_FOUND)
  MESSAGE(STATUS "Link-time optimisation enabled")
  SET_LTO(shogun shogun-static shogun_deps libshogun)
endif()

# FIXME: remove all the INCLUDES/SYSTEM_INCLUDES/POSTLINKFLAGS
include_directories(${INCLUDES})
target_link_libraries(shogun PUBLIC ${POSTLINKFLAGS})

##### testing
IF (ENABLE_TESTING)
  enable_testing()
  FIND_PACKAGE(Jinja2)
  FIND_PACKAGE(Ctags)

  # Find GTEST and GMOCK frameworks
  include(external/GoogleTestNMock)

  add_library(shogun_test_main ${CMAKE_CURRENT_SOURCE_DIR}/util/unittest_main.cc)
  if (APPLE)
    target_link_libraries(shogun_test_main PUBLIC
      gmock
      gtest
      shogun-core
    )
  elseif(MSVC)
    target_link_libraries(shogun_test_main PUBLIC
      gmock
      gtest
      shogun-core
    )
  else()
    target_link_libraries(shogun_test_main PUBLIC
      gmock
      gtest
      shogun-core
    )
  endif()

  set(SHOGUN_TEST_LINK_LIBS shogun_test_main)

  ADD_SHOGUN_TEST(lib/any_test)

  ADD_SHOGUN_TEST(base/Plugin_test)
  add_library(testplugin SHARED ${CMAKE_CURRENT_SOURCE_DIR}/util/MockLibrary.cpp)
  target_link_libraries(testplugin shogun-core)
  CommonOutputDir(testplugin)

  add_library(testpluginwomanifest SHARED ${CMAKE_CURRENT_SOURCE_DIR}/util/MockLibraryWOManifest.cpp)
  target_link_libraries(testpluginwomanifest shogun-core)
  set_property(TARGET testplugin PROPERTY POSITION_INDEPENDENT_CODE TRUE)
  add_dependencies(Plugin_test testplugin testpluginwomanifest)
  CommonOutputDir(testpluginwomanifest)

  target_compile_definitions(Plugin_test PRIVATE
    SHARED_LIB_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}"
    SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}")

  ADD_SHOGUN_TEST(base/class_list_test)
  add_dependencies(class_list_test testplugin testpluginwomanifest)
ENDIF()

################# benchmarking
IF (BUILD_BENCHMARKS)
  IF (NOT BUILD_TYPE_UC STREQUAL "RELEASE")
    MESSAGE(WARNING "You are building benchmark against non-Release mode!"
      "The results might not reflect the actual runtimes...")
  ENDIF()
  find_package(benchmark CONFIG REQUIRED)

  enable_testing()
  add_library(shogun_benchmark_main ${CMAKE_CURRENT_SOURCE_DIR}/util/benchmark_main.cc)
  if (APPLE)
    target_link_libraries(shogun_benchmark_main benchmark::benchmark shogun-static)
  elseif(MSVC)
    target_link_libraries(shogun_benchmark_main
      benchmark::benchmark
      Shlwapi.lib
      shogun-static
    )
  else()
    target_link_libraries(shogun_benchmark_main
      benchmark::benchmark
      pthread
      shogun-static
    )
  endif()

  set(SHOGUN_BENCHMARK_LINK_LIBS shogun_benchmark_main)

  ADD_SHOGUN_BENCHMARK(features/RandomFourierDotFeatures_benchmark)
  ADD_SHOGUN_BENCHMARK(features/hashed/HashedDocDotFeatures_benchmark)
  ADD_SHOGUN_BENCHMARK(lib/RefCount_benchmark)
  ADD_SHOGUN_BENCHMARK(mathematics/linalg/backend/eigen/BasicOps_benchmark)
  ADD_SHOGUN_BENCHMARK(mathematics/linalg/backend/eigen/Misc_benchmark)
  ADD_SHOGUN_BENCHMARK(lib/SGMatrix_benchmark)
  ADD_SHOGUN_BENCHMARK(util/PutPerceptron_benchmark)
  ADD_SHOGUN_BENCHMARK(util/ZipIterator_benchmark)
ENDIF()

#############################################
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

# set the desidered targets to be installed
set(INSTALL_TARGETS shogun-core)
# TODO: shogun-core should have a static target
# if (LIBSHOGUN_BUILD_STATIC)
#  LIST(APPEND INSTALL_TARGETS shogun-static)
# endif()

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
