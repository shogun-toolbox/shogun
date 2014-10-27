# Instructions:
# $ mkdir build
# $ cd build
# $ cmake ..
# $ make

project(shogun)
cmake_minimum_required(VERSION 2.8.8)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(ShogunUtils)

############# minimum library versions ###################
SET(EIGEN_VERSION_MINIMUM 3.1.2)
SET(VIENNACL_VERSION_MINIMUM 1.5.0)

# CCACHE
OPTION(ENABLE_CCACHE "Enable ccache for compilation" ON)
FIND_PACKAGE(CCache)
if(CCACHE_FOUND AND ENABLE_CCACHE)
	SET(CCACHE "ccache" CACHE STRING "ccache")
	SET(CMAKE_C_COMPILER_ARG1 ${CMAKE_C_COMPILER})
	SET(CMAKE_C_COMPILER ${CCACHE})
	SET(CMAKE_CXX_COMPILER_ARG1 ${CMAKE_CXX_COMPILER})
	SET(CMAKE_CXX_COMPILER ${CCACHE})
	SET(CMAKE_C_FLAGS "${CCACHE_FLAGS} ${CMAKE_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CCACHE_FLAGS} ${CMAKE_CXX_FLAGS}")
endif()

################# VARIABLES #####################
SET(DEFINES "SWIG_TYPE_TABLE=shogun;SHOGUN")
SET(INCLUDES "")

SET(EXT_SRC_HEADER "h")
SET(EXT_SRC_C "c")
SET(EXT_SRC_CPP "cpp")
SET(EXT_SRC_SWIG "_wrap.cxx")
SET(EXT_SRC_TEST "_unittest.cc")
SET(EXT_INP_SWIG "i")
SET(EXT_IF_SWIG "py")
SET(EXT_CPP_TMP "${EXT_SRC_CPP}.templ")
SET(EXT_CPP_SH "${EXT_SRC_CPP}.sh")
SET(EXT_CPP_PY "${EXT_SRC_CPP}.py")
SET(EXT_SRC_TEST_TMP "${EXT_SRC_TEST}.jinja2")

SET(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)
SET(LIBSHOGUN_SRC_DIR ${CMAKE_SOURCE_DIR}/src/shogun)
SET(COMMON_MODULAR_SRC_DIR ${CMAKE_SOURCE_DIR}/src/interfaces/modular/)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src
	${CMAKE_SOURCE_DIR}/src/shogun
	${CMAKE_BINARY_DIR}/src
	${CMAKE_BINARY_DIR}/src/shogun)

# check whether any of the modular interfaces are turned ON
IF (
	PythonModular OR LuaModular OR RModular OR
	OctaveModular OR JavaModular OR PerlModular OR
	RubyModular OR CSharpModular
   )
	IF(RModular)
		FIND_PACKAGE(SWIG 2.0.5 REQUIRED)
	ELSE()
		FIND_PACKAGE(SWIG 2.0.4 REQUIRED)
	ENDIF()

	# SWIG >= 3.0.0 has some new handling with C# (Mono) and breaks
	# typemapping created for earlier versions of SWIG.
	# see: http://www.swig.org/Doc3.0/CSharp.html#CSharp_introduction_swig2_compatibility
	IF(NOT "${SWIG_VERSION}" VERSION_LESS "3.0.0")
		LIST(APPEND CMAKE_SWIG_FLAGS "-DSWIG2_CSHARP")
	ENDIF(NOT "${SWIG_VERSION}" VERSION_LESS "3.0.0")

	# use our own UseSWIG.cmake in order to be able to enable ccache-swig
	SET(SWIG_USE_FILE ${CMAKE_SOURCE_DIR}/cmake/UseSWIG.cmake)
	IF(ENABLE_CCACHE AND CCACHE_SWIG)
		SET(CCACHE_SWIG_EXECUTABLE ${CCACHE_SWIG})
	ENDIF()
	SET(COMPILE_MODULAR_INTERFACE 1)
ENDIF()

# Detect OS
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	SET(DARWIN 1)
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	SET(LINUX 1)
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
	SET(FREEBSD 1)
ENDIF()

# Get processor type, sets MACHINE macro
SET(MACHINE ${CMAKE_SYSTEM_PROCESSOR})

SET(EXT_LIB_SWIG_RUBY_MODULAR ".so")
if(DARWIN)
	SET(EXT_LIB_SWIG_RUBY_MODULAR ".bundle")
ENDIF()

################ COMPILER #######################
# g++ version needs to be => 4.3
IF(CMAKE_COMPILER_IS_GNUCXX)
	# in order to support cmake 2.8.7 and older
	IF(NOT CMAKE_CXX_COMPILER_VERSION)
		include(CheckCompiler)
	ENDIF()
	IF("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "4.3.0")
		MESSAGE(FATAL_ERROR "g++ version is too old")
	ENDIF()
ENDIF()

#Build type
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
       "Choose the type of build, options are: Debug Release Distribution."
       FORCE )
endif()

# set the flags for the build types
IF(MSVC)
	SET(COMPILER_WARNINGS "/Wall")
ELSE()
	SET(COMPILER_WARNINGS "-Wall -Wno-unused-parameter -Wformat -Wformat-security -Wparentheses -Wshadow -Wno-unknown-pragmas -Wno-deprecated")
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(RELEASE_COMPILER_FLAGS "-fexpensive-optimizations -frerun-cse-after-loop -fcse-follow-jumps -finline-functions -fschedule-insns2 -fthread-jumps -fforce-addr -fstrength-reduce -funroll-loops -mfpmath=sse")
ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	SET(RELEASE_COMPILER_FLAGS "-finline-functions -funroll-loops")
ENDIF()
SET(SWIG_CXX_COMPILER_FLAGS "-O0 -g")
SET(CMAKE_C_FLAGS "${COMPILER_WARNINGS} ${CMAKE_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${COMPILER_WARNINGS} ${CMAKE_CXX_FLAGS}")
SET(CMAKE_C_FLAGS_RELEASE "-O3 ${RELEASE_COMPILER_FLAGS}")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 ${RELEASE_COMPILER_FLAGS}")
SET(CMAKE_C_FLAGS_DISTRIBUTION "-O2")
SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O2")
SET(CMAKE_C_FLAGS_DEBUG "-g")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")

OPTION(ENABLE_COVERAGE "Enable code coverage" OFF)
IF(ENABLE_COVERAGE)
	IF(NOT CMAKE_COMPILER_IS_GNUCXX)
		MESSAGE(FATAL_ERROR "Compiler is not GNU gcc! Aborting...")
	ENDIF()
	IF (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
		MESSAGE(WARNING "Code coverage results with an optimised (non-Debug) build may be misleading")
	ENDIF()
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
	SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
	SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} --coverage")
ENDIF()

IF(ENABLE_TSAN)
	IF (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
		MESSAGE(WARNING "Thread-sanitizer results with an optimised (non-Debug) build may be misleading")
	ENDIF()
	SET(SANITIZER_FLAGS "-fsanitize=thread -fPIE")
	SET(SANITIZER_LIBRARY -pie)
	IF(CMAKE_COMPILER_IS_GNUCXX)
		LIST(APPEND SANITIZER_LIBRARY -ltsan)
	ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		LIST(APPEND SANITIZER_LIBRARY -fsanitize=thread)
	ENDIF()
ENDIF()

IF(ENABLE_ASAN)
	IF (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
		MESSAGE(WARNING "Address-sanitizer results with an optimised (non-Debug) build may be misleading")
	ENDIF()
	SET(SANITIZER_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
	SET(SANITIZER_LIBRARY -fsanitize=address)
	IF(CMAKE_COMPILER_IS_GNUCXX)
		LIST(APPEND SANITIZER_LIBRARY -lasan)
	ENDIF()
ENDIF()

IF(ENABLE_MSAN)
	IF (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
		MESSAGE(WARNING "Memory-sanitizer results with an optimised (non-Debug) build may be misleading")
	ENDIF()
	SET(SANITIZER_FLAGS "-fsanitize=memory -fsanitize-memory-track-origins -fno-omit-frame-pointer")
	SET(SANITIZER_LIBRARY)
	IF(CMAKE_COMPILER_IS_GNUCXX)
		LIST(APPEND SANITIZER_LIBRARY -lmsan)
	ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		LIST(APPEND SANITIZER_LIBRARY -fsanitize=memory)
	ENDIF()
ENDIF()

IF(ENABLE_UBSAN)
	IF (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
		MESSAGE(WARNING "Memory-sanitizer results with an optimised (non-Debug) build may be misleading")
	ENDIF()
	SET(SANITIZER_FLAGS "-fsanitize=undefined")
	SET(SANITIZER_LIBRARY)
	IF(CMAKE_COMPILER_IS_GNUCXX)
		LIST(APPEND SANITIZER_LIBRARY -lubsan)
	ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		LIST(APPEND SANITIZER_LIBRARY -fsanitize=undefined)
	ENDIF()
ENDIF()

# check for supported c++11 features
#
# clang with -std=c++11 and -stdlib=libc++ does not work
# well with swig generated cxx hence disable c++11 for this case.

# this has been only fixed in swig 2.0.12 or later.
IF (NOT ((CYGWIN AND ENABLE_TESTING) OR (DARWIN AND COMPILE_MODULAR_INTERFACE
	AND SWIG_VERSION VERSION_LESS "2.0.12") OR MatlabStatic))
	INCLUDE(CheckCXX11Features)

	IF(_HAS_CXX11_FLAG)
		LIST(APPEND DEFINES HAVE_CXX11)
		SET(HAVE_CXX11 1)
		SET(CMAKE_CXX_FLAGS "${CXX11_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
		SET(SWIG_CXX_COMPILER_FLAGS "${CXX11_COMPILER_FLAGS} ${SWIG_CXX_COMPILER_FLAGS}")
	ENDIF()

	IF(_HAS_CXX0X_FLAG)
		LIST(APPEND DEFINES HAVE_CXX0X)
		SET(HAVE_CXX0X 1)
		SET(CMAKE_CXX_FLAGS "${CXX11_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
		SET(SWIG_CXX_COMPILER_FLAGS "${CXX11_COMPILER_FLAGS} ${SWIG_CXX_COMPILER_FLAGS}")
	ENDIF()

	IF (HAVE_CXX11_ATOMIC)
		LIST(APPEND DEFINES HAVE_CXX11_ATOMIC)
	ENDIF()
ELSEIF(DARWIN AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") AND NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0.0"))
	# osx clang 5.0.0 or later uses libc++ by default
	# this is causing problems with source generated by swig version earlier than 3.0.0
	# force to use libstdc++ for compilation of sources
	SET(CMAKE_CXX_FLAGS "-stdlib=libstdc++ ${CMAKE_CXX_FLAGS}")
	SET(SWIG_CXX_COMPILER_FLAGS "-stdlib=libstdc++ ${SWIG_CXX_COMPILER_FLAGS}")
ENDIF()

include(CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX("unordered_map" HAVE_STD_UNORDERED_MAP)
IF (HAVE_STD_UNORDERED_MAP)
	LIST(APPEND DEFINES HAVE_STD_UNORDERED_MAP)
ENDIF()

IF(CMAKE_BUILD_TYPE MATCHES "Release")
	# there's a bug on FreeBSD -march=native.
	# for some CPU types it does not detect the right flags
	# see: http://www.freebsd.org/cgi/query-pr.cgi?pr=178792&cat=
	IF (NOT (FREEBSD AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")))
		SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
		SET(SWIG_CXX_COMPILER_FLAGS "${SWIG_CXX_COMPILER_FLAGS} -march=native")
	ENDIF()
ENDIF()

###### MALLOC LIBRARY ###########
SET(EXTERNAL_MALLOC_CFLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
if (MALLOC_REPLACEMENT MATCHES "Jemalloc")
	include(external/Jemalloc)

	SET(USE_JEMALLOC 1)
	LIST(APPEND DEFINES USE_JEMALLOC)
	LIST(INSERT INCLUDES 0 ${Jemalloc_INCLUDE_DIR})
	SET(CMAKE_CXX_FLAGS "${EXTERNAL_MALLOC_CFLAGS} ${CMAKE_CXX_FLAGS}")
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${Jemalloc_LIBRARIES})
elseif(MALLOC_REPLACEMENT MATCHES "TCMalloc")
	include(external/TCMalloc)

	SET(USE_TCMALLOC 1)
	LIST(APPEND DEFINES USE_TCMALLOC)
	LIST(INSERT INCLUDES 0 ${TCMalloc_INCLUDE_DIR})
	SET(CMAKE_CXX_FLAGS "${EXTERNAL_MALLOC_CFLAGS} ${CMAKE_CXX_FLAGS}")
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${TCMalloc_LIBRARIES})
elseif(MALLOC_REPLACEMENT MATCHES "Hoard")
	find_package(Hoard)
	if (Hoard_FOUND)
		SET(USE_HOARD 1)
		LIST(APPEND DEFINES USE_HOARD)
		SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${Hoard_LIBRARIES})
	else ()
	message(FATAL_ERROR "Unable to use hoard malloc: library not found")
	endif ()
else()
	message(STATUS "Using system's malloc")
endif()

############### VERSION #####################
FILE(STRINGS "@CMAKE_SOURCE_DIR@/NEWS" NEWS LIMIT_COUNT 5)
STRING(REGEX REPLACE ".*SHOGUN Release version ([0-9.]*).*" "\\1" VERSION "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*\\(libshogun ([0-9.]*).*" "\\1" LIBSHOGUNVER "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*\\(libshogun ([0-9]*).*" "\\1" LIBSHOGUNSO "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*data ([0-9.]*).*" "\\1" DATAVER "${NEWS}")
STRING(REGEX REPLACE "([0-9]*).[0-9]*.[0-9]*" "\\1" SHOGUN_VERSION_MAJOR "${VERSION}")
STRING(REGEX REPLACE "[0-9]*.([0-9]*).[0-9]*" "\\1" SHOGUN_VERSION_MINOR "${VERSION}")
STRING(REGEX REPLACE "[0-9]*.[0-9]*.([0-9]*)" "\\1" SHOGUN_VERSION_PATCH "${VERSION}")

SET(BUILD_STATIC "Build a statically linked binary" OFF)
################# EXAMPLES ##################
OPTION(BUILD_EXAMPLES "Build Examples" ON)

################# DATATYPES #################
LIST(APPEND DEFINES USE_CHAR)
LIST(APPEND DEFINES USE_BOOL)
LIST(APPEND DEFINES USE_UINT8)
LIST(APPEND DEFINES USE_UINT16)
#LIST(APPEND DEFINES USE_UINT32)
LIST(APPEND DEFINES USE_UINT64)
#LIST(APPEND DEFINES USE_INT8)
#LIST(APPEND DEFINES USE_INT16)
LIST(APPEND DEFINES USE_INT32)
LIST(APPEND DEFINES USE_INT64)
LIST(APPEND DEFINES USE_FLOAT32)
LIST(APPEND DEFINES USE_FLOAT64)
LIST(APPEND DEFINES USE_COMPLEX128)
#LIST(APPEND DEFINES USE_FLOATMAX)

IF(MSVC)
	include(external/MSIntTypes)
	LIST(APPEND INCLUDES ${THIRD_PARTY_DIR}/MSIntTypes)

	include(external/MSDirent)
	LIST(APPEND INCLUDES ${MSDIRENT_INCLUDE_DIR})
ENDIF()

# detect word size
IF(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT DARWIN)
	LIST(APPEND CMAKE_SWIG_FLAGS "-DSWIGWORDSIZE64")
ENDIF()

#integration
OPTION(OpenCV "OpenCV Integration" OFF)

#interfaces
OPTION(PythonModular "Python Modular" OFF)
OPTION(LuaModular "Lua Modular" OFF)
OPTION(RModular "R Modular" OFF)
OPTION(OctaveModular "Octave Modular" OFF)
OPTION(JavaModular "Java Modular" OFF)
OPTION(PerlModular "Perl Modular" OFF)
OPTION(RubyModular "Ruby Modular" OFF)
OPTION(CSharpModular "CSharp Modular" OFF)

OPTION(PythonStatic "Python Static" OFF)
OPTION(RStatic "R Static" OFF)
OPTION(CmdLineStatic "CmdLine Static" OFF)
OPTION(MatlabStatic "Matlab Static" OFF)
OPTION(OctaveStatic "Octave Static" OFF)

# SVMLight
OPTION(USE_SVMLIGHT "SVMLight" ON)
SET(SVMLightWarning "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
SET(SVMLightWarning "${SVMLightWarning}\nWARNING: SHOGUN is built using SVMlight which was written")
SET(SVMLightWarning "${SVMLightWarning}\nby Thorsten Joachims and uses a different non GPL compatible license.")
SET(SVMLightWarning "${SVMLightWarning}\nTo build a fully GPL'd SHOGUN use")
SET(SVMLightWarning "${SVMLightWarning}\nThe SVMlight license can be found in LICENSE.SVMlight.  In case")
SET(SVMLightWarning "${SVMLightWarning}\nyou do not know or cannot agree to the licensing terms expressed in")
SET(SVMLightWarning "${SVMLightWarning}\nLICENSE.SVMlight press ctrl+c to abort configure now.")
SET(SVMLightWarning "${SVMLightWarning}\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
IF(USE_SVMLIGHT)
	LIST(APPEND DEFINES USE_SVMLIGHT)
ENDIF()

#Trace memory allocs
OPTION(TRACE_MEMORY_ALLOCS "Memory allocation tracing" OFF)
IF(TRACE_MEMORY_ALLOCS)
	LIST(APPEND DEFINES TRACE_MEMORY_ALLOCS)
ENDIF()

# HMM
OPTION(USE_HMMDEBUG "HMM cache" OFF)
IF(USE_HMMDEBUG)
	LIST(APPEND DEFINES USE_HMMDEBUG)
ENDIF()

OPTION(USE_HMMCACHE "HMM cache" ON)
IF(USE_HMMCACHE)
	LIST(APPEND DEFINES USE_HMMCACHE)
ENDIF()

OPTION(USE_HMMPARALLEL "Parallel structures in hmm training" OFF)
IF(USE_HMMPARALLEL)
	SET(USE_HMMPARALLEL_STRUCTURES 1)
	LIST(APPEND DEFINES USE_HMMPARALLEL_STRUCTURES USE_HMMPARALLEL)
ENDIF()

# Viterbi path debug
OPTION(USE_PATHDEBUG "Viterbi path debugging" OFF)
IF(USE_PATHDEBUG)
	LIST(APPEND DEFINES USE_PATHDEBUG)
ENDIF()

# big states
OPTION(USE_BIGSTATES "Big (16bit) state" ON)
IF(USE_BIGSTATES)
	LIST(APPEND DEFINES USE_BIGSTATES)
ENDIF()

# Large file
OPTION(HAVE_LARGEFILE "Large file support" ON)
IF(HAVE_LARGEFILE)
	LIST(APPEND DEFINES HAVE_LARGEFILE)
ENDIF()

#kernelcache to use 4-byte-floating-point values instead of 8-byte-doubles
OPTION(USE_SHORTREAL_KERNELCACHE "Kernelcache to use 4-byte-floating-point values instead of 8-byte-doubles" ON)
IF(USE_SHORTREAL_KERNELCACHE)
	LIST(APPEND DEFINES USE_SHORTREAL_KERNELCACHE)
ENDIF()

# PRNG settings
SET(MEXP "19937" CACHE STRING "PRNG Mersenne exponent")
SET(SFMT_MEXP ${MEXP})
SET(DSFMT_MEXP ${MEXP})
LIST(APPEND DEFINES DSFMT_MEXP=${MEXP} SFMT_MEXP=${MEXP})

# Reference counting
OPTION(USE_REFERENCE_COUNTING "Reference Counting" ON)
IF(USE_REFERENCE_COUNTING)
	LIST(APPEND DEFINES USE_REFERENCE_COUNTING)
ENDIF()

OPTION(USE_LOGCACHE "Use (1+exp(x)) log cache (is much faster but less accurate)" OFF)
IF(USE_LOGCACHE)
	LIST(APPEND DEFINES USE_LOGCACHE)
ENDIF()

OPTION(USE_LOGSUMARRAY "Use sum array, supposed to be a bit more accurate" OFF)
IF(USE_LOGSUMARRAY)
	LIST(APPEND DEFINES USE_LOGSUMARRAY)
ENDIF()

######################### LIBRARIES #########################

#check for symbols
include (CheckCXXSymbolExists)
CHECK_CXX_SYMBOL_EXISTS(isfinite "cmath" HAVE_DECL_ISFINITE)
AppendToDefines(HAVE_DECL_ISFINITE)

CHECK_CXX_SYMBOL_EXISTS(isinf "cmath" HAVE_DECL_ISINF)
AppendToDefines(HAVE_DECL_ISINF)

CHECK_CXX_SYMBOL_EXISTS(isnan "cmath" HAVE_DECL_ISNAN)
AppendToDefines(HAVE_DECL_ISNAN)

CHECK_CXX_SYMBOL_EXISTS(signgam "cmath" HAVE_DECL_SIGNGAM)
AppendToDefines(HAVE_DECL_SIGNGAM)

# check for math functions
include(CheckFunctionExists)
IF(UNIX)
	SET(CMAKE_REQUIRED_LIBRARIES m)
ENDIF()
CHECK_FUNCTION_EXISTS(log2 HAVE_LOG2)
IF(HAVE_LOG2)
	LIST(APPEND DEFINES HAVE_LOG2)
ENDIF()

CHECK_FUNCTION_EXISTS(powl HAVE_POWL)
IF(HAVE_POWL)
	LIST(APPEND DEFINES HAVE_POWL)
ENDIF()

CHECK_FUNCTION_EXISTS(lgammal HAVE_LGAMMAL)
IF(HAVE_LGAMMAL)
	LIST(APPEND DEFINES HAVE_LGAMMAL)
ENDIF()

CHECK_FUNCTION_EXISTS(sqrtl HAVE_SQRTL)
IF(HAVE_SQRTL)
	LIST(APPEND DEFINES HAVE_SQRTL)
ENDIF()

CHECK_FUNCTION_EXISTS(finite HAVE_FPCLASS)
AppendToDefines(HAVE_FPCLASS)

CHECK_FUNCTION_EXISTS(fpclass HAVE_FPCLASS)
AppendToDefines(HAVE_FPCLASS)

CHECK_FUNCTION_EXISTS(isfinite HAVE_ISFINITE)
AppendToDefines(HAVE_ISFINITE)

CHECK_FUNCTION_EXISTS(isinf HAVE_ISINF)
AppendToDefines(HAVE_ISINF)

CHECK_FUNCTION_EXISTS(isnan HAVE_ISNAN)
AppendToDefines(HAVE_ISNAN)

include(CheckCXXSourceCompiles)
CHECK_CXX_SOURCE_COMPILES(
  "#include <cmath>\nint main() { return std::isinf( 0 ); }\n"
  HAVE_STD_ISINF )
AppendToDefines(HAVE_STD_ISINF)

CHECK_CXX_SOURCE_COMPILES(
  "#include <cmath>\nint main() { return std::isfinite( 0 ); }\n"
  HAVE_STD_ISFINITE )
AppendToDefines(HAVE_STD_ISFINITE)

CHECK_CXX_SOURCE_COMPILES(
  "#include <cmath>\nint main() { return std::isnan( 0 ); }\n"
  HAVE_STD_ISNAN )
AppendToDefines(HAVE_STD_ISNAN)

# check SSE and SSE2 intrinsics header
IF((NOT CYGWIN) AND (NOT DISABLE_SSE))
include(CheckIncludeFile)
CHECK_INCLUDE_FILE(xmmintrin.h HAVE_BUILTIN_VECTOR)
CHECK_INCLUDE_FILE(emmintrin.h HAVE_SSE2)
IF(HAVE_SSE2)
	LIST(APPEND DEFINES HAVE_SSE2)
ENDIF(HAVE_SSE2)
ENDIF((NOT CYGWIN) AND (NOT DISABLE_SSE))

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

FIND_PACKAGE(GDB)
IF (GDB_FOUND)
	SET(GDB_DEFAULT_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/src/.gdb)
ENDIF()

FIND_PACKAGE(Mosek)
IF (MOSEK_FOUND)
	SET(USE_MOSEK 1)
	LIST(APPEND DEFINES USE_MOSEK)
	LIST(APPEND INCLUDES ${MOSEK_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${MOSEK_LIBRARY})
ENDIF()

FIND_PACKAGE(Threads)
IF (CMAKE_USE_PTHREADS_INIT)
	SET(HAVE_PTHREAD 1)
	LIST(APPEND DEFINES HAVE_PTHREAD)
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${CMAKE_THREAD_LIBS_INIT})
ENDIF()

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

FIND_PACKAGE(PkgConfig)

# prefer original LAPACK, if needed
OPTION(USE_ORIGINAL_LAPACK "Original LAPACK" OFF)

FIND_PACKAGE(LAPACK)
IF (LAPACK_FOUND)
	# find out whether it is Accelerate.framework we found for LaPack/BLAS
	IF("${LAPACK_LIBRARIES}" MATCHES ".*/Accelerate.framework$")
		SET(HAVE_MVEC 1)
		LIST(APPEND DEFINES HAVE_MVEC)

		#Accelerate framework provides HAVE_CATLAS
		SET(HAVE_CATLAS 1)
		LIST(APPEND DEFINES HAVE_CATLAS)

		SET(HAVE_LAPACK 1)
		LIST(APPEND DEFINES HAVE_LAPACK)
		SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LAPACK_LIBRARIES})
	ELSEIF("${LAPACK_LIBRARIES}" MATCHES ".*/mkl_.*")
		SET(HAVE_LAPACK 1)
		SET(HAVE_MKL 1)
		LIST(APPEND DEFINES HAVE_LAPACK HAVE_MKL)
	ELSE()
		FIND_PACKAGE(Atlas)
		IF(ATLAS_FOUND)
			SET(HAVE_ATLAS 1)
			SET(HAVE_LAPACK 1)
			LIST(APPEND DEFINES HAVE_LAPACK HAVE_ATLAS)
			LIST(APPEND INCLUDES ${ATLAS_INCLUDES})
			IF(USE_ORIGINAL_LAPACK)
				SET(ATLAS_CLAPACK_LIBRARY)
				FOREACH(ITR ${ATLAS_LIBRARIES})
					IF(ITR MATCHES ".*atlas.*lapack.*" OR ITR MATCHES ".*lapack.*atlas.*")
					STRING(REGEX REPLACE "lapack" "clapack" ITR ${ITR})
					LIST(APPEND ATLAS_CLAPACK_LIBRARY ${ITR})
					ENDIF(ITR MATCHES ".*atlas.*lapack.*" OR ITR MATCHES ".*lapack.*atlas.*")
				ENDFOREACH(ITR ${ATLAS_LIBRARIES})
				MESSAGE(STATUS "using ATLAS-CLAPACK from: ${ATLAS_CLAPACK_LIBRARY}")
				SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LAPACK_LIBRARIES} ${ATLAS_CLAPACK_LIBRARY} ${ATLAS_LIBRARIES})
			ELSE(USE_ORIGINAL_LAPACK)
				SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${ATLAS_LIBRARIES})
			ENDIF(USE_ORIGINAL_LAPACK)
		ELSE()
			FIND_PACKAGE(CBLAS)
		    if(CBLAS_LIBRARY)
			SET(HAVE_LAPACK 1)
			LIST(APPEND DEFINES HAVE_LAPACK)
				SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LAPACK_LIBRARIES} ${CBLAS_LIBRARY})
			else()
				SET(HAVE_LAPACK 1)
				LIST(APPEND DEFINES HAVE_LAPACK)
				SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LAPACK_LIBRARIES})
		    endif()
		ENDIF()
	ENDIF()
ENDIF()

FIND_PACKAGE(GLPK)
IF (GLPK_FOUND)
	SET(USE_GLPK 1)
	LIST(APPEND DEFINES USE_GLPK)
	LIST(APPEND INCLUDES ${GLPK_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${GLPK_LIBRARY})
ENDIF()

FIND_PACKAGE(CPLEX)
IF (CPLEX_FOUND)
	SET(USE_CPLEX 1)
	LIST(APPEND DEFINES USE_CPLEX)
	LIST(APPEND INCLUDES ${CPLEX_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${CPLEX_LIBRARY})
ENDIF()

FIND_PACKAGE(ARPACK)
IF (ARPACK_FOUND)
	SET(HAVE_ARPACK 1)
	LIST(APPEND DEFINES HAVE_ARPACK)
	LIST(APPEND INCLUDES ${ARPACK_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${ARPACK_LIB})
ENDIF()

OPTION(BUNDLE_EIGEN "Bundle Eigen" OFF)
IF(BUNDLE_EIGEN)
	include(external/Eigen3)
	SET(HAVE_EIGEN3 1)
	LIST(APPEND DEFINES HAVE_EIGEN3)
	LIST(APPEND SYSTEM_INCLUDES ${EIGEN_INCLUDE_DIR})
ELSE()
	FIND_PACKAGE(Eigen3 ${EIGEN_VERSION_MINIMUM})
	IF (EIGEN3_FOUND)
		SET(HAVE_EIGEN3 1)
		LIST(APPEND DEFINES HAVE_EIGEN3)
		LIST(APPEND SYSTEM_INCLUDES ${EIGEN_INCLUDE_DIR})
	ENDIF()
ENDIF()

# OpenCL/ViennaCL detection
FIND_PACKAGE(ViennaCL ${VIENNACL_VERSION_MINIMUM})
IF (VIENNACL_FOUND)
	SET(HAVE_VIENNACL 1)
	LIST(APPEND DEFINES HAVE_VIENNACL)
	LIST(APPEND DEFINES VIENNACL_WITH_OPENCL)
	LIST(APPEND INCLUDES ${VIENNACL_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${VIENNACL_LIBRARIES})
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENCL_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENCL_CXX_FLAGS}")
	SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OPENCL_EXE_LINKER_FLAGS}")
ENDIF()

IF (OpenCV)
	# find OpenCV library if intended by the user.
	FIND_PACKAGE(OpenCV)
	IF (OpenCV_FOUND)
		SET(HAVE_OPENCV 1)
		LIST(APPEND DEFINES HAVE_OPENCV)
		LIST(APPEND INCLUDES ${OpenCV_INCLUDE_DIRS})
		SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${OpenCV_LIBRARIES})
	ENDIF()
ENDIF()

# COLPACK
OPTION(BUNDLE_COLPACK "Bundle COLPACK" OFF)
IF(BUNDLE_COLPACK)
	include(external/ColPack)
	SET(HAVE_COLPACK 1)
	LIST(APPEND DEFINES HAVE_COLPACK)
	LIST(APPEND INCLUDES ${COLPACK_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${COLPACK_LDFLAGS})
ELSE()
	IF (PKG_CONFIG_FOUND)
		pkg_search_module(COLPACK libColPack>=1.0.9 ColPack>=1.0.9)
		IF (COLPACK_FOUND)
			SET(HAVE_COLPACK 1)
			LIST(APPEND DEFINES HAVE_COLPACK)
			LIST(APPEND INCLUDES ${COLPACK_INCLUDE_DIRS})
			SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${COLPACK_LDFLAGS})
		ENDIF()
	ENDIF()
ENDIF()

OPTION(BUNDLE_NLOPT "Bundle NLOPT" OFF)
IF(BUNDLE_NLOPT)
	include(external/NLopt)
	SET(HAVE_NLOPT 1)
	LIST(APPEND DEFINES HAVE_NLOPT)
	LIST(APPEND INCLUDES ${NLOPT_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${NLOPT_LIBRARIES})
ELSE()
	FIND_PACKAGE(NLopt)
	IF (NLOPT_FOUND)
		SET(HAVE_NLOPT 1)
		LIST(APPEND DEFINES HAVE_NLOPT)
		LIST(APPEND INCLUDES ${NLOPT_INCLUDE_DIR})
		SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${NLOPT_LIBRARIES})
	ENDIF()
ENDIF()

FIND_PACKAGE(LpSolve)
IF (LPSOLVE_FOUND)
	SET(USE_LPSOLVE 1)
	LIST(APPEND DEFINES USE_LPSOLVE YY_NEVER_INTERACTIVE PARSER_LP INVERSE_ACTIVE=INVERSE_LUSOL RoleIsExternalInvEngine)
	LIST(APPEND INCLUDES ${LPSOLVE_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LPSOLVE_LIBRARIES})
ENDIF()

FIND_PACKAGE(ColPack)
IF(COLPACK_FOUND)
	SET(HAVE_COLPACK 1)
	LIST(APPEND DEFINES HAVE_COLPACK)
	LIST(APPEND INCLUDES ${COLPACK_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${COLPACK_LIBRARIES})
ENDIF()

OPTION(BUNDLE_ARPREC "Bundle ARPREC" OFF)
IF(BUNDLE_ARPREC)
	include(external/ARPREC)
	SET(HAVE_ARPREC 1)
	LIST(APPEND DEFINES HAVE_ARPREC)
	LIST(APPEND INCLUDES ${ARPREC_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${ARPREC_LIBRARIES})
ELSE()
	FIND_PACKAGE(ARPREC)
	IF(ARPREC_FOUND)
		SET(HAVE_ARPREC 1)
		LIST(APPEND DEFINES HAVE_ARPREC)
		LIST(APPEND INCLUDES ${ARPREC_INCLUDE_DIR})
		SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${ARPREC_LIBRARIES})
	ENDIF()
ENDIF()

FIND_PACKAGE(Doxygen)
IF(DOXYGEN_FOUND)
	SET(HAVE_DOXYGEN 1)
	LIST(APPEND DEFINES HAVE_DOXYGEN)
ENDIF()

# detect PYTHON
FIND_PACKAGE(PythonInterp REQUIRED)

# JSON
OPTION(BUNDLE_JSON "Bundle JSON" OFF)
IF(BUNDLE_JSON)
	include(external/JSON)
	SET(HAVE_JSON 1)
	LIST(APPEND DEFINES HAVE_JSON)
	LIST(APPEND INCLUDES ${JSON_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${JSON_LDFLAGS})
ELSE()
	IF (PKG_CONFIG_FOUND)
		pkg_search_module(JSON libjson>=0.11 json>=0.11 json-c>=0.11)
		IF (JSON_FOUND)
			SET(HAVE_JSON 1)
			LIST(APPEND DEFINES HAVE_JSON)
			LIST(APPEND INCLUDES ${JSON_INCLUDE_DIRS})
			SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${JSON_LDFLAGS})
		ENDIF()
	ENDIF()
ENDIF()

FIND_PACKAGE(LibXml2)
IF (LIBXML2_FOUND)
	SET(HAVE_XML 1)
	LIST(APPEND DEFINES HAVE_XML)
	LIST(APPEND INCLUDES ${LIBXML2_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LIBXML2_LIBRARIES})
ENDIF()

FIND_PACKAGE(HDF5)
IF (HDF5_FOUND)
	SET(HAVE_HDF5 1)
	LIST(APPEND DEFINES HAVE_HDF5)
	LIST(APPEND INCLUDES ${HDF5_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${HDF5_LIBRARIES})
ENDIF()

FIND_PACKAGE(CURL)
IF (CURL_FOUND)
	SET(HAVE_CURL 1)
	LIST(APPEND DEFINES HAVE_CURL)
	LIST(APPEND INCLUDES ${CURL_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${CURL_LIBRARIES})
ENDIF(CURL_FOUND)

FIND_PACKAGE(ZLIB)
IF (ZLIB_FOUND)
	SET(USE_GZIP 1)
	LIST(APPEND DEFINES USE_GZIP)
	LIST(APPEND INCLUDES ${ZLIB_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${ZLIB_LIBRARIES})
ENDIF()

FIND_PACKAGE(BZip2)
IF (BZIP2_FOUND)
	SET(USE_BZIP2 1)
	LIST(APPEND DEFINES USE_BZIP2)
	LIST(APPEND INCLUDES ${BZIP_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${BZIP2_LIBRARIES})
ENDIF()

FIND_PACKAGE(LibLZMA)
IF (LIBLZMA_FOUND)
	SET(USE_LZMA 1)
	LIST(APPEND DEFINES USE_LZMA)
	LIST(APPEND INCLUDES ${LIBLZMA_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LIBLZMA_LIBRARIES})
ENDIF()

FIND_PACKAGE(SNAPPY)
IF (SNAPPY_FOUND)
	SET(USE_SNAPPY 1)
	LIST(APPEND DEFINES USE_SNAPPY)
	LIST(APPEND INCLUDES ${SNAPPY_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${SNAPPY_LIBRARIES})
ENDIF()

FIND_PACKAGE(LZO)
IF (LZO_FOUND)
	SET(USE_LZO 1)
	LIST(APPEND DEFINES USE_LZO)
	LIST(APPEND INCLUDES ${LZO_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LZO_LIBS})
ENDIF()

FIND_PACKAGE(Spinlock)
IF (SPINLOCK_FOUND)
	SET(USE_SPINLOCKS 1)
	LIST(APPEND DEFINES USE_SPINLOCKS)
ENDIF()

FIND_PACKAGE(Protobuf)
IF (PROTOBUF_FOUND)
	SET(HAVE_PROTOBUF 1)
	LIST(APPEND DEFINES HAVE_PROTOBUF)
	LIST(APPEND INCUDES ${PROTOBUF_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${PROTOBUF_LIBRARIES})
ENDIF()

# Check if supported linear algebra backend is found
IF (HAVE_CXX0X OR HAVE_CXX11)
	IF (EIGEN3_FOUND OR VIENNACL_FOUND)
		SET(HAVE_LINALG_LIB 1)
		LIST(APPEND DEFINES HAVE_LINALG_LIB)
		MESSAGE("-- Supported backend(s) for linear algebra libraries found")
	ELSE ()
		MESSAGE("-- NO supported backend(s) for linear algebra libraries found")
	ENDIF ()
ELSE ()
	MESSAGE ("-- Linear algebra uses c++11 features. Please use a supported compiler")
ENDIF ()

# Linear algebra default global backend setups
OPTION (LinAlgBackend "Set global linear algebra backend library for all modules")
IF (LinAlgBackend STREQUAL "EIGEN3")
	FIND_PACKAGE(Eigen3 ${EIGEN_VERSION_MINIMUM} REQUIRED)
	IF (EIGEN3_FOUND)
		SET(USE_EIGEN3 1)
		LIST(APPEND DEFINES USE_EIGEN3)
		MESSAGE("-- Eigen3 set as default global linear algebra backend library")
	ELSE ()
		MESSAGE("-- Could NOT set Eigen3 as default global linear algebra backend")
	ENDIF ()
ELSEIF (LinAlgBackend STREQUAL "VIENNACL")
	FIND_PACKAGE(ViennaCL ${VIENNACL_VERSION_MINIMUM} REQUIRED)
	IF (VIENNACL_FOUND)
		SET(USE_VIENNACL 1)
		LIST(APPEND DEFINES USE_VIENNACL)
		MESSAGE("-- ViennaCL set as default global linear algebra backend library")
	ELSE ()
		MESSAGE("-- Could NOT set ViennaCL as default global linear algebra backend")
	ENDIF ()
ENDIF ()

# Linear algebra default module specific backend setup

# Core module
OPTION (CoreLib "Set linear algebra backend library for core module")
IF (CoreLib STREQUAL "EIGEN3")
	FIND_PACKAGE(Eigen3 ${EIGEN_VERSION_MINIMUM} REQUIRED)
	IF (EIGEN3_FOUND)
		SET(USE_EIGEN3_CORE 1)
		LIST(APPEND DEFINES USE_EIGEN3_CORE)
		MESSAGE("-- Eigen3 set as default core module (linalg) backend library")
	ELSE ()
		MESSAGE("-- Could NOT set Eigen3 as default core module (linalg) backend")
	ENDIF ()
ELSEIF (CoreLib STREQUAL "VIENNACL")
	FIND_PACKAGE(ViennaCL ${VIENNACL_VERSION_MINIMUM} REQUIRED)
	IF (VIENNACL_FOUND)
		SET(USE_VIENNACL_CORE 1)
		LIST(APPEND DEFINES USE_VIENNACL_CORE)
		MESSAGE("-- ViennaCL set as default core module (linalg) backend library")
	ELSE ()
		MESSAGE("-- Could NOT set ViennaCL as default core module (linalg) backend")
	ENDIF ()
ENDIF ()

# Reduction module
OPTION (ReduxLib "Set linear algebra backend library for reduction module")
IF (ReduxLib STREQUAL "EIGEN3")
	FIND_PACKAGE(Eigen3 ${EIGEN_VERSION_MINIMUM} REQUIRED)
	IF (EIGEN3_FOUND)
		SET(USE_EIGEN3_REDUX 1)
		LIST(APPEND DEFINES USE_EIGEN3_REDUX)
		MESSAGE("-- Eigen3 set as default reduction module (linalg) backend library")
	ELSE ()
		MESSAGE("-- Could NOT set Eigen3 as default reduction module (linalg) backend")
	ENDIF ()
ELSEIF (ReduxLib STREQUAL "VIENNACL")
	FIND_PACKAGE(ViennaCL ${VIENNACL_VERSION_MINIMUM} REQUIRED)
	IF (VIENNACL_FOUND)
		SET(USE_VIENNACL_REDUX 1)
		LIST(APPEND DEFINES USE_VIENNACL_REDUX)
		MESSAGE("-- ViennaCL set as default reduction module (linalg) backend library")
	ELSE ()
		MESSAGE("-- Could NOT set ViennaCL as default reduction module (linalg) backend")
	ENDIF ()
ENDIF ()

# Linear solver module
OPTION (LinSolverLib "Set linear algebra backend library for linear solver module")
IF (LinSolverLib STREQUAL "EIGEN3")
	FIND_PACKAGE(Eigen3 ${EIGEN_VERSION_MINIMUM} REQUIRED)
	IF (EIGEN3_FOUND)
		SET(USE_EIGEN3_LINSLV 1)
		LIST(APPEND DEFINES USE_EIGEN3_LINSLV)
		MESSAGE("-- Eigen3 set as default linear solver module (linalg) backend library")
	ELSE ()
		MESSAGE("-- Could NOT set Eigen3 as default linear solver module (linalg) backend")
	ENDIF ()
ELSEIF (LinSolverLib STREQUAL "VIENNACL")
	FIND_PACKAGE(ViennaCL ${VIENNACL_VERSION_MINIMUM} REQUIRED)
	IF (VIENNACL_FOUND)
		SET(USE_VIENNACL_LINSLV 1)
		LIST(APPEND DEFINES USE_VIENNACL_LINSLV)
		MESSAGE("-- ViennaCL set as default linear solver module (linalg) backend library")
	ELSE ()
		MESSAGE("-- Could NOT set ViennaCL as default linear solver module (linalg) backend")
	ENDIF ()
ENDIF ()

# Eigen solver module
OPTION (EigenSolver "Set linear algebra backend library for eigen solver module")
IF (EigenSolver STREQUAL "EIGEN3")
	FIND_PACKAGE(Eigen3 ${EIGEN_VERSION_MINIMUM} REQUIRED)
	IF (EIGEN3_FOUND)
		SET(USE_EIGEN3_EIGSLV 1)
		LIST(APPEND DEFINES USE_EIGEN3_EIGSLV)
		MESSAGE("-- Eigen3 set as default eigen solver module (linalg) backend library")
	ELSE ()
		MESSAGE("-- Could NOT set Eigen3 as default eigen solver module (linalg) backend")
	ENDIF ()
ELSEIF (EigenSolver STREQUAL "VIENNACL")
	FIND_PACKAGE(ViennaCL ${VIENNACL_VERSION_MINIMUM} REQUIRED)
	IF (VIENNACL_FOUND)
		SET(USE_VIENNACL_EIGSLV 1)
		LIST(APPEND DEFINES USE_VIENNACL_EIGSLV)
		MESSAGE("-- ViennaCL set as default eigen solver module (linalg) backend library")
	ELSE ()
		MESSAGE("-- Could NOT set ViennaCL as default eigen solver module (linalg) backend")
	ENDIF ()
ENDIF ()

#SWIG Interfaces
LIST(APPEND CMAKE_SWIG_FLAGS "-w473")
LIST(APPEND CMAKE_SWIG_FLAGS "-w454")
LIST(APPEND CMAKE_SWIG_FLAGS "-w312")
LIST(APPEND CMAKE_SWIG_FLAGS "-w325")
LIST(APPEND CMAKE_SWIG_FLAGS "-fvirtual")

OPTION(USE_SWIG_DIRECTORS "Enable SWIG director classes" OFF)
IF(USE_SWIG_DIRECTORS)
	LIST(APPEND DEFINES USE_SWIG_DIRECTORS)
ENDIF()

# python modular
IF (PythonModular OR PythonStatic)
	# find python library for the found python interpreter
	# i.e. versions should match!
	FIND_PACKAGE(PythonLibs REQUIRED)
	FIND_PACKAGE(NumPy REQUIRED)

	LIST(APPEND DEFINES HAVE_PYTHON)

	IF(PythonModular)
		#custom swig flags for python modular interface
		IF(${PYTHON_VERSION_MAJOR} VERSION_EQUAL 3)
			SET(TARGET_SWIGFLAGS "-builtin\;-modern\;-modernargs\;-py3")
			LIST(APPEND DEFINES PYTHON3)
		ELSE()
			SET(TARGET_SWIGFLAGS "-builtin\;-modern\;-modernargs")
		ENDIF()

		IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/python_modular)
			add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/python_modular)
		ENDIF()
	ENDIF()

	IF(PythonStatic)
		IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/python_static)
			add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/python_static)
		ENDIF()
	ENDIF()
ENDIF()

# lua modular
IF (LuaModular)
	#FIND_PACKAGE(Lua51 REQUIRED)
	pkg_search_module(LUA REQUIRED lua lua5.1)
	IF (NOT LUA_FOUND)
	MESSAGE(FATAL_ERROR "Lua could not be found!\n"
		"Lua is required for compiling lua modular interface!")
    ENDIF ()
	SET(LUA_EXECUTABLE lua)
	UNSET(TARGET_SWIGFLAGS)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/lua_modular)
		add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/lua_modular)
	ENDIF()
ENDIF()

# java modular
IF (JavaModular)
	FIND_PACKAGE(Java REQUIRED)
	FIND_PACKAGE(JNI REQUIRED)

	#find jblas
	INCLUDE(UseJava)
	FIND_JAR(JBLAS NAMES jblas jblas-1.2.0
			PATHS /usr/lib/java /opt/local/share/java
				  ENV JBLAS_PATH)
	IF(JBLAS)
		LIST(APPEND DEFINES HAVE_JBLAS)
		SET(HAVE_JBLAS 1)
	ELSE()
		MESSAGE(FATAL_ERROR "JBlas could not be found!\n"
			"It is required for java modular interface!!!")
	ENDIF()

	SET(TARGET_SWIGFLAGS "-package\;org.shogun")
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/java_modular)
		add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/java_modular)
	ENDIF()
ENDIF()

# ruby modular
IF (RubyModular)
	FIND_PACKAGE(Ruby REQUIRED)
	FIND_PACKAGE(RubyNArray REQUIRED)
	SET(NARRAY_LIB ${RUBY_NARRAY_LIBRARY})
	UNSET(TARGET_SWIGFLAGS)
	add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/ruby_modular)
ENDIF()

# octave modular
IF (OctaveModular OR OctaveStatic)
	FIND_PACKAGE(Octave 3.6 REQUIRED)
	LIST(APPEND DEFINES HAVE_OCTAVE)
	LIST(APPEND DEFINES "OCTAVE_APIVERSION=${OCTAVE_API_VERSION}")
	SET(OCTAVE_APIVERSION ${OCTAVE_API_VERSION})

	IF(OctaveModular)
		UNSET(TARGET_SWIGFLAGS)
		IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/octave_modular)
			add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/octave_modular)
		ENDIF()
	ENDIF()

	IF(OctaveStatic)
		IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/octave_static)
			add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/octave_static)
		ENDIF()
	ENDIF()
ENDIF()

# csharp modular
IF (CSharpModular)
	FIND_PACKAGE(CSharp REQUIRED)
	UNSET(TARGET_SWIGFLAGS)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/csharp_modular)
		add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/csharp_modular)
	ENDIF()
ENDIF()

# r modular
IF (RModular OR RStatic)
	FIND_PACKAGE(R REQUIRED)

	IF (RModular)
		UNSET(TARGET_SWIGFLAGS)
		IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/r_modular)
			add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/r_modular)
		ENDIF()
	ENDIF()

	IF (RStatic)
		IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/r_static)
			add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/r_static)
		ENDIF()
	ENDIF()
ENDIF()

# perl modular
IF (PerlModular)
	FIND_PACKAGE(FindPerlLibs REQUIRED)
	UNSET(TARGET_SWIGFLAGS)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/perl_modular)
		#add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/perl_modular)
	ENDIF()
ENDIF()

IF (MatlabStatic)
	FIND_PACKAGE(Matlab REQUIRED)
	IF(MATLAB_FOUND)
		IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/matlab_static)
			add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/matlab_static)
		ENDIF()
	ELSE()
		MESSAGE(FATAL_ERROR "Could not find Matlab, which is required for compiling matlab_static interface. Try setting MATLAB_ROOT enviroment variable to the right path")
	ENDIF()
ENDIF()

IF (CmdLineStatic)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/cmdline_static)
		add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/cmdline_static)
	ENDIF()
ENDIF()

IF (SVMLight)
	MESSAGE(STATUS ${SVMLightWarning})
ENDIF()

################# TESTING ###################
#TODO: finish!!!
OPTION(BUILD_DASHBOARD_REPORTS "Set to ON to activate reporting of Shogun builds" OFF)
IF(BUILD_DASHBOARD_REPORTS)
	file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/configs/valgrind.supp" VALGRIND_SUPPRESSION_FILE)
	SET(MEMORYCHECK_SUPPRESSIONS_FILE ${VALGRIND_SUPPRESSION_FILE} CACHE FILEPATH "File that contains suppressions for the memory checker")
	SET(MEMORYCHECK_COMMAND_OPTIONS "-q --tool=memcheck --leak-check=full --track-origins=yes --num-callers=50 --error-exitcode=1")
	include(CTest)
ENDIF()

OPTION(ENABLE_TESTING "Enable testing" OFF)
OPTION(TRAVIS_DISABLE_UNIT_TESTS "Disable unit testing to speed up jobs on travis-ci" OFF)
OPTION(TRAVIS_DISABLE_LIBSHOGUN_TESTS "Disable libshogun tests to speed up jobs on travis-ci" OFF)
IF(ENABLE_TESTING)
	FIND_PACKAGE(Jinja2)
	IF(NOT BUILD_DASHBOARD_REPORTS)
		enable_testing()
	ENDIF()

	IF(EXISTS ${CMAKE_SOURCE_DIR}/tests)
		# add integration tests
		add_subdirectory(${CMAKE_SOURCE_DIR}/tests/integration)


		# add unit tests
		IF (NOT TRAVIS_DISABLE_UNIT_TESTS)
			add_subdirectory(${CMAKE_SOURCE_DIR}/tests/unit)
		ENDIF()
	ENDIF()
ENDIF()

# save configuration options
MergeCFLAGS()
SET(CONFIGURE_OPTIONS "TODO")
SET(COMPFLAGS_CPP "${MERGED_CXX_FLAGS}")
SET(LINKFLAGS "${POSTLINKFLAGS}")

IF(EXISTS ${CMAKE_SOURCE_DIR}/src/shogun)
	add_subdirectory(${CMAKE_SOURCE_DIR}/src/shogun)
ENDIF()

IF(EXISTS ${CMAKE_SOURCE_DIR}/examples)
	IF(BUILD_EXAMPLES OR ENABLE_TESTING)
		add_subdirectory(${CMAKE_SOURCE_DIR}/examples)
	ENDIF()
ENDIF()

IF(DOXYGEN_FOUND)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/doc)
		add_subdirectory(${CMAKE_SOURCE_DIR}/doc)
	ENDIF()
ENDIF()

include(ShogunPackaging)

message(STATUS "===================================================================================================================")
message(STATUS "Summary of Configuration Variables")
include(FeatureSummary)
feature_summary(WHAT ALL)

message(STATUS "===================================================================================================================")
message(STATUS "Enabled Integration")

PrintInterfaceStatus("OpenCV Integration" OpenCV)

message(STATUS
"===================================================================================================================")
message(STATUS "Enabled Interfaces")

message(STATUS "  libshogun is ON")

PrintInterfaceStatus("python modular" PythonModular)
PrintInterfaceStatus("octave modular" OctaveModular)
PrintInterfaceStatus("java modular" JavaModular)
PrintInterfaceStatus("perl modular" PerlModular)
PrintInterfaceStatus("ruby modular" RubyModular)
PrintInterfaceStatus("csharp modular" CSharpModular)
PrintInterfaceStatus("R modular" RModular)
PrintInterfaceStatus("lua modular" LuaModular)

message(STATUS "")
message(STATUS "Enabled legacy interfaces")
PrintInterfaceStatus("cmdline static" CmdLineStatic)
PrintInterfaceStatus("python static" PythonStatic)
PrintInterfaceStatus("octave static" OctaveStatic)
PrintInterfaceStatus("matlab static" MatlabStatic)
PrintInterfaceStatus("R static" RStatic)

message(STATUS "===================================================================================================================")

message(STATUS "To compile shogun type")
message(STATUS "  make")
message(STATUS "")
message(STATUS "To install shogun to ${CMAKE_INSTALL_PREFIX} type")
message(STATUS "  make install")
message(STATUS "")
message(STATUS "or to install to a custom directory")
message(STATUS "  make install DESTDIR=/my/special/path")
message(STATUS "  (or rerun cmake with -DCMAKE_INSTALL_PREFIX=/my/special/path) to just change the prefix")
message(STATUS "===================================================================================================================")


