# Instructions:
# $ mkdir build
# $ cd build
# $ cmake ..
# $ make

### Policies need to be set *BEFORE* calling 'project()' !!!
# Needed for hardening-flags in Fedora.  This policy will pass
# all compiler-flags to stuff like 'FIND_PACKAGE()'.  Introduced
# with CMake 3.2.0.  See documentation:
# http://www.cmake.org/cmake/help/v3.3/policy/CMP0056.html
IF(POLICY CMP0056)
	CMAKE_POLICY(SET CMP0056 NEW)
	SET(CMAKE_POLICY_DEFAULT_CMP0056 NEW)
ENDIF(POLICY CMP0056)

IF(POLICY CMP0042)
	CMAKE_POLICY(SET CMP0042 NEW)
	SET(CMAKE_POLICY_DEFAULT_CMP0042 NEW)
ENDIF(POLICY CMP0042)

IF(POLICY CMP0054)
	CMAKE_POLICY(SET CMP0054 NEW)
	SET(CMAKE_POLICY_DEFAULT_CMP0054 NEW)
ENDIF(POLICY CMP0054)

project(shogun)
cmake_minimum_required(VERSION 3.1)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(ShogunUtils)

SET(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

#### set required C++ standard level of the compiler
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (MSVC)
	if (MSVC_VERSION VERSION_LESS 1900)
		message(FATAL_ERROR "C++11 is required to use Shogun, but the version of Visual Studio you are using is too old and doesn't support C++11. You need Visual Studio 2015 or newer. ")
	else()
		include(CheckCXXCompilerFlag)
		CHECK_CXX_COMPILER_FLAG("/std:c++14" _cpp_latest_flag_supported)
		if (_cpp_latest_flag_supported)
			add_compile_options("/std:c++14")
		endif()
	endif()
endif()

### FIXME: remove these flags when the codebase
# is cleared up
set(HAVE_CXX11 ON)

############# minimum library versions ###################
SET(EIGEN_VERSION_MINIMUM 3.1.2)
SET(VIENNACL_VERSION_MINIMUM 1.5.0)

# Store system's or distribution's C[XX]FLAGS.
SET(SYSTEM_C_FLAGS "${CMAKE_C_FLAGS}")
SET(SYSTEM_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
STRING(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UC)
IF(NOT ("${BUILD_TYPE_UC}" STREQUAL "DISTRIBUTION"))
	SET(CMAKE_C_FLAGS "")
	SET(CMAKE_CXX_FLAGS "")
ENDIF(NOT ("${BUILD_TYPE_UC}" STREQUAL "DISTRIBUTION"))

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
SET(COMMON_INTERFACE_SRC_DIR ${CMAKE_SOURCE_DIR}/src/interfaces/swig/)

SET(AVAILABLE_INTERFACES
    INTERFACE_PYTHON;INTERFACE_OCTAVE;INTERFACE_JAVA;INTERFACE_PERL;INTERFACE_RUBY;INTERFACE_CSHARP;INTERFACE_R;INTERFACE_LUA;INTERFACE_SCALA;INTERFACE_GO)
SET(INTERFACE_PYTHON_DESCRIPTION "Python")
SET(INTERFACE_OCTAVE_DESCRIPTION "Octave")
SET(INTERFACE_JAVA_DESCRIPTION "Java")
SET(INTERFACE_PERL_DESCRIPTION "Perl")
SET(INTERFACE_RUBY_DESCRIPTION "Ruby")
SET(INTERFACE_CSHARP_DESCRIPTION "C#")
SET(INTERFACE_R_DESCRIPTION "R")
SET(INTERFACE_LUA_DESCRIPTION "Lua")
SET(INTERFACE_SCALA_DESCRIPTION "Scala")
SET(INTERFACE_GO_DESCRIPTION "Go")
SET(LIBSHOGUN ON CACHE BOOL "Compile shogun library")

IsAnyTrue("${AVAILABLE_INTERFACES}" ANY_INTERFACE_ENABLED)
IF (${ANY_INTERFACE_ENABLED})
	# SWIG3 is the minimum requirement because of C++11 support
	SET(SWIG_VERSION_MINIMUM 3.0.0)

	IF(INTERFACE_CSHARP)
		# We require SWIG 3.0.7 to support functions with a few SGVector or
		# SGMatrix arguments. The required SWIG feature is called
		# "Support for special variable expansion in typemap attributes."
		# and was introduced just in 3.0.7.
		SET(SWIG_VERSION_MINIMUM 3.0.7)

		# SWIG >= 3.0.0 has some new handling with C# (Mono) and breaks
		# typemapping created for earlier versions of SWIG.
		# see: http://www.swig.org/Doc3.0/CSharp.html#CSharp_introduction_swig2_compatibility
		LIST(APPEND CMAKE_SWIG_FLAGS "-DSWIG2_CSHARP")
	ELSEIF(INTERFACE_PYTHON)
		# SWIG was broken for combining -builtin and -modernargs
		# from v3.0.0 and until 3.0.4.  This bug was fixed in
		# v3.0.5.  Make CMake emit an error and fail to configure.
		SET(SWIG_VERSION_MINIMUM 3.0.5)
	ENDIF()

	FIND_PACKAGE(SWIG ${SWIG_VERSION_MINIMUM} REQUIRED)
	# use our own UseSWIG.cmake in order to be able to enable ccache-swig
	SET(SWIG_USE_FILE ${CMAKE_SOURCE_DIR}/cmake/UseSWIG.cmake)
	IF(ENABLE_CCACHE AND CCACHE_SWIG)
		SET(CCACHE_SWIG_EXECUTABLE ${CCACHE_SWIG})
	ENDIF()
	SET(COMPILE_INTERFACE 1)
ENDIF()

# Detect OS
DetectSystemName()

# Get processor type, sets MACHINE macro
SET(MACHINE ${CMAKE_SYSTEM_PROCESSOR})

SET(EXT_LIB_SWIG_RUBY ".so")
if(DARWIN)
	SET(EXT_LIB_SWIG_RUBY ".bundle")
ENDIF()

################ COMPILER #######################
# g++ version needs to be => 4.3
IF(CMAKE_COMPILER_IS_GNUCXX)
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
IF(NOT MSVC)
	SET(COMPILER_WARNINGS "-Wall -Wno-unused-parameter -Wformat -Wformat-security -Wparentheses -Wshadow -Wno-unknown-pragmas -Wno-deprecated")
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(RELEASE_COMPILER_FLAGS "-fexpensive-optimizations -frerun-cse-after-loop -fcse-follow-jumps -finline-functions -fschedule-insns2 -fthread-jumps -fforce-addr -fstrength-reduce -funroll-loops")
	IF (${MACHINE} MATCHES "x86_64" OR ${MACHINE} MATCHES "i686")
		SET(RELEASE_COMPILER_FLAGS "${RELEASE_COMPILER_FLAGS} -mfpmath=sse")
	ENDIF()
ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
	SET(RELEASE_COMPILER_FLAGS "-funroll-loops")
ENDIF()
SET(SWIG_CXX_COMPILER_FLAGS "-O0 -g")
SET(CMAKE_C_FLAGS "${COMPILER_WARNINGS} ${CMAKE_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${COMPILER_WARNINGS} ${CMAKE_CXX_FLAGS}")
IF(MSVC)
	SET(CMAKE_C_FLAGS_RELEASE "/O2 ${RELEASE_COMPILER_FLAGS}")
	SET(CMAKE_CXX_FLAGS_RELEASE "/O2 ${RELEASE_COMPILER_FLAGS}")
	SET(CMAKE_C_FLAGS_DISTRIBUTION "/Ot")
	SET(CMAKE_CXX_FLAGS_DISTRIBUTION "/Ot")
	SET(CMAKE_C_FLAGS_DEBUG "/DEBUG /Od /Zi")
	SET(CMAKE_CXX_FLAGS_DEBUG "/DEBUG /Od /Zi")
	add_compile_options("/bigobj")
ELSE()
	SET(CMAKE_C_FLAGS_RELEASE "-O3 ${RELEASE_COMPILER_FLAGS}")
	SET(CMAKE_CXX_FLAGS_RELEASE "-O3 ${RELEASE_COMPILER_FLAGS}")
	SET(CMAKE_C_FLAGS_DISTRIBUTION "-O2")
	SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O2")
	SET(CMAKE_C_FLAGS_DEBUG "-g")
	SET(CMAKE_CXX_FLAGS_DEBUG "-g")
ENDIF()

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

# Fix build on Mac OSX 10.10 Yosemite when using mp-gcc-4X.
# See: https://github.com/shogun-toolbox/shogun/issues/2635
IF(DARWIN AND (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
	SET(CMAKE_CXX_FLAGS "-mno-avx -flax-vector-conversions -D'__has_extension(x)=0' -DvImage_Utilities_h -DvImage_CVUtilities_h ${CMAKE_CXX_FLAGS}")
	SET(SWIG_CXX_COMPILER_FLAGS "-mno-avx -flax-vector-conversions -D'__has_extension(x)=0' -DvImage_Utilities_h -DvImage_CVUtilities_h ${SWIG_CXX_COMPILER_FLAGS}")
ENDIF(DARWIN AND (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))

###### MALLOC LIBRARY ###########
SET(EXTERNAL_MALLOC_CFLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
if (MALLOC_REPLACEMENT MATCHES "Jemalloc")
	include(external/Jemalloc)

	SET(USE_JEMALLOC 1)
	LIST(INSERT INCLUDES 0 ${Jemalloc_INCLUDE_DIR})
	SET(CMAKE_CXX_FLAGS "${EXTERNAL_MALLOC_CFLAGS} ${CMAKE_CXX_FLAGS}")
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${Jemalloc_LIBRARIES})
elseif(MALLOC_REPLACEMENT MATCHES "TCMalloc")
	include(external/TCMalloc)

	SET(USE_TCMALLOC 1)
	LIST(INSERT INCLUDES 0 ${TCMalloc_INCLUDE_DIR})
	SET(CMAKE_CXX_FLAGS "${EXTERNAL_MALLOC_CFLAGS} ${CMAKE_CXX_FLAGS}")
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${TCMalloc_LIBRARIES})
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
endif()

############### VERSION #####################
FILE(STRINGS "${CMAKE_SOURCE_DIR}/NEWS" NEWS LIMIT_COUNT 5)
STRING(REGEX REPLACE ".*SHOGUN Release version ([0-9.]*).*" "\\1" VERSION "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*\\(libshogun ([0-9.]*).*" "\\1" LIBSHOGUNVER "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*\\(libshogun ([0-9]*).*" "\\1" LIBSHOGUNSO "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*data ([0-9.]*).*" "\\1" DATAVER "${NEWS}")
STRING(REGEX REPLACE "([0-9]*).[0-9]*.[0-9]*" "\\1" SHOGUN_VERSION_MAJOR "${VERSION}")
STRING(REGEX REPLACE "[0-9]*.([0-9]*).[0-9]*" "\\1" SHOGUN_VERSION_MINOR "${VERSION}")
STRING(REGEX REPLACE "[0-9]*.[0-9]*.([0-9]*)" "\\1" SHOGUN_VERSION_PATCH "${VERSION}")

################# EXAMPLES ##################
OPTION(BUILD_EXAMPLES "Build Examples" ON)
OPTION(BUILD_META_EXAMPLES "Generate API examples from meta-examples" ON)
# note the examples dir is added below after tests have been defined

################# DATATYPES #################
IF(COMPILE_INTERFACE)
	OPTION(USE_CHAR "Support for char datatype" ON)
	OPTION(USE_BOOL "Support for bool datatype" ON)
	OPTION(USE_UINT8 "Support for uint8_t datatype" ON)
	OPTION(USE_UINT16 "Support for uint16_t datatype" ON)
	OPTION(USE_UINT32 "Support for uint32_t datatype" OFF)
	OPTION(USE_UINT64 "Support for uint64_t datatype" ON)
	OPTION(USE_INT8 "Support for int8_t datatype" OFF)
	OPTION(USE_INT16 "Support for int16_t datatype" OFF)
	OPTION(USE_INT32 "Support for int32_t datatype" ON)
	OPTION(USE_INT64 "Support for int64_t datatype" ON)
	OPTION(USE_FLOAT32 "Support for float32_t datatype" ON)
	OPTION(USE_FLOAT64 "Support for float64_t datatype" ON)
	OPTION(USE_COMPLEX128 "Support for complex128_t datatype" ON)
	OPTION(USE_FLOATMAX "Support for floatmax_t datatype" OFF)
ENDIF(COMPILE_INTERFACE)

# detect word size
IF(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT (DARWIN OR WIN32))
	SET(SWIGWORDSIZE64 TRUE)
ENDIF()

#interfaces
ForEach(Interface ${AVAILABLE_INTERFACES})
    OPTION(${Interface} ${Interface}Description OFF)
EndForEach(Interface)

# Debugging Python-interface with CTest
OPTION(ENABLE_PYTHON_DEBUG "Enable Python-interface-debugging with CTest" OFF)

SET(SVMLightWarning "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
SET(SVMLightWarning "${SVMLightWarning}\nWARNING: SHOGUN is built using SVMlight which was written")
SET(SVMLightWarning "${SVMLightWarning}\nby Thorsten Joachims and uses a different non GPL compatible license.")
SET(SVMLightWarning "${SVMLightWarning}\nTo build a fully GPL'd SHOGUN use")
SET(SVMLightWarning "${SVMLightWarning}\nThe SVMlight license can be found in LICENSE.SVMlight.  In case")
SET(SVMLightWarning "${SVMLightWarning}\nyou do not know or cannot agree to the licensing terms expressed in")
SET(SVMLightWarning "${SVMLightWarning}\nLICENSE.SVMlight press ctrl+c to abort configure now.")
SET(SVMLightWarning "${SVMLightWarning}\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

#Trace memory allocs
OPTION(TRACE_MEMORY_ALLOCS "Memory allocation tracing" OFF)

# HMM
OPTION(USE_HMMDEBUG "HMM cache" OFF)

OPTION(USE_HMMCACHE "HMM cache" ON)

OPTION(USE_HMMPARALLEL "Parallel structures in hmm training" OFF)
IF(USE_HMMPARALLEL)
	SET(USE_HMMPARALLEL_STRUCTURES 1)
ENDIF()

# Viterbi path debug
OPTION(USE_PATHDEBUG "Viterbi path debugging" OFF)

# big states
OPTION(USE_BIGSTATES "Big (16bit) state" ON)

#kernelcache to use 4-byte-floating-point values instead of 8-byte-doubles
OPTION(USE_SHORTREAL_KERNELCACHE "Kernelcache to use 4-byte-floating-point values instead of 8-byte-doubles" ON)

# PRNG settings
SET(MEXP "19937" CACHE STRING "PRNG Mersenne exponent")
SET(SFMT_MEXP ${MEXP})
SET(DSFMT_MEXP ${MEXP})

OPTION(USE_LOGCACHE "Use (1+exp(x)) log cache (is much faster but less accurate)" OFF)

OPTION(USE_LOGSUMARRAY "Use sum array, supposed to be a bit more accurate" OFF)

######################### LIBRARIES #########################

FIND_PACKAGE(GDB)
IF (GDB_FOUND)
	SET(GDB_DEFAULT_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/src/.gdb)
ENDIF()

FIND_PACKAGE(Doxygen 1.8.6)
IF(DOXYGEN_FOUND)
	SET(HAVE_DOXYGEN 1)
ENDIF()

# detect PYTHON
FIND_PACKAGE(PythonInterp REQUIRED)

# save configuration options
IF (LIBSHOGUN)
	MergeCFLAGS()
	add_subdirectory(${CMAKE_SOURCE_DIR}/src/shogun)
	add_library(shogun::shogun ALIAS shogun)
	if(LIBSHOGUN_BUILD_STATIC)
		add_library(shogun::shogun-static ALIAS shogun-static)
	endif()
	set(shogun_INCLUDE_DIR ${CMAKE_BINARY_DIR}/src)
ELSE()
	find_package(Shogun ${VERSION} CONFIG REQUIRED)
ENDIF()

#SWIG Interfaces
ForEach(SwigFlag "-w473" "-w454" "-w312" "-w325" "-fvirtual")
    LIST(APPEND CMAKE_SWIG_FLAGS ${SwigFlag})
EndForEach()

OPTION(SWIG_SINGLE_THREADED "Build interfaces single-threaded to reduce memory usage" OFF)

OPTION(USE_SWIG_DIRECTORS "Enable SWIG director classes" ON)

# Respect system's or distribution's C[XX]FLAGS.
OPTION(SWIG_WITH_SYSTEM_CFLAGS "Enable system's C[XX]FLAGS for compilation of swig-binaries" ON)

IF(NOT ("${BUILD_TYPE_UC}" STREQUAL "DISTRIBUTION"))
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SYSTEM_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYSTEM_CXX_FLAGS}")
ENDIF(NOT ("${BUILD_TYPE_UC}" STREQUAL "DISTRIBUTION"))

IF(SWIG_WITH_SYSTEM_CFLAGS)
	SET(SWIG_CXX_COMPILER_FLAGS "${SWIG_CXX_COMPILER_FLAGS} ${SYSTEM_CXX_FLAGS}")
ENDIF(SWIG_WITH_SYSTEM_CFLAGS)

OPTION(REDUCE_SWIG_DEBUG "Reduce debuginfo when compiling interfaces" OFF)
IF(REDUCE_SWIG_DEBUG)
	SET(SWIG_CXX_COMPILER_FLAGS "${SWIG_CXX_COMPILER_FLAGS} -g1")
ENDIF(REDUCE_SWIG_DEBUG)

# python
IF (INTERFACE_PYTHON)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/python)
		add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/python)
	ENDIF()
ENDIF()

# lua
IF (INTERFACE_LUA)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/lua)
		add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/lua)
	ENDIF()
ENDIF()

# scala
IF (INTERFACE_SCALA)
	# Java needed because Scala extends Java Classes and uses the executable generated from Java example for Integration testing
	set(INTERFACE_JAVA "ON")
	FIND_PACKAGE(Scala REQUIRED)
	IF(NOT INTERFACE_JAVA)
		IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/java)
			add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/java)
		ENDIF()
	ENDIF()
ENDIF()

# java
IF (INTERFACE_JAVA)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/java)
		add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/java)
	ENDIF()
ENDIF()

# ruby
IF (INTERFACE_RUBY)
	add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/ruby)
ENDIF()

# octave
IF (INTERFACE_OCTAVE)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/octave)
		add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/octave)
	ENDIF()
ENDIF()

# csharp
IF (INTERFACE_CSHARP)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/csharp)
		add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/csharp)
	ENDIF()
ENDIF()

# R
IF (INTERFACE_R)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/r)
		add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/r)
	ENDIF()
ENDIF()

# perl
IF (INTERFACE_PERL)
	FIND_PACKAGE(FindPerlLibs REQUIRED)
	UNSET(TARGET_SWIGFLAGS)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/perl)
		#add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/perl)
	ENDIF()
ENDIF()

IF (INTERFACE_GO)
	IF(EXISTS ${CMAKE_SOURCE_DIR}/src/interfaces/go)
		add_subdirectory(${CMAKE_SOURCE_DIR}/src/interfaces/go)
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
OPTION(TRAVIS_DISABLE_META_CPP "Disable cpp meta examples and integration testing to speed up jobs on travis-ci" OFF)
OPTION(DISABLE_META_INTEGRATION_TESTS "Disable meta integration testing to speed up build" OFF)


IF(ENABLE_TESTING)
	IF (NOT LIBSHOGUN)
		MESSAGE(FATAL_ERROR "Cannot compile tests without libshogun!")
	ENDIF()

	IF(EXISTS ${CMAKE_SOURCE_DIR}/tests)
		IF(NOT BUILD_DASHBOARD_REPORTS)
			enable_testing()
		ENDIF()

		IF (NOT TRAVIS_DISABLE_UNIT_TESTS AND EXISTS ${CMAKE_SOURCE_DIR}/tests/unit)
			add_subdirectory(${CMAKE_SOURCE_DIR}/tests/unit)
		ENDIF()

		IF(BUILD_META_EXAMPLES AND NOT DISABLE_META_INTEGRATION_TESTS AND EXISTS ${CMAKE_SOURCE_DIR}/tests/meta)
            add_subdirectory(${CMAKE_SOURCE_DIR}/tests/meta)
        ENDIF()
	ENDIF()
ENDIF()

IF(EXISTS ${CMAKE_SOURCE_DIR}/examples)
	IF(ENABLE_TESTING AND NOT BUILD_EXAMPLES)
	    message(STATUS "Tests require (disabled) examples, enabling.")
	ENDIF()
    IF(ENABLE_TESTING OR BUILD_EXAMPLES)
	    add_subdirectory(${CMAKE_SOURCE_DIR}/examples)
	ENDIF()

    # always build minimal example(s)
    IF(EXISTS ${CMAKE_SOURCE_DIR}/examples/minimal)
        add_subdirectory(${CMAKE_SOURCE_DIR}/examples/minimal)
    ENDIF()

	IF(BUILD_META_EXAMPLES)
        # allow meta examples without adding examples dir itself
        add_subdirectory(${CMAKE_SOURCE_DIR}/examples/meta)
    ENDIF()
ENDIF()

IF(EXISTS ${CMAKE_SOURCE_DIR}/doc)
	add_subdirectory(${CMAKE_SOURCE_DIR}/doc)
ENDIF()

include(ShogunPackaging)

PrintLine()
PrintStatus("Summary of Configuration Variables")
include(FeatureSummary)
feature_summary(WHAT ALL)

PrintLine()
PrintStatus("Integrations")

PrintInterfaceStatus("OpenCV Integration" OpenCV)

PrintLine()
PrintStatus("Interfaces")

ForEach(Interface ${AVAILABLE_INTERFACES})
    PrintInterfaceStatus("${${Interface}_DESCRIPTION}" ${Interface})
EndForEach(Interface)

PrintLine()
PrintStatus("To compile shogun type")
PrintStatus("  make")
PrintStatus("")
PrintStatus("To install shogun to ${CMAKE_INSTALL_PREFIX} type")
PrintStatus("  make install")
PrintStatus("")
PrintStatus("or to install to a custom directory")
PrintStatus("  make install DESTDIR=/my/special/path")
PrintStatus("  (or rerun cmake with -DCMAKE_INSTALL_PREFIX=/my/special/path) to just change the prefix")
PrintLine()
