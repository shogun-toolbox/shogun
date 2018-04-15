include(CMakeParseArguments)

Macro(IsAnyTrue LIST RESULT)
	Set(${RESULT} "FALSE")
	ForEach(Element ${LIST})
		If(${Element})
			Set(${RESULT} "TRUE")
		EndIf()
	EndForEach(Element)
EndMacro()

MACRO(MergeCFLAGS)
	SET(MERGED_C_FLAGS ${CMAKE_C_FLAGS})
	SET(MERGED_CXX_FLAGS ${CMAKE_CXX_FLAGS})

	IF (CMAKE_BUILD_TYPE MATCHES Release)
		SET(MERGED_C_FLAGS "${MERGED_C_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
		SET(MERGED_CXX_FLAGS "${MERGED_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
	ELSEIF (CMAKE_BUILD_TYPE MATCHES Distribution)
		SET(MERGED_C_FLAGS "${MERGED_C_FLAGS} ${CMAKE_C_FLAGS_DISTRIBUTION}")
		SET(MERGED_CXX_FLAGS "${MERGED_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DISTRIBUTION}")
	ELSEIF (CMAKE_BUILD_TYPE MATCHES Debug)
		SET(MERGED_C_FLAGS "${MERGED_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
		SET(MERGED_CXX_FLAGS "${MERGED_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
	ENDIF()
ENDMACRO()

macro(DetectSystemName)
	IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
		SET(DARWIN 1)
		set(CMAKE_MACOSX_RPATH TRUE)

		# use, i.e. don't skip the full RPATH for the build tree
		set(CMAKE_SKIP_BUILD_RPATH FALSE)

		# when building, don't use the install RPATH already
		# (but later on when installing)
		set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

		set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

		# add the automatically determined parts of the RPATH
		# which point to directories outside the build tree to the install RPATH
		set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

		# the RPATH to be used when installing, but only if it's not a system directory
		list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
		if(${isSystemDir} STREQUAL "-1")
			set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
		endif(${isSystemDir} STREQUAL "-1")
	ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
		SET(LINUX 1)
	ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
		SET(FREEBSD 1)
	ENDIF()
endmacro()

MACRO(PrintInterfaceStatus INTERFACE_NAME INTERFACE_FLAG)
	IF ( ${INTERFACE_FLAG} )
		message(STATUS "  ${INTERFACE_NAME} is ON")
	ELSE()
		STRING(LENGTH ${INTERFACE_NAME} IFACE_NAME_LENGTH)
		IF (IFACE_NAME_LENGTH LESS 3)
			SET(INDENT "\t\t\t")
		ELSEIF (IFACE_NAME_LENGTH LESS 10)
			SET(INDENT "\t\t")
		ELSE ()
			SET(INDENT "\t")
		ENDIF ()
		message(STATUS "  ${INTERFACE_NAME} is OFF ${INDENT} enable with -D${INTERFACE_FLAG}=ON")
		UNSET(INDENT)
	ENDIF()
ENDMACRO()

# based on compiz_discover_tests
function (shogun_discover_tests EXECUTABLE)

        add_dependencies (${EXECUTABLE} discover_gtest_tests)

        add_custom_command (TARGET ${EXECUTABLE}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -D UNIT_TEST_CMD=${CMAKE_BINARY_DIR}/bin/${EXECUTABLE}
                     -D DISCOVER_CMD=${CMAKE_BINARY_DIR}/bin/discover_gtest_tests
                     -D WORKING_DIR=${CMAKE_CURRENT_BINARY_DIR}
                     -P ${CMAKE_MODULE_PATH}/discover_unit_tests.cmake
            COMMENT "Discovering Tests in ${EXECUTABLE}"
            DEPENDS
            VERBATIM)
endfunction ()

MACRO(AddMetaIntegrationTest META_TARGET CONDITION)
    IF (${CONDITION})
        add_test(NAME integration_meta_${META_TARGET}-${NAME_WITH_DIR}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        COMMAND meta_example_integration_tester ${REL_DIR} ${NAME}.dat ${META_TARGET} generated_results reference_results)
                    set_tests_properties(
                        integration_meta_${META_TARGET}-${NAME_WITH_DIR}
	                        PROPERTIES
	                        LABELS "integration"
	                        DEPENDS generated_${META_TARGET}-${NAME_WITH_DIR}
                    )
    ENDIF()
ENDMACRO()

MACRO(AddLibShogunExample EXAMPLE_CPP)
	STRING(REGEX REPLACE ".cpp\$" "" EXAMPLE "${EXAMPLE_CPP}")

	add_executable(${EXAMPLE} EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_CPP})
	if(WIN32)
		target_link_libraries(${EXAMPLE} shogun::shogun-static ${SANITIZER_LIBRARY})
	else()
		target_link_libraries(${EXAMPLE} shogun::shogun ${SANITIZER_LIBRARY})
	endif()
	IF(SANITIZER_FLAGS)
		set_target_properties(${EXAMPLE} PROPERTIES COMPILE_FLAGS ${SANITIZER_FLAGS})
	ENDIF()

	# Add examples to the dependencies of modular interfaces to make sure
	# nothing will infer with them being build single-threaded.
	IF(SWIG_SINGLE_THREADED)
		FOREACH(SG_INTERFACE_TARGET ${SG_INTERFACE_TARGETS})
			ADD_DEPENDENCIES(${SG_INTERFACE_TARGET} ${EXAMPLE})
		ENDFOREACH(SG_INTERFACE_TARGET ${SG_INTERFACE_TARGETS})
	ENDIF(SWIG_SINGLE_THREADED)
ENDMACRO()


function(PrintLine)
	message(STATUS "===================================================================================================================")
endfunction()

function(PrintStatus MSG)
	message(STATUS "${MSG}")
endfunction()

# FIXME: add support for modern target based dependency
#FIND_PACKAGE(target [REQUIRED] [VERSION])
#if (<target>_FOUND)
#  set(target_flag 1)
#  target_link_libraries(shogun SCOPE <target>)
#endif()
macro(ADD_LIBRARY_DEPENDENCY)
	set(options REQUIRED)
	set(oneValueArgs LIBRARY CONFIG_FLAG VERSION SCOPE)
	set(multiValueArgs TARGETS)
	cmake_parse_arguments(ADD_LIBRARY_DEPENDENCY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
	SET(LIBRARY_PREFIX ${ADD_LIBRARY_DEPENDENCY_LIBRARY})
	STRING(TOUPPER ${ADD_LIBRARY_DEPENDENCY_LIBRARY} LIBRARY_PREFIX_UPPER)
	OPTION(ENABLE_${LIBRARY_PREFIX_UPPER} "Use ${LIBRARY_PREFIX}" ON)
	if (${ADD_LIBRARY_DEPENDENCY_REQUIRED})
		find_package(${ADD_LIBRARY_DEPENDENCY_LIBRARY} REQUIRED ${ADD_LIBRARY_DEPENDENCY_VERSION})
	else()
		find_package(${ADD_LIBRARY_DEPENDENCY_LIBRARY} ${ADD_LIBRARY_DEPENDENCY_VERSION})
	endif()
	if ((${LIBRARY_PREFIX}_FOUND OR ${LIBRARY_PREFIX_UPPER}_FOUND) AND ENABLE_${LIBRARY_PREFIX_UPPER})
		if (${LIBRARY_PREFIX}_INCLUDE_DIR)
			set(LIBRARY_HEADER ${${LIBRARY_PREFIX}_INCLUDE_DIR})
		elseif (${LIBRARY_PREFIX_UPPER}_INCLUDE_DIR)
			set(LIBRARY_HEADER ${${LIBRARY_PREFIX_UPPER}_INCLUDE_DIR})
		elseif (${LIBRARY_PREFIX}_INCLUDE_DIRS)
			set(LIBRARY_HEADER ${${LIBRARY_PREFIX}_INCLUDE_DIRS})
		elseif (${LIBRARY_PREFIX_UPPER}_INCLUDE_DIRS)
			set(LIBRARY_HEADER ${${LIBRARY_PREFIX_UPPER}_INCLUDE_DIRS})
		else ()
			message(${${LIBRARY_PREFIX}_INCLUDE_DIR})
			message(FATAL_ERROR "Found ${ADD_LIBRARY_DEPENDENCY_LIBRARY}, but not it's headers!")
		endif()

		set(${ADD_LIBRARY_DEPENDENCY_CONFIG_FLAG} ON CACHE BOOL "Use ${LIBRARY_PREFIX}" FORCE)

		if (${LIBRARY_PREFIX}_LIBRARIES)
			set(LIBRARY_LIBS ${${LIBRARY_PREFIX}_LIBRARIES})
		elseif(${LIBRARY_PREFIX_UPPER}_LIBRARIES)
			set(LIBRARY_LIBS ${${LIBRARY_PREFIX_UPPER}_LIBRARIES})
		else()
			message(FATAL_ERROR "Found ${ADD_LIBRARY_DEPENDENCY_LIBRARY}, but not it's libraries!")
		endif()

		ForEach (element ${ADD_LIBRARY_DEPENDENCY_TARGETS})
			if (TARGET ${element})
				get_target_property(TARGET_TYPE ${element} TYPE)
				if (${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
					target_include_directories(${element} INTERFACE ${LIBRARY_HEADER})
					target_link_libraries(${element} INTERFACE ${LIBRARY_LIBS})
				else()
					if (NOT ${TARGET_TYPE} STREQUAL OBJECT_LIBRARY)
						target_link_libraries(${element} ${ADD_LIBRARY_DEPENDENCY_SCOPE} ${LIBRARY_LIBS})
					endif()
					target_include_directories(${element} ${ADD_LIBRARY_DEPENDENCY_SCOPE} ${LIBRARY_HEADER})
				endif()
			endif()
		EndForEach()
	else()
		set(${ADD_LIBRARY_DEPENDENCY_CONFIG_FLAG} OFF CACHE BOOL "Use ${LIBRARY_PREFIX}" FORCE)
	endif()
endmacro()

macro(SHOGUN_DEPENDENCIES)
	ADD_LIBRARY_DEPENDENCY(TARGETS shogun shogun-static libshogun shogun_deps ${ARGN})
endmacro()

function(SHOGUN_LINK_LIBS)
	set(SCOPE PRIVATE)
	target_link_libraries(shogun ${SCOPE} ${ARGN})
	if (LIBSHOGUN_BUILD_STATIC)
		target_link_libraries(shogun-static ${SCOPE} ${ARGN})
	endif()
	target_link_libraries(shogun_deps INTERFACE ${ARGN})
endfunction()

function(SHOGUN_COMPILE_OPTS)
	set(SCOPE PRIVATE)
	target_compile_options(libshogun ${SCOPE} ${ARGN})
	target_compile_options(shogun ${SCOPE} ${ARGN})
	if (LIBSHOGUN_BUILD_STATIC)
		target_compile_options(shogun-static ${SCOPE} ${ARGN})
	endif()
	target_compile_options(shogun_deps INTERFACE ${ARGN})
endfunction()

function(SHOGUN_INCLUDE_DIRS)
	set(options SYSTEM)
	set(oneValueArgs SCOPE)
	set(multiValueArgs)
	cmake_parse_arguments(SHOGUN_INCLUDE_DIRS "${options}" "${oneValueArgs}" "multiValueArgs" ${ARGN})
	SET(DIRS ${SHOGUN_INCLUDE_DIRS_UNPARSED_ARGUMENTS})
	if(SHOGUN_INCLUDE_DIRS_SYSTEM)
		set(SYSTEM "SYSTEM")
	endif()
	target_include_directories(libshogun ${SYSTEM} ${SHOGUN_INCLUDE_DIRS_SCOPE} ${DIRS})
	target_include_directories(shogun ${SYSTEM} ${SHOGUN_INCLUDE_DIRS_SCOPE} ${DIRS})
	if (LIBSHOGUN_BUILD_STATIC)
		target_include_directories(shogun-static ${SYSTEM} ${SHOGUN_INCLUDE_DIRS_SCOPE} ${DIRS})
	endif()
	target_include_directories(shogun_deps ${SYSTEM} INTERFACE ${DIRS})
endfunction()

function(SET_LINALG_BACKEND COMPONENT FLAG)
	OPTION(USE_EIGEN3_${FLAG} "Use ${COMPONENT} Eigen3" ON)
	CMAKE_DEPENDENT_OPTION(
		USE_VIENNACL_${FLAG} "Use ${COMPONENT} ViennaCL" OFF
        "VIENNACL_FOUND;USE_VIENNACL;NOT USE_EIGEN3_${FLAG}" ON)
	if(NOT ${COMPONENT})
	  set(${COMPONENT} EIGEN3 CACHE STRING
		"Set linear algebra backend ${COMPONENT}: EIGEN3, VIENNACL"
		FORCE)
	endif()

	if (${COMPONENT} STREQUAL "EIGEN3")
		MESSAGE(STATUS "Eigen3 set as default ${COMPONENT}")
	elseif (${COMPONENT} STREQUAL "VIENNACL")
		set(${COMPONENT}_USE_EIGEN3 OFF CACHE BOOL "Use ${COMPONENT}_USE_EIGEN3" FORCE)
		IF (USE_VIENNACL_${FLAG})
			MESSAGE(STATUS "ViennaCL set as default ${COMPONENT}")
		ELSE()
			MESSAGE(FATAL_ERROR "Could NOT set ViennaCL as ${COMPONENT}!")
		ENDIF ()
	ENDIF ()
endfunction()

macro(CREATE_DATA_SYMLINK SRC DST)
	IF(WIN32)
		EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E copy_directory ${SRC} ${DST})
	ELSE()
		EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SRC} ${DST})
	ENDIF()
endmacro()

function(GET_META_EXAMPLE_VARS META_EXAMPLE EX_NAME REL_DIR NAME_WITH_DIR)
	get_filename_component(EXAMPLE_NAME ${META_EXAMPLE} NAME_WE)
	set(${EX_NAME} ${EXAMPLE_NAME} PARENT_SCOPE)
	get_filename_component(FULL_DIR ${META_EXAMPLE} DIRECTORY)
	get_filename_component(EXAMPLE_REL_DIR ${FULL_DIR} NAME)
	set(${REL_DIR} ${EXAMPLE_REL_DIR} PARENT_SCOPE)
	set(EXAMPLE_NAME_WITH_DIR "${EXAMPLE_REL_DIR}-${EXAMPLE_NAME}")
	set(${NAME_WITH_DIR} ${EXAMPLE_NAME_WITH_DIR} PARENT_SCOPE)
endfunction()

function(GET_INTERFACE_VARS INTERFACE DIRECTORY EXTENSION)
    string(REGEX MATCH "INTERFACE_([a-zA-Z]+)" _dir ${INTERFACE})
	STRING(TOLOWER "${CMAKE_MATCH_1}" _dir)
	SET(${DIRECTORY} ${_dir} PARENT_SCOPE)


	# set the extension
	if (${_dir} STREQUAL "python")
		SET(${EXTENSION} "py" PARENT_SCOPE)
	elseif(${_dir} STREQUAL "octave")
		SET(${EXTENSION} "m" PARENT_SCOPE)
	elseif(${_dir} STREQUAL "java")
		SET(${EXTENSION} "java" PARENT_SCOPE)
	elseif(${_dir} STREQUAL "csharp")
		SET(${EXTENSION} "cs" PARENT_SCOPE)
	elseif(${_dir} STREQUAL "lua")
		SET(${EXTENSION} "lua" PARENT_SCOPE)
	elseif(${_dir} STREQUAL "r")
		SET(${EXTENSION} "R" PARENT_SCOPE)
	elseif(${_dir} STREQUAL "ruby")
		SET(${EXTENSION} "rb" PARENT_SCOPE)
	elseif(${_dir} STREQUAL "scala")
		SET(${EXTENSION} "scala" PARENT_SCOPE)
	else()
		MESSAGE(FATAL_ERROR "Undefined interface ${INTERFACE}")
	endif()
endfunction()

# inspired by arrow's benchmarking:
# https://github.com/apache/arrow/blob/apache-arrow-0.9.0/cpp/cmake_modules/BuildUtils.cmake#L223
function(ADD_SHOGUN_BENCHMARK REL_BENCHMARK_NAME)
	if(NOT BUILD_BENCHMARKS)
		return()
	endif()
	get_filename_component(BENCHMARK_NAME ${REL_BENCHMARK_NAME} NAME_WE)

	if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${REL_BENCHMARK_NAME}.cc)
		# This benchmark has a corresponding .cc file, set it up as an executable.
		add_executable(${BENCHMARK_NAME} "${REL_BENCHMARK_NAME}.cc")
		set_target_properties (${BENCHMARK_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
		set_target_properties (${BENCHMARK_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
		set_target_properties (${BENCHMARK_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
		target_link_libraries(${BENCHMARK_NAME} ${SHOGUN_BENCHMARK_LINK_LIBS})
		set(NO_COLOR "--color_print=false")
	endif()

	add_test(${BENCHMARK_NAME} ${CMAKE_BINARY_DIR}/bin/${BENCHMARK_NAME} ${NO_COLOR})
	set_tests_properties(${BENCHMARK_NAME} PROPERTIES LABELS "benchmark")
	if(ARGN)
		set_tests_properties(${BENCHMARK_NAME} PROPERTIES ${ARGN})
	endif()
endfunction()
