MACRO(GENERATE_MODULAR_TARGET MODULAR_NAME MODULAR_DIR MODULAR_LIBARIES)

get_target_property(ShogunIncludes shogun::shogun INTERFACE_INCLUDE_DIRECTORIES)
INCLUDE_DIRECTORIES(${ShogunIncludes})

# set compiler SWIG generated cxx compiler flags
SET(CMAKE_CXX_FLAGS ${SWIG_CXX_COMPILER_FLAGS})
# unset any release or distribution flags
# we don't want them when compiling SWIG generated source
SET(CMAKE_CXX_FLAGS_RELEASE "")
SET(CMAKE_CXX_FLAGS_DISTRIBUTION "")
SET(CMAKE_CXX_FLAGS_DEBUG "")

if(${MODULAR_NAME} STREQUAL "python")
	SET(PREPEND_TARGET "_")
endif()

set(modular_files)
FILE(GLOB_RECURSE MODULAR_FILES ${COMMON_MODULAR_SRC_DIR}/*.i)
FILE(GLOB_RECURSE CUSTOM_MODULAR_FILES ${MODULAR_DIR}/*.i)
LIST(APPEND MODULAR_FILES ${CUSTOM_MODULAR_FILES})
FOREACH(file ${MODULAR_FILES})
	get_filename_component(fname ${file} NAME)
	list(APPEND modular_files ${fname})
	ADD_CUSTOM_COMMAND(OUTPUT ${fname}
		DEPENDS ${file}
		COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${file} ${fname}
		COMMENT ""
	)
ENDFOREACH()

ADD_CUSTOM_TARGET(${MODULAR_NAME}_modular_src
	DEPENDS shogun::shogun ${modular_files}
	COMMENT "copying SWIG files")

INCLUDE(${SWIG_USE_FILE})
SET_SOURCE_FILES_PROPERTIES(modshogun.i PROPERTIES CPLUSPLUS ON)
IF(DEFINED TARGET_SWIGFLAGS)
	SET_SOURCE_FILES_PROPERTIES(modshogun.i PROPERTIES SWIG_FLAGS ${TARGET_SWIGFLAGS})
ENDIF()
SET(SWIG_MODULE_${MODULAR_NAME}_modular_EXTRA_DEPS ${modular_files})
SWIG_ADD_MODULE(${MODULAR_NAME}_modular ${MODULAR_NAME} modshogun.i sg_print_functions.cpp)
SWIG_LINK_LIBRARIES(${MODULAR_NAME}_modular shogun::shogun ${MODULAR_LIBARIES})
SET_TARGET_PROPERTIES(${SWIG_MODULE_${MODULAR_NAME}_modular_REAL_NAME} PROPERTIES OUTPUT_NAME ${PREPEND_TARGET}modshogun)
ADD_DEPENDENCIES(${SWIG_MODULE_${MODULAR_NAME}_modular_REAL_NAME} ${MODULAR_NAME}_modular_src)

#ADD_CUSTOM_COMMAND(TARGETS ${PREPEND_TARGET}${MODULAR_NAME}_modular
#				   POST_BUILD
#				   COMMAND ${PYTHON_EXECUTABLE}
#				   ARGS ${CMAKE_SOURCE_DIR}/src/.scrub_docstrings.py )

IF(DOXYGEN_FOUND)
	configure_file(${COMMON_MODULAR_SRC_DIR}/modshogun.doxy.in modshogun.doxy)

	ADD_CUSTOM_COMMAND(
	OUTPUT    modshogun
	COMMAND   ${DOXYGEN_EXECUTABLE}
	ARGS	  modshogun.doxy
	DEPENDS   shogun::shogun
	COMMENT   "Generating doxygen doc"
	)

	ADD_CUSTOM_COMMAND(
	OUTPUT    modshogun_doxygen.i
	COMMAND   ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/src/.doxy2swig.py
	ARGS	  --quiet --no-function-definition modshogun/doxygen_xml/index.xml modshogun_doxygen.i
	DEPENDS   modshogun
	)
	ADD_CUSTOM_TARGET(${MODULAR_NAME}_doxy2swig DEPENDS modshogun_doxygen.i)
	ADD_DEPENDENCIES(${SWIG_MODULE_${MODULAR_NAME}_modular_REAL_NAME} ${MODULAR_NAME}_doxy2swig)
ELSE()
	#TODO add scrubing
ENDIF()

# Make sure all modular interfaces are build single-threaded to reduce
# excessive memory consumption during build.
IF(SWIG_SINGLE_THREADED)
	FOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
		ADD_DEPENDENCIES(${SWIG_MODULE_${MODULAR_NAME}_modular_REAL_NAME}
			${SG_MODULAR_INTERFACE_TARGET})
	ENDFOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
	SET(SG_MODULAR_INTERFACE_TARGETS
		"${SWIG_MODULE_${MODULAR_NAME}_modular_REAL_NAME};${SG_MODULAR_INTERFACE_TARGETS}"
		CACHE STRING "List of modular-interfaces beeing build." FORCE)
ENDIF(SWIG_SINGLE_THREADED)

CONFIGURE_FILE(${COMMON_MODULAR_SRC_DIR}/swig_config.h.in swig_config.h)

ENDMACRO()
