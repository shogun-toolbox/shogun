if(SYSTEM_INCLUDES)
  INCLUDE_DIRECTORIES(SYSTEM ${SYSTEM_INCLUDES})
endif()
include_directories(${INCLUDES})

include(GenerateExportHeader)

FILE(GLOB_RECURSE LIBSHOGUN_SRC *.${EXT_SRC_CPP} *.${EXT_SRC_C})
FILE(GLOB_RECURSE LIBSHOGUN_HEADERS *.${EXT_SRC_HEADER})
FILE(GLOB_RECURSE LIBSHOGUN_SRC_TMP *.${EXT_CPP_TMP})

IF (PROTOBUF_FOUND)
	FILE(GLOB protobuf_src ${CMAKE_CURRENT_SOURCE_DIR}/io/protobuf/*.proto)

	FOREACH(FIL ${protobuf_src})
		get_filename_component(FIL_WE ${FIL} NAME_WE)
		SET(name "${CMAKE_CURRENT_SOURCE_DIR}/io/protobuf/${FIL_WE}")

		if (EXISTS "${name}.pb.cc")
			file(REMOVE "${name}.pb.cc")
		endif()
		if (EXISTS "${name}.pb.h")
			file(REMOVE "${name}.pb.h")
		endif()

		list(APPEND LIBSHOGUN_SRC "${name}.pb.cc")
		list(APPEND LIBSHOGUN_HEADERS "${name}.pb.h")
	ENDFOREACH()
ENDIF()

# add target to compile the libshogun sources
add_library(libshogun OBJECT ${LIBSHOGUN_SRC} base/class_list.cpp lib/config.h.in)
set_property(TARGET libshogun PROPERTY POSITION_INDEPENDENT_CODE TRUE)
IF (SANITIZER_FLAGS)
  set_property(libshogun PROPERTY COMPILE_FLAGS ${SANITIZER_FLAGS})
ENDIF()

IF (PROTOBUF_FOUND)
	FOREACH(FIL ${protobuf_src})
		get_filename_component(FIL_WE ${FIL} NAME_WE)

		SET(name "${CMAKE_CURRENT_SOURCE_DIR}/io/protobuf/${FIL_WE}")

		ADD_CUSTOM_COMMAND(
			 OUTPUT "${name}.pb.cc" "${name}.pb.h"
			 COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${FIL}
				 --cpp_out ${CMAKE_CURRENT_SOURCE_DIR}/io/protobuf
				 --proto_path ${CMAKE_CURRENT_SOURCE_DIR}/io/protobuf
			 COMMENT "Running C++ protocol buffer compiler on ${FIL_WE}.proto"
			)
		ADD_CUSTOM_TARGET("${FIL_WE}Protobuf" DEPENDS "${name}.pb.cc")
		add_dependencies(libshogun "${FIL_WE}Protobuf")
	ENDFOREACH()
ENDIF()

# Generate versionstring.h
ADD_CUSTOM_TARGET(
    version
    ${CMAKE_COMMAND} -D SRC=${CMAKE_CURRENT_SOURCE_DIR}/lib/versionstring.h.in
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
        COMMAND ${CMAKE_COMMAND} -E copy ${template} ${generated_cpp}
        COMMAND ${PYTHON_EXECUTABLE} ${generator_script} ${generated_cpp} -in ${CMAKE_CURRENT_BINARY_DIR}/headers_list.txt
        DEPENDS ${template} ${generator_script} version ${LIBSHOGUN_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/headers_list.txt
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating ${generated_cpp}"
    )
  ELSE()
    ADD_CUSTOM_COMMAND(OUTPUT ${generated_cpp}
        COMMAND ${CMAKE_COMMAND} -E copy ${template} ${generated_cpp}
        COMMAND ${PYTHON_EXECUTABLE} ${generator_script} ${generated_cpp} ${LIBSHOGUN_HEADERS}
        DEPENDS ${template} ${generator_script} version ${LIBSHOGUN_HEADERS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating ${generated_cpp}"
    )
  ENDIF()

  ADD_CUSTOM_TARGET(${generated_target} DEPENDS ${generated_cpp})
  add_dependencies(libshogun ${generated_target})
ENDFOREACH()

# add target for dynamic
if(MSVC)
  SET(POSTLINKFLAGS ${POSTLINKFLAGS} winmm)
endif()
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
target_link_libraries(shogun ${POSTLINKFLAGS})

# add bundled libraries to dependency
if(SHOGUN_DEPENDS)
  add_dependencies(libshogun ${SHOGUN_DEPENDS})
endif()

# add target for static library if enabled
if (BUILD_STATIC)
  add_library(shogun-static STATIC $<TARGET_OBJECTS:libshogun> ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h)
  set_property(TARGET shogun-static PROPERTY OUTPUT_NAME shogun)
  if(MSVC)
    target_link_libraries(shogun winmm.lib)
  endif()
ENDIF()

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

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/lib/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h)

set(INCLUDE_INSTALL_DIR include/shogun)
INSTALL(
  FILES
    ${LIBSHOGUN_HEADERS}
    "${CMAKE_CURRENT_BINARY_DIR}/shogun_export.h"
    ${CMAKE_CURRENT_BINARY_DIR}/lib/config.h
    ${CMAKE_CURRENT_BINARY_DIR}/lib/versionstring.h
  DESTINATION include/shogun
  COMPONENT headers)

IF (LIB_INSTALL_DIR)
  SET(SHOGUN_LIB_INSTALL ${LIB_INSTALL_DIR})
ELSE()
  SET(SHOGUN_LIB_INSTALL lib)
ENDIF()

INSTALL(
  TARGETS shogun 
  EXPORT ShogunTargets
  ARCHIVE DESTINATION ${SHOGUN_LIB_INSTALL}
  LIBRARY DESTINATION ${SHOGUN_LIB_INSTALL}
  INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
)

include(CMakePackageConfigHelpers)
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
