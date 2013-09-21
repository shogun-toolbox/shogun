if(SYSTEM_INCLUDES)
  INCLUDE_DIRECTORIES(SYSTEM ${SYSTEM_INCLUDES})
endif()
include_directories(${INCLUDES} . ..)

FILE(GLOB_RECURSE LIBSHOGUN_SRC *.${EXT_SRC_CPP} *.${EXT_SRC_C})
FILE(GLOB_RECURSE LIBSHOGUN_HEADERS *.${EXT_SRC_HEADER})
FILE(GLOB_RECURSE LIBSHOGUN_SRC_TMP *.${EXT_CPP_TMP})

add_library(shogun SHARED ${LIBSHOGUN_SRC} lib/config.h base/class_list.cpp)

# Generate versionstring.h
ADD_CUSTOM_TARGET(
    version
    ${CMAKE_COMMAND} -D SRC=${CMAKE_CURRENT_SOURCE_DIR}/lib/versionstring.h.in
                     -D DST=${CMAKE_CURRENT_SOURCE_DIR}/lib/versionstring.h
                     -D ROOT_DIR=${CMAKE_SOURCE_DIR}
                     -D PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
                     -P ${CMAKE_MODULE_PATH}/version.cmake
    COMMENT "Generating version header"
)
add_dependencies(shogun version)

# generate source files from templates
FOREACH(template ${LIBSHOGUN_SRC_TMP})
  STRING(REGEX REPLACE ${EXT_CPP_TMP} ${EXT_SRC_CPP} generated_cpp "${template}")
  STRING(REGEX REPLACE ${EXT_CPP_TMP} ${EXT_CPP_PY} generator_script "${template}")
  STRING(REGEX REPLACE ".*/(.*).${EXT_CPP_TMP}" "\\1" generated_target "${template}")
  
  IF(MSVC)
    FOREACH(h ${LIBSHOGUN_HEADERS})
      FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/headers_list.txt "${h}\n")
    ENDFOREACH()
  
    ADD_CUSTOM_COMMAND(OUTPUT ${generated_cpp}
        COMMAND ${CMAKE_COMMAND} -E copy ${template} ${generated_cpp}
        COMMAND ${PYTHON_EXECUTABLE} ${generator_script} ${generated_cpp} -in ${CMAKE_CURRENT_BINARY_DIR}/headers_list.txt
        DEPENDS ${template} ${generator_script} ${LIBSHOGUN_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/headers_list.txt
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating ${generated_cpp}"
    )
  ELSE()
    ADD_CUSTOM_COMMAND(OUTPUT ${generated_cpp}
        COMMAND ${CMAKE_COMMAND} -E copy ${template} ${generated_cpp}
        COMMAND ${PYTHON_EXECUTABLE} ${generator_script} ${generated_cpp} ${LIBSHOGUN_HEADERS}
        DEPENDS ${template} ${generator_script} ${LIBSHOGUN_HEADERS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating ${generated_cpp}"
    )
  ENDIF()

  ADD_CUSTOM_TARGET(${generated_target} DEPENDS ${generated_cpp})
  add_dependencies(shogun ${generated_target})
ENDFOREACH()

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/lib/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/lib/config.h)
set_target_properties(shogun PROPERTIES 
                      VERSION ${LIBSHOGUNVER}
                      SOVERSION ${LIBSHOGUNSO}
                      COMPILE_DEFINITIONS "${DEFINES}")
IF (SANITIZER_FLAGS)
  set_target_properties(shogun PROPERTIES COMPILE_FLAGS ${SANITIZER_FLAGS})
ENDIF()
target_link_libraries(shogun ${POSTLINKFLAGS})

# add bundled libraries to dependency
if(SHOGUN_DEPENDS)
  add_dependencies(shogun ${SHOGUN_DEPENDS})
endif()

INSTALL(DIRECTORY ./ 
        DESTINATION include/shogun
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
        PATTERN ".settings" EXCLUDE)
SET(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} headers)

IF (LIB_INSTALL_DIR)
  SET(SHOGUN_LIB_INSTALL ${LIB_INSTALL_DIR})
ELSE()
  SET(SHOGUN_LIB_INSTALL lib)
ENDIF()

INSTALL(TARGETS shogun 
  ARCHIVE DESTINATION ${SHOGUN_LIB_INSTALL}
  LIBRARY DESTINATION ${SHOGUN_LIB_INSTALL}
  COMPONENT libraries)

SET(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} libraries PARENT_SCOPE)

# cpack
set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Libraries" PARENT_SCOPE)
set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION "Shogun libraries" PARENT_SCOPE)
set(CPACK_COMPONENT_LIBRARIES_GROUP "Runtime" PARENT_SCOPE)

set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "C++ Headers" PARENT_SCOPE)
set(CPACK_COMPONENT_HEADERS_DESCRIPTION "C/C++ header files for Shogun" PARENT_SCOPE)
set(CPACK_COMPONENT_HEADERS_GROUP "Development" PARENT_SCOPE)

set(CPACK_COMPONENT_GROUP_DEVELOPMENT_EXPANDED ON PARENT_SCOPE)
set(CPACK_COMPONENT_GROUP_DEVELOPMENT_DESCRIPTION
   "All of the packages that's required for developing with Shogun" PARENT_SCOPE)

set(CPACK_COMPONENT_HEADERS_DEPENDS libraries PARENT_SCOPE)

