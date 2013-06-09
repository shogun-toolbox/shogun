include_directories(. ..)

FILE(GLOB_RECURSE LIBSHOGUN_SRC *.${EXT_SRC_CPP} *.${EXT_SRC_C})
FILE(GLOB_RECURSE LIBSHOGUN_HEADERS *.${EXT_SRC_HEADER})
FILE(GLOB_RECURSE LIBSHOGUN_SRC_TMP *.${EXT_CPP_TMP})

add_library(shogun SHARED ${LIBSHOGUN_SRC} lib/config.h)

# Generate versionstring.h
ADD_CUSTOM_TARGET(
    version
    ${CMAKE_COMMAND} -D SRC=${CMAKE_CURRENT_SOURCE_DIR}/lib/versionstring.h.in
                     -D DST=${CMAKE_CURRENT_SOURCE_DIR}/lib/versionstring.h
                     -D ROOT_DIR=${CMAKE_SOURCE_DIR}
                     -P ${CMAKE_MODULE_PATH}/version.cmake
    COMMENT "Generating version header"
)
add_dependencies(shogun version)

# generate source files from templates
FOREACH(template ${LIBSHOGUN_SRC_TMP})
  STRING(REGEX REPLACE ${EXT_CPP_TMP} ${EXT_SRC_CPP} generated_cpp "${template}")
  STRING(REGEX REPLACE ${EXT_CPP_TMP} ${EXT_CPP_PY} generator_script "${template}")
  STRING(REGEX REPLACE ".*/(.*).${EXT_CPP_TMP}" "\\1" generated_target "${template}")
  
  ADD_CUSTOM_COMMAND(OUTPUT ${generated_cpp}
      COMMAND ${CMAKE_COMMAND} -E copy ${template} ${generated_cpp}
      COMMAND ${PYTHON_EXECUTABLE} ${generator_script} ${generated_cpp} ${LIBSHOGUN_HEADERS}
      DEPENDS ${template} ${generator_script} ${LIBSHOGUN_HEADERS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating ${generated_cpp}"
  )
  ADD_CUSTOM_TARGET(${generated_target} DEPENDS ${generated_cpp})
  add_dependencies(shogun ${generated_target})
ENDFOREACH()

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/lib/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/lib/config.h)

set_target_properties(shogun PROPERTIES VERSION ${LIBSHOGUNVER} SOVERSION ${LIBSHOGUNSO})
target_link_libraries(shogun ${POSTLINKFLAGS})
