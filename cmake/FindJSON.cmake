macro(_json_get_version)
  file(READ "${JSON_INCLUDE_DIR}/json_c_version.h" _json_version_header)

  string(REGEX MATCH "define[ \t]+JSON_C_MAJOR_VERSION[ \t]+([0-9]+)" _json_major_version_match "${_json_version_header}")
  set(JSON_C_MAJOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+JSON_C_MINOR_VERSION[ \t]+([0-9]+)" _json_minor_version_match "${_json_version_header}")
  set(JSON_C_MINOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+JSON_C_MICRO_VERSION[ \t]+([0-9]+)" _json_micro_version_match "${_json_version_header}")
  set(JSON_C_MICRO_VERSION "${CMAKE_MATCH_1}")

  set(JSON_VERSION_STRING "${JSON_C_MAJOR_VERSION}.${JSON_C_MINOR_VERSION}.${JSON_C_MICRO_VERSION}")

endmacro(_json_get_version)

find_path(JSON_INCLUDE_DIR NAMES json_c_version.h json.h 
    PATHS
        ${CMAKE_INSTALL_PREFIX}/include
        PATH_SUFFIXES json json-c
    )
find_library(JSON_LIBRARY NAMES json-c json)

if(JSON_INCLUDE_DIR)
    _json_get_version()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JSON
    REQUIRED_VARS JSON_INCLUDE_DIR JSON_LIBRARY
    VERSION_VAR JSON_VERSION_STRING)

if(JSON_FOUND)
    set(JSON_LIBRARIES ${JSON_LIBRARY})
endif()

mark_as_advanced(JSON_INCLUDE_DIR JSON_LIBRARIES)
