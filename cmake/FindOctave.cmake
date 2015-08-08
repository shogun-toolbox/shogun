# Try to find the build flags to compile octave shared objects (oct and mex files)
# Once done this will define
#
# MKOCTFILE_EXECUTABLE - mkoctfile executable
# OCTAVE_API_VERSION - octave api version
# OCTAVE_CONFIG_EXECUTABLE - octave-config executable
# OCTAVE_EXECUTABLE - octave interpreter
# OCTAVE_FOUND - if Octave is found
# OCTAVE_CXXFLAGS - extra flags
# OCTAVE_INCLUDE_DIRS - include directories
# OCTAVE_LINK_DIRS - link directories
# OCTAVE_LIBRARY_RELEASE - the release version
# OCTAVE_LIBRARY_DEBUG - the debug version
# OCTAVE_LIBRARY - a default library, with priority debug.
# OCTAVE_OCT_LOCAL_API_FILE_DIR - installation directory

# find octave-config-executable
set(OCTAVE_CONFIG_EXECUTABLE OCTAVE_CONFIG_EXECUTABLE-NOTFOUND)
find_program(OCTAVE_CONFIG_EXECUTABLE NAMES octave-config octave-config-3.2.4)

# use mkoctfile
set(MKOCTFILE_EXECUTABLE MKOCTFILE_EXECUTABLE-NOTFOUND)
find_program(MKOCTFILE_EXECUTABLE NAME mkoctfile PATHS)

if(OCTAVE_CONFIG_EXECUTABLE AND MKOCTFILE_EXECUTABLE)
  set(OCTAVE_FOUND 1)

  execute_process ( COMMAND ${OCTAVE_CONFIG_EXECUTABLE} -p LOCALAPIOCTFILEDIR
    OUTPUT_VARIABLE OCTAVE_OCT_LOCAL_API_FILE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE )

  execute_process ( COMMAND ${OCTAVE_CONFIG_EXECUTABLE} -p API_VERSION
    OUTPUT_VARIABLE OCTAVE_API_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE )
  STRING(REGEX REPLACE "[a-z\\+-]" "" OCTAVE_API_VERSION ${OCTAVE_API_VERSION})

  execute_process ( COMMAND ${OCTAVE_CONFIG_EXECUTABLE} -v
    OUTPUT_VARIABLE OCTAVE_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE )

  find_program(OCTAVE_EXECUTABLE NAMES octave octave-3.2.4)

  message(STATUS "Found Octave:      ${OCTAVE_EXECUTABLE}")
  message(STATUS "Octave-version:    ${OCTAVE_VERSION_STRING}")
  message(STATUS "Octave-API:        ${OCTAVE_API_VERSION}")
  message(STATUS "Octave-installdir: ${OCTAVE_OCT_LOCAL_API_FILE_DIR}")

  execute_process(
    COMMAND ${MKOCTFILE_EXECUTABLE} -p ALL_CXXFLAGS
    OUTPUT_VARIABLE _mkoctfile_cppflags
    RESULT_VARIABLE _mkoctfile_failed)
  string(REGEX REPLACE "[\r\n]" " " _mkoctfile_cppflags "${_mkoctfile_cppflags}")
  execute_process(
    COMMAND ${MKOCTFILE_EXECUTABLE} -p INCFLAGS
    OUTPUT_VARIABLE _mkoctfile_includedir
    RESULT_VARIABLE _mkoctfile_failed)
  string(REGEX REPLACE "[\r\n]" " " _mkoctfile_includedir "${_mkoctfile_includedir}")
  execute_process(
    COMMAND ${MKOCTFILE_EXECUTABLE} -p ALL_LDFLAGS
    OUTPUT_VARIABLE _mkoctfile_ldflags
    RESULT_VARIABLE _mkoctfile_failed)
  string(REGEX REPLACE "[\r\n]" " " _mkoctfile_ldflags "${_mkoctfile_ldflags}")
  execute_process(
    COMMAND ${MKOCTFILE_EXECUTABLE} -p LFLAGS
    OUTPUT_VARIABLE _mkoctfile_lflags
    RESULT_VARIABLE _mkoctfile_failed)
  string(REGEX REPLACE "[\r\n]" " " _mkoctfile_lflags "${_mkoctfile_lflags}")
  execute_process(
    COMMAND ${MKOCTFILE_EXECUTABLE} -p LIBS
    OUTPUT_VARIABLE _mkoctfile_libs
    RESULT_VARIABLE _mkoctfile_failed)
  string(REGEX REPLACE "[\r\n]" " " _mkoctfile_libs "${_mkoctfile_libs}")
  execute_process(
    COMMAND ${MKOCTFILE_EXECUTABLE} -p OCTAVE_LIBS
    OUTPUT_VARIABLE _mkoctfile_octlibs
    RESULT_VARIABLE _mkoctfile_failed)
  string(REGEX REPLACE "[\r\n]" " " _mkoctfile_octlibs "${_mkoctfile_octlibs}")
  set(_mkoctfile_libs "${_mkoctfile_libs} ${_mkoctfile_octlibs}")

  string(REGEX MATCHALL "(^| )-l([./+-_\\a-zA-Z]*)" _mkoctfile_libs "${_mkoctfile_libs}")
  string(REGEX REPLACE "(^| )-l" "" _mkoctfile_libs "${_mkoctfile_libs}")

  string(REGEX MATCHALL "(^| )-L([./+-_\\a-zA-Z]*)" _mkoctfile_ldirs "${_mkoctfile_lflags}")
  string(REGEX REPLACE "(^| )-L" "" _mkoctfile_ldirs "${_mkoctfile_ldirs}")

  string(REGEX REPLACE "(^| )-l([./+-_\\a-zA-Z]*)" " " _mkoctfile_ldflags "${_mkoctfile_ldflags}")
  string(REGEX REPLACE "(^| )-L([./+-_\\a-zA-Z]*)" " " _mkoctfile_ldflags "${_mkoctfile_ldflags}")

  string(REGEX REPLACE "(^| )-I" " " _mkoctfile_includedir "${_mkoctfile_includedir}")

  separate_arguments(_mkoctfile_includedir)

  set( OCTAVE_CXXFLAGS "${_mkoctfile_cppflags}" )
  set( OCTAVE_LINK_FLAGS "${_mkoctfile_ldflags}" )
  set( OCTAVE_INCLUDE_DIRS ${_mkoctfile_includedir})
  set( OCTAVE_LINK_DIRS ${_mkoctfile_ldirs})
  set( OCTAVE_LIBRARY ${_mkoctfile_libs})
  set( OCTAVE_LIBRARY_RELEASE ${OCTAVE_LIBRARY})
  set( OCTAVE_LIBRARY_DEBUG ${OCTAVE_LIBRARY})
endif(OCTAVE_CONFIG_EXECUTABLE AND MKOCTFILE_EXECUTABLE)

MARK_AS_ADVANCED(
    MKOCTFILE_EXECUTABLE
    OCTAVE_API_VERSION
    OCTAVE_CONFIG_EXECUTABLE
    OCTAVE_EXECUTABLE
    OCTAVE_LIBRARY_FOUND
    OCTAVE_CXXFLAGS
    OCTAVE_LINK_FLAGS
    OCTAVE_INCLUDE_DIRS
    OCTAVE_LINK_DIRS
    OCTAVE_LIBRARY
    OCTAVE_LIBRARY_RELEASE
    OCTAVE_LIBRARY_DEBUG
    OCTAVE_OCT_LOCAL_API_FILE_DIR
    OCTAVE_VERSION_STRING
)
