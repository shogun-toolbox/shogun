# Inspired from https://github.com/frs69wq/simgrid/blob/master/buildtools/Cmake/Modules/FindScala.cmake
set(_SCALA_PATHS
  /opt
  /opt/local
  /opt/csw
  /sw
  /usr
  /usr/share/java
  )

find_package(Java COMPONENTS Runtime)
if(JAVA_FOUND)
    include(UseJava)
else()
    message(WARNING "JAVA count not be found!" "\nIt is required for Scala Modular Interface!!!")
endif()

find_program(Scala_SCALA_EXECUTABLE
  NAMES scala
  PATHS ${_SCALA_PATHS}
  )

find_program(Scala_SCALAC_EXECUTABLE
  NAMES scalac
  PATHS ${_SCALA_PATHS}
  )

find_jar(Scala_JAR_EXECUTABLE "scala-library")

if(Scala_SCALA_EXECUTABLE)
    execute_process(COMMAND ${Scala_SCALA_EXECUTABLE} -version
      RESULT_VARIABLE SCALA_SEARCH_SUCCESS
      OUTPUT_VARIABLE SCALA_VERSION
      ERROR_VARIABLE SCALA_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE)
    if( SCALA_SEARCH_SUCCESS )
      message( FATAL_ERROR "Error executing scala -version" )
    else()
      string(TOLOWER ${SCALA_VERSION} SCALA_VERSION)
      string( REGEX REPLACE ".*([0-9]+\\.[0-9]+\\.[0-9_.]+.*)" "\\1" SCALA_VERSION "${SCALA_VERSION}" )
      string( REGEX REPLACE "([0-9]+\\.[0-9]+\\.[0-9_.]).*" "\\1" SCALA_VERSION ${SCALA_VERSION} )
    endif()
endif()

include(FindPackageHandleStandardArgs)
if (CMAKE_VERSION LESS 2.8.3)
  find_package_handle_standard_args(SCALA DEFAULT_MSG Scala_SCALA_EXECUTABLE)
else ()
  find_package_handle_standard_args(SCALA 
      REQUIRED_VARS Scala_SCALA_EXECUTABLE Scala_SCALAC_EXECUTABLE Scala_JAR_EXECUTABLE
      VERSION_VAR SCALA_VERSION)
endif ()

mark_as_advanced(
  Scala_SCALA_EXECUTABLE
  Scala_SCALAC_EXECUTABLE
  Scala_JAR_EXECUTABLE
)
