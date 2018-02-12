# find python library for the found python interpreter
# i.e. versions should match!
FIND_PACKAGE(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)
FIND_PACKAGE(NumPy 1.7 REQUIRED)

SET(HAVE_PYTHON 1)

#custom swig flags for python interface
IF(${PYTHON_VERSION_MAJOR} VERSION_EQUAL 3)
  SET(TARGET_SWIGFLAGS "-builtin\;-modern\;-modernargs\;-threads\;-py3")
  SET(PYTHON3 1)
ELSE()
  SET(TARGET_SWIGFLAGS "-builtin\;-modern\;-modernargs\;-threads")
ENDIF()

# SWIG-generated Python-wrappers fail to build
# for Python >=3.5 with SWIG < 3.0.8.  Make CMake
# emit an error and fail to configure.
IF ((NOT "${PYTHON_VERSION_STRING}" VERSION_LESS "3.5") AND
  ("${SWIG_VERSION}" VERSION_LESS "3.0.8"))
  MESSAGE(FATAL_ERROR
    "Building Python interface for Python >= 3.5 requires SWIG >= 3.0.8.")
ENDIF ((NOT "${PYTHON_VERSION_STRING}" VERSION_LESS "3.5") AND
  ("${SWIG_VERSION}" VERSION_LESS "3.0.8"))

include(ShogunInterfaces)
include_directories(${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_DIRS})

EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('LDSHARED'))"
                RESULT_VARIABLE PYTHON_CVPY_PROCESS
                OUTPUT_VARIABLE PYTHON_LDSHARED
                OUTPUT_STRIP_TRAILING_WHITESPACE)

IF ("${PYTHON_LDSHARED}" MATCHES "dynamic_lookup")
    MESSAGE("Using static linking for Python")
    SET(PYTHON_LINK_LIBRARIES "NO")
ELSE()
    SET(PYTHON_LINK_LIBRARIES "${PYTHON_LIBRARIES}")
ENDIF()

GENERATE_INTERFACE_TARGET(python "${CMAKE_CURRENT_SOURCE_DIR}" "${PYTHON_LINK_LIBRARIES}")

IF (NOT PYTHON_LINK_LIBRARIES)
    IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        SET_TARGET_PROPERTIES(${SWIG_MODULE_interface_python_REAL_NAME} PROPERTIES
                              LINK_FLAGS "-Wl,-flat_namespace,-undefined,dynamic_lookup")
    ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        SET_TARGET_PROPERTIES(${SWIG_MODULE_interface_python_REAL_NAME} PROPERTIES
                              LINK_FLAGS "-undefined dynamic_lookup")
    ELSE()
        MESSAGE(FATAL_ERROR "No libraries to link with the interface are provided but don't know how to handle dynamic lookup")
    ENDIF()
ENDIF()

# pacakges directory
if(CMAKE_HOST_UNIX)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import *; print(get_python_lib())"
                  RESULT_VARIABLE PYTHON_CVPY_PROCESS
                  OUTPUT_VARIABLE PYTHON_STD_PACKAGES_PATH
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if("${PYTHON_STD_PACKAGES_PATH}" MATCHES "site-packages")
    set(_PYTHON_PACKAGES_PATH "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages")
  else() #debian based assumed, install to the dist-packages.
    set(_PYTHON_PACKAGES_PATH "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages")
  endif()
  if(EXISTS "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}/${PYTHON_PACKAGES_PATH}")
    set(_PYTHON_PACKAGES_PATH "lib${LIB_SUFFIX}/${_PYTHON_PACKAGES_PATH}")
  else()
    set(_PYTHON_PACKAGES_PATH "lib/${_PYTHON_PACKAGES_PATH}")
  endif()
elseif(CMAKE_HOST_WIN32)
  get_filename_component(PYTHON_PATH "${PYTHON_EXECUTABLE}" PATH)
  file(TO_CMAKE_PATH "${PYTHON_PATH}" PYTHON_PATH)
  if(NOT EXISTS "${PYTHON_PATH}/Lib/site-packages")
    unset(PYTHON_PATH)
    get_filename_component(PYTHON_PATH "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}\\InstallPath]" ABSOLUTE)
    if(NOT PYTHON_PATH)
       get_filename_component(PYTHON_PATH "[HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}\\InstallPath]" ABSOLUTE)
    endif()
    file(TO_CMAKE_PATH "${PYTHON_PATH}" PYTHON_PATH)
  endif()
  set(_PYTHON_PACKAGES_PATH "${PYTHON_PATH}/Lib/site-packages")
endif()
SET(PYTHON_PACKAGES_PATH "${_PYTHON_PACKAGES_PATH}" CACHE PATH "Where to install the python packages.")

INSTALL(TARGETS _interface_python
		DESTINATION ${PYTHON_PACKAGES_PATH}
    COMPONENT python)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/shogun.py
		DESTINATION ${PYTHON_PACKAGES_PATH}
    COMPONENT python
)
