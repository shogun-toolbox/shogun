##############################################################################
# @file  FindMOSEK.cmake
# @brief Find MOSEK (http://www.mosek.com) package.
#
# @par Input variables:
# <table border="0">
#   <tr>
#     @tp @b MOSEK_DIR @endtp
#     <td>The MOSEK package files are searched under the specified root
#         directory. If they are not found there, the default search paths
#         are considered. This variable can also be set as environment variable.</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_NO_OMP @endtp
#     <td>Whether to use the link libraries build without OpenMP, i.e.,
#         multi-threading, enabled. By default, the multi-threaded libraries
#         are used.</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_FIND_COMPONENTS @endtp
#     <td>The @c COMPONENTS argument(s) of the find_package() command can
#         be used to also look for optional MOSEK components.
#         Valid component values are "mex", "jar", and "pypkg".</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_TOOLS_SUFFIX @endtp
#     <td>Platform specific path suffix for tools, i.e., "tools/platform/linux64x86"
#         on 64-bit Linux systems. If not specified, this module determines the
#         right suffix depending on the CMake system variables.</td>
#   </tr>
#   <tr>
#     @tp @b MATLAB_RELEASE @endtp
#     <td>Release of MATLAB installation. Set to the 'Release' return value of
#         the "ver ('MATLAB')" command of MATLAB without brackets. If this
#         variable is not set and the basis_get_matlab_release() command is
#         available, it is invoked to determine the release version automatically.
#         Otherwise, an error is raised if the "mex" component is searched.</td>
#   </tr>
#   <tr>
#     @tp @b MEX_EXT @endtp
#     <td>The extension of MEX-files. If this variable is not set and the
#         basis_mexext() command is available, it is invoked to determine the
#         extension automatically. Otherwise, the MEX extension defaults to "mexa64".</td>
#   </tr>
#   <tr>
#     @tp @b PYTHON_VERSION_MAJOR @endtp
#     <td>Major version of Python installation as determined by FindPythonInterp.cmake module.</td>
#   </tr>
# </table>
#
# @par Output variables:
# <table border="0">
#   <tr>
#     @tp @b MOSEK_FOUND @endtp
#     <td>Whether the package was found and the following CMake variables are valid.</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_<component>_FOUND @endtp
#     <td>Whether the component requested by @c MOSEK_FIND_COMPONENTS was found.</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_INCLUDE_DIR @endtp
#     <td>Package include directories.</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_INCLUDES @endtp
#     <td>Include directories including prerequisite libraries (non-cached).</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_LIBRARY @endtp
#     <td>Package libraries.</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_LIBRARIES @endtp
#     <td>Package libraries and prerequisite libraries (non-cached).</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_mosekopt_MEX @endtp
#     <td>Package mosekopt MEX-file.</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_MEX_FILES @endtp
#     <td>List of MEX-files (non-cached).</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_mosek_JAR @endtp
#     <td>Package mosek Java library (.jar file).</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_CLASSPATH @endtp
#     <td>List of Java package libraries and prerequisite libraries (non-cached).</td>
#   </tr>
#   <tr>
#     @tp @b MOSEK_PYTHONPATH @endtp
#     <td>Path to Python modules of this package.</td>
#   </tr>
# </table>
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.<br />
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
#
# @ingroup CMakeFindModules
##############################################################################

# ----------------------------------------------------------------------------
# optional components to look for
set (_MOSEK_OPTIONAL_COMPONENTS mex jar pypkg)
foreach (CMP IN LISTS _MOSEK_OPTIONAL_COMPONENTS)
  set (MOSEK_FIND_${CMP} FALSE)
endforeach ()
foreach (CMP IN LISTS MOSEK_FIND_COMPONENTS)
  if (NOT CMP MATCHES "^(mex|jar|pypkg)$")
    message (FATAL_ERROR "Invalid MOSEK component: ${CMP}")
  endif ()
  set (MOSEK_FIND_${CMP} TRUE)
endforeach ()

# ----------------------------------------------------------------------------
# remember CMAKE_FIND_LIBRARY_SUFFIXES to be able to restore it
set (_MOSEK_CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")

# ----------------------------------------------------------------------------
# versions - library suffixes

# known MOSEK versions, all entries have to be specified in descending order!
set (_MOSEK_VERSIONS_MAJOR 6 7)
set (_MOSEK6_VERSIONS      6.0)
set (_MOSEK7_VERSIONS      7.0)

# get a full list of particular versions (<major>.<minor>) to look for
set (_MOSEK_FIND_VERSIONS)
if (MOSEK_FIND_VERSION)
  if (MOSEK_FIND_VERSION MATCHES "^([0-9]+\\.[0-9]+)(\\.[0-9]+.*)?$")
    set (_MOSEK_FIND_VERSION_MAJOR_MINOR "${CMAKE_MATCH_1}")
    list (APPEND _MOSEK_FIND_VERSIONS ${_MOSEK_FIND_VERSION_MAJOR_MINOR})
    if (NOT MOSEK_FIND_VERSION_EXACT)
      string (REGEX REPLACE "^([0-9]+).*" "\\1" _MOSEK_FIND_VERSION_MAJOR "${CMAKE_MATCH_1}")
      foreach (_MOSEK_VERSION IN LISTS _MOSEK${_MOSEK_FIND_VERSION_MAJOR}_VERSIONS)
        if (NOT _MOSEK_VERSION VERSION_LESS _MOSEK_FIND_VERSION_MAJOR_MINOR)
          list (APPEND _MOSEK_FIND_VERSIONS ${_MOSEK_VERSION})
        endif()
      endforeach()
      unset (_MOSEK_FIND_VERSION_MAJOR)
    endif ()
    unset (_MOSEK_FIND_VERSION_MAJOR_MINOR)
  else ()
    if (APPLE)
      list (APPEND _MOSEK_LIBRARY_SUFFIXES .dylib.${_MOSEK_FIND_VERSION_MAJOR_MINOR})
    elseif (UNIX)
      list (APPEND _MOSEK_LIBRARY_SUFFIXES .so.${_MOSEK_FIND_VERSION_MAJOR_MINOR})
    endif ()
    set (_MOSEK_FIND_OTHER_VERSIONS ${_MOSEK${MOSEK_FIND_VERSION}_VERSIONS})
  endif ()
else ()
  foreach (_MOSEK_VERSION_MAJOR IN LISTS _MOSEK_VERSIONS_MAJOR)
    list (APPEND _MOSEK_FIND_VERSIONS ${_MOSEK${_MOSEK_VERSION_MAJOR}_VERSIONS})
  endforeach ()
endif ()

# ----------------------------------------------------------------------------
# initialize search
if (NOT MOSEK_DIR)
  set (MOSEK_DIR "$ENV{MOSEK_DIR}" CACHE PATH "Installation prefix for MOSEK." FORCE)
endif ()

# MATLAB components
if (MOSEK_FIND_mex)
  # MATLAB version
  if (NOT MATLAB_RELEASE)
    if (COMMAND basis_get_matlab_release)
      basis_get_matlab_release (MATLAB_RELEASE)
      if (NOT MATLAB_RELEASE)
        message (FATAL_ERROR "Failed to determine release version of MATLAB installation."
                             " This information is required to be able to find the right MOSEK MEX-files."
                             " Alternatively, set MATLAB_RELEASE manually and try again.")
      endif ()
    else ()
      message (FATAL_ERROR "MATLAB_RELEASE variable not set."
                           " This information is required to be able to find the right MOSEK MEX-files."
                           " Set MATLAB_RELEASE to the correct MATLAB release version, e.g., R2009b,"
                           " and try again.")
    endif ()
  endif ()
  string (TOLOWER "${MATLAB_RELEASE}" MATLAB_RELEASE_L)
  # search path for MOSEK MATLAB toolbox
  if (NOT MOSEK_TOOLBOX_SUFFIX)
    if (MOSEK_DIR)
      file (
        GLOB_RECURSE
          MOSEK_TOOLBOX_SUFFIXES
        RELATIVE "${MOSEK_DIR}"
        "${MOSEK_DIR}/toolbox/*/*.mex*"
      )
      set (MOSEK_TOOLBOX_VERSIONS)
      foreach (MOSEK_MEX_FILE IN LISTS MOSEK_TOOLBOX_SUFFIXES)
        get_filename_component (MOSEK_TOOLBOX_SUFFIX  "${MOSEK_MEX_FILE}" PATH)
        get_filename_component (MOSEK_TOOLBOX_VERSION "${MOSEK_TOOLBOX_SUFFIX}" NAME)
        list (APPEND MOSEK_TOOLBOX_VERSIONS "${MOSEK_TOOLBOX_VERSION}")
        set (MOSEK_TOOLBOX_SUFFIX)
      endforeach ()
      list (SORT MOSEK_TOOLBOX_VERSIONS)
      list (REVERSE MOSEK_TOOLBOX_VERSIONS)
      string (REGEX MATCH "[0-9][0-9]*" MATLAB_RELEASE_YEAR "${MATLAB_RELEASE}")
      foreach (MOSEK_TOOLBOX_VERSION IN LISTS MOSEK_TOOLBOX_VERSIONS)
        if (MOSEK_TOOLBOX_VERSION MATCHES "[rR]([0-9][0-9]*)[ab]")
          if (CMAKE_MATCH_1 EQUAL MATLAB_RELEASE_VERSION OR
              CMAKE_MATCH_1 LESS  MATLAB_RELEASE_VERSION)
            set (MATLAB_TOOLBOX_SUFFIX "toolbox/${MOSEK_TOOLBOX_VERSION}")
            break ()
          endif ()
        endif ()
      endforeach ()
    endif ()
    if (NOT MOSEK_TOOLBOX_SUFFIX)
      set (MOSEK_TOOLBOX_SUFFIX "toolbox/${MATLAB_RELEASE_L}")
    endif ()
  endif ()
  # extension of MEX-files
  if (NOT MEX_EXT)
    if (COMMAND basis_mexext)
      basis_mexext ()
    else ()
      set (MEX_EXT "mexa64")
    endif ()
  endif ()
endif ()

# Python components
if (MOSEK_FIND_pypkg)
  if (NOT PYTHON_VERSION_MAJOR)
    message (FATAL_ERROR "Python interpreter not found or not added as dependency before MOSEK. "
                         "The information about the Python version is required to be able to find "
                         "the right MOSEK Python modules. Therefore, add the PythonInterp package "
                         "as dependency to BasisProjects.cmake before the entry of MOSEK. "
                         "The FindPythonInterp.cmake module will determine the version of the "
                         "Python installation. Alternatively, set PYTHON_VERSION_MAJOR manually.")
  endif ()
endif ()

# library name
set (MOSEK_LIBRARY_NAME "mosek")
if (MOSEK_NO_OMP)
  set (MOSEK_LIBRARY_NAME "${MOSEK_LIBRARY_NAME}noomp")
endif ()
if (UNIX)
  if (NOT CMAKE_SIZE_OF_VOID_P EQUAL 4)
    set (MOSEK_LIBRARY_NAME "${MOSEK_LIBRARY_NAME}64")
  endif ()
endif ()

# append/set library version suffixes
if (WIN32)
  if (_MOSEK_FIND_VERSIONS)
    foreach (_MOSEK_VERSION IN LISTS _MOSEK_FIND_VERSIONS)
      string (REPLACE "." "_" _MOSEK_VERSION "${_MOSEK_VERSION}")
      list (APPEND MOSEK_LIBRARY_NAMES "${MOSEK_LIBRARY_NAME}${_MOSEK_VERSION}")
    endforeach ()
  else ()
    set (MOSEK_LIBRARY_NAMES "${MOSEK_LIBRARY_NAME}")
  endif ()
else ()
  set (MOSEK_LIBRARY_NAMES "${MOSEK_LIBRARY_NAME}")
  if (_MOSEK_FIND_VERSIONS)
    set (CMAKE_FIND_LIBRARY_SUFFIXES)
    foreach (_MOSEK_VERSION IN LISTS _MOSEK_FIND_VERSIONS)
      if (APPLE)
        list (APPEND CMAKE_FIND_LIBRARY_SUFFIXES .${_MOSEK_VERSION}.dylib)
      else ()
        list (APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.${_MOSEK_VERSION})
      endif ()
    endforeach ()
    if (NOT MOSEK_FIND_VERSION)
      if (APPLE)
        list (APPEND CMAKE_FIND_LIBRARY_SUFFIXES .dylib)
      else ()
        list (APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so)
      endif ()
    endif ()
  endif ()
endif ()

# search path for MOSEK tools
if (NOT MOSEK_TOOLS_SUFFIX)
  set (MOSEK_TOOLS_SUFFIX "tools/platform/")
  if (WIN32)
    set (MOSEK_TOOLS_SUFFIX "${MOSEK_TOOLS_SUFFIX}win")
  elseif (APPLE)
    set (MOSEK_TOOLS_SUFFIX "${MOSEK_TOOLS_SUFFIX}osx")
  else ()
    set (MOSEK_TOOLS_SUFFIX "${MOSEK_TOOLS_SUFFIX}linux")
  endif ()
  if (CMAKE_SIZE_OF_VOID_P EQUAL 4)
    set (MOSEK_TOOLS_SUFFIX "${MOSEK_TOOLS_SUFFIX}32")
  else ()
    set (MOSEK_TOOLS_SUFFIX "${MOSEK_TOOLS_SUFFIX}64")
  endif ()
  set (MOSEK_TOOLS_SUFFIX "${MOSEK_TOOLS_SUFFIX}x86")
endif ()

unset (_MOSEK_FIND_VERSIONS)

#-------------------------------------------------------------
# find include files and library
foreach (_MOSEK_I IN ITEMS 1 2) # try twice in case MOSEK_DIR
                                # was not set, but known in
                                # second iteration

  # find files
  if (MOSEK_DIR)

    find_path (
      MOSEK_INCLUDE_DIR
        NAMES         mosek.h
        HINTS         "${MOSEK_DIR}"
        PATH_SUFFIXES "${MOSEK_TOOLS_SUFFIX}/h"
        DOC           "Include directory for MOSEK libraries."
        NO_DEFAULT_PATH
    )

    find_library (
      MOSEK_LIBRARY
        NAMES         ${MOSEK_LIBRARY_NAMES}
        HINTS         "${MOSEK_DIR}"
        PATH_SUFFIXES "${MOSEK_TOOLS_SUFFIX}/bin"
        DOC           "MOSEK link library."
        NO_DEFAULT_PATH
    )

  else ()

    find_path (
      MOSEK_INCLUDE_DIR
        NAMES mosek.h
        HINTS ENV C_INCLUDE_PATH ENV CXX_INCLUDE_PATH
        DOC   "Include directory for MOSEK libraries."
    )

    find_library (
      MOSEK_LIBRARY
        NAMES ${MOSEK_LIBRARY_NAMES}
        HINTS ENV LD_LIBRARY_PATH
        DOC   "MOSEK link library."
    )

  endif ()

  # derive MOSEK_DIR
  if (NOT MOSEK_DIR)
    if (COMMAND basis_sanitize_for_regex)
      basis_sanitize_for_regex (_MOSEK_TOOLS_SUFFIX_RE "${MOSEK_TOOLS_SUFFIX}")
    else ()
      set (_MOSEK_TOOLS_SUFFIX_RE "${MOSEK_TOOLS_SUFFIX}")
    endif ()
    if (MOSEK_INCLUDE_DIR)
      string (REGEX REPLACE "${_MOSEK_TOOLS_SUFFIX_RE}/.*$" "" _MOSEK_DIR "${MOSEK_INCLUDE_DIR}")
      set (MOSEK_DIR "${_MOSEK_DIR}" CACHE PATH "Installation prefix for MOSEK." FORCE)
    elseif (MOSEK_LIBRARY)
      string (REGEX REPLACE "${_MOSEK_TOOLS_SUFFIX_RE}/.*$" "" _MOSEK_DIR "${MOSEK_LIBRARY}")
      set (MOSEK_DIR "${_MOSEK_DIR}" CACHE PATH "Installation prefix for MOSEK." FORCE)
    endif ()
    unset (_MOSEK_TOOLS_SUFFIX_RE)
    unset (_MOSEK_DIR)
  endif ()

  # skip second iteration if both found already
  if (MOSEL_INCLUDE_DIR AND MOSEK_LIBRARY)
    break ()
  endif ()
endforeach ()

mark_as_advanced (MOSEK_INCLUDE_DIR)
mark_as_advanced (MOSEK_LIBRARY)

# MATLAB components
if (MOSEK_FIND_mex)
  if (MOSEK_DIR)

    find_file (
      MOSEK_mosekopt_MEX
        NAMES         mosekopt.${MEX_EXT}
        HINTS         "${MOSEK_DIR}"
        PATH_SUFFIXES "${MOSEK_TOOLBOX_SUFFIX}"
        DOC           "The mosekopt MEX-file of the MOSEK package."
        NO_DEFAULT_PATH
    )

  else ()

    find_file (
      MOSEK_mosekopt_MEX
        NAMES         mosekopt.${MEX_EXT}
        PATH_SUFFIXES "${MOSEK_TOOLBOX_SUFFIX}"
        DOC           "The mosekopt MEX-file of the MOSEK package."
    )

  endif ()

  if (MOSEK_mosekopt_MEX)
    set (MOSEK_MEX_FILES "${MOSEK_mosekopt_MEX}")
  endif ()

  mark_as_advanced (MOSEK_mosekopt_MEX)
endif ()

# Java components
if (MOSEK_FIND_jar)
  if (MOSEK_DIR)

    find_file (
      MOSEK_mosek_JAR
        NAMES         mosek.jar
        HINTS         "${MOSEK_DIR}"
        PATH_SUFFIXES "${MOSEK_TOOLS_SUFFIX}/bin"
        DOC           "The Java library (.jar file) of the MOSEK package."
        NO_DEFAULT_PATH
    )

  else ()

    find_file (
      MOSEK_mosek_JAR
        NAMES mosek.jar
        HINTS ENV CLASSPATH
        DOC   "The Java library (.jar file) of the MOSEK package."
    )

  endif ()

  if (MOSEK_mosek_JAR)
    set (MOSEK_CLASSPATH "${MOSEK_mosek_JAR}")
  endif ()

  mark_as_advanced (MOSEK_mosek_JAR)
endif ()

# Python components
if (MOSEK_FIND_pypkg)
  if (MOSEK_DIR)

    find_path (
      MOSEK_PYTHONPATH
        NAMES "mosek/array.py"
        HINTS ENV PYTHONPATH
        DOC   "Path to MOSEK Python module."
    )

  else ()

    find_path (
      MOSEK_PYTHONPATH
        NAMES "mosek/array.py"
        HINTS "${MOSEK_DIR}/${MOSEK_PATH_SUFFIX}/python/${PYTHON_VERSION_MAJOR}"
        DOC   "Path to MOSEK Python module."
        NO_DEFAULT_PATH
    )

  endif ()

  mark_as_advanced (MOSEK_PYTHONPATH)
endif ()

# ----------------------------------------------------------------------------
# prerequisite libraries
set (MOSEK_INCLUDES  "${MOSEK_INCLUDE_DIR}")
set (MOSEK_LIBRARIES "${MOSEK_LIBRARY}")

# ----------------------------------------------------------------------------
# aliases / backwards compatibility
set (MOSEK_INCLUDE_DIRS "${MOSEK_INCLUDES}")

# ----------------------------------------------------------------------------
# debugging
if (BASIS_DEBUG AND COMMAND basis_dump_variables)
  basis_dump_variables ("${CMAKE_CURRENT_BINARY_DIR}/FindMOSEKVariables.cmake")
endif ()

# ----------------------------------------------------------------------------
# handle the QUIETLY and REQUIRED arguments and set *_FOUND to TRUE
# if all listed variables are found or TRUE
include (FindPackageHandleStandardArgs)

set (MOSEK_REQUIRED_VARS
  MOSEK_DIR
  MOSEK_INCLUDE_DIR
  MOSEK_LIBRARY
)

if (MOSEK_FIND_mex)
  list (APPEND MOSEK_REQUIRED_VARS MOSEK_mosekopt_MEX)
  if (MOSEK_mosekopt_MEX)
    set (MOSEK_mex_FOUND TRUE)
  else ()
    set (MOSEK_mex_FOUND FALSE)
  endif ()
endif ()
if (MOSEK_FIND_jar)
  list (APPEND MOSEK_REQUIRED_VARS MOSEK_mosek_JAR)
  if (MOSEK_mosek_JAR)
    set (MOSEK_jar_FOUND TRUE)
  else ()
    set (MOSEK_jar_FOUND FALSE)
  endif ()
endif ()
if (MOSEK_FIND_pypkg)
  list (APPEND MOSEK_REQUIRED_VARS MOSEK_PYTHONPATH)
  if (MOSEK_PYTHONPATH)
    set (MOSEK_pypkg_FOUND TRUE)
  else ()
    set (MOSEK_pypkg_FOUND FALSE)
  endif ()
endif ()

find_package_handle_standard_args (
  MOSEK
# MESSAGE
    DEFAULT_MSG
# VARIABLES
    ${MOSEK_REQUIRED_VARS}
)

set (CMAKE_FIND_LIBRARY_SUFFIXES "${_MOSEK_CMAKE_FIND_LIBRARY_SUFFIXES}")
unset (_MOSEK_CMAKE_FIND_LIBRARY_SUFFIXES)

foreach (CMP IN LISTS _MOSEK_OPTIONAL_COMPONENTS)
  unset (MOSEK_FIND_${CMP})
endforeach ()
