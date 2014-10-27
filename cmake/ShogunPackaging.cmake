SET(CPACK_PACKAGE_CONTACT shogun@shogun-toolbox.org)

# general cpack settings
set(CPACK_PACKAGE_NAME "shogun")
set(CPACK_PACKAGE_VENDOR "shogun")
#set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/README")
#set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/src/COPYRIGHT")
set(CPACK_PACKAGE_VERSION ${VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${SHOGUN_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${SHOGUN_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${SHOGUN_VERSION_PATCH})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Shogun Machine Learning Toolbox")
set(CPACK_PACKAGE_DESCRIPTION "Large Scale Machine Learning Toolbox
 SHOGUN - is a new machine learning toolbox with focus on large scale kernel
 methods and especially on Support Vector Machines (SVM) with focus to
 bioinformatics. It provides a generic SVM object interfacing to several
 different SVM implementations. Each of the SVMs can be combined with a variety
 of the many kernels implemented. It can deal with weighted linear combination
 of a number of sub-kernels, each of which not necessarily working on the same
 domain, where  an optimal sub-kernel weighting can be learned using Multiple
 Kernel Learning.  Apart from SVM 2-class classification and regression
 problems, a number of linear methods like Linear Discriminant Analysis (LDA),
 Linear Programming Machine (LPM), (Kernel) Perceptrons and also algorithms to
 train hidden markov models are implemented. The input feature-objects can be
 dense, sparse or strings and of type int/short/double/char and can be
 converted into different feature types. Chains of preprocessors (e.g.
 substracting the mean) can be attached to each feature object allowing for
 on-the-fly pre-processing.
 .
 SHOGUN comes in different flavours, a stand-a-lone version and also with
 interfaces to Matlab(tm), R, Octave, Readline and Python. This is the core
 library with the machine learning methods and ui helpers all interfaces are
 based on.")
SET(CPACK_STRIP_FILES ON)

# Heuristics to figure out cpack generator
set(CPACK_GENERATOR "TGZ")
if(MSVC)
	set(CPACK_GENERATOR "NSIS")
	set(CPACK_NSIS_MODIFY_PATH ON)
elseif(APPLE)
	set(CPACK_GENERATOR "PackageMaker")
	set(CPACK_OSX_PACKAGE_VERSION "${${UPPER_PROJECT_NAME}_OSX_VERSION}")
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
	if(EXISTS "/etc/issue")
		file(READ "/etc/issue" LINUX_ISSUE)
		if(LINUX_ISSUE MATCHES "(Fedora)|(SUSE)|(Mandriva)")
			set(CPACK_GENERATOR "RPM")
		elseif(LINUX_ISSUE MATCHES "(Ubuntu)|(Debian)")
			set(CPACK_GENERATOR "DEB")
		endif()
	endif()
endif()

if(CPACK_GENERATOR STREQUAL "DEB")
	# debian package settings
	set(CPACK_DEB_COMPONENT_INSTALL ON)
	set(CPACK_COMPONENTS_IGNORE_GROUPS ON)

	set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}_${CPACK_PACKAGE_VERSION}_${CMAKE_SYSTEM_PROCESSOR}")
	set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}_${CPACK_PACKAGE_VERSION}-${OPENCV_PACKAGE_ARCH_SUFFIX}")
	set(CPACK_DEBIAN_PACKAGE_SECTION "science")

	set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libstdc++6, libsnappy1, zlib1g, liblzma5, libbz2-1.0, liblzo2-2, libxml2, libhdf5-8, libnlopt0, liblapack3, libglpk36, libcurl3-gnutls, libarpack2")

	set(CPACK_COMPONENT_libraries_DISPLAY_NAME "lib${CMAKE_PROJECT_NAME}${LIBSHOGUNSO}")
	set(CPACK_COMPONENT_headers_DISPLAY_NAME "lib${CMAKE_PROJECT_NAME}-dev")
	set(CPACK_COMPONENT_python_DISPLAY_NAME "python-${CMAKE_PROJECT_NAME}")
elseif(CPACK_GENERATOR STREQUAL "RPM")
	# rpm package settings
	set(CPACK_RPM_COMPONENT_INSTALL ON)
	set(CPACK_COMPONENTS_IGNORE_GROUPS ON)

	set(CPACK_RPM_PACKAGE_NAME "shogun")
	set(CPACK_RPM_PACKAGE_REQUIRES "snappy, libxml2, NLopt, xz, lapack, glpk, libcurl, bzip2, arpack, ColPack")
	set(CPACK_PACKAGE_FILE_NAME "${CPACK_RPM_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}.${CMAKE_SYSTEM_PROCESSOR}")

	set(CPACK_COMPONENT_libraries_DISPLAY_NAME "${CMAKE_PROJECT_NAME}")
	set(CPACK_COMPONENT_headers_DISPLAY_NAME "${CMAKE_PROJECT_NAME}-devel")
	set(CPACK_COMPONENT_python_DISPLAY_NAME "python-${CMAKE_PROJECT_NAME}")
else()
	set(CPACK_COMPONENT_libraries_DISPLAY_NAME "Shogun libraries")
	set(CPACK_COMPONENT_headers_DISPLAY_NAME "C++ headers")
	set(CPACK_COMPONENT_python_DISPLAY_NAME "Python interface")
	set(CPACK_COMPONENT_r_DISPLAY_NAME "R interface")
	set(CPACK_COMPONENT_ruby_DISPLAY_NAME "Ruby interface")
	set(CPACK_COMPONENT_csharp_DISPLAY_NAME "C# Interface")
	set(CPACK_COMPONENT_java_DISPLAY_NAME "Java interface")
	set(CPACK_COMPONENT_octave_DISPLAY_NAME "OCTAVE interface")
	set(CPACK_COMPONENT_lua_DISPLAY_NAME "Lua interface")
endif()

set(CPACK_COMPONENT_libraries_DESCRIPTION "Shogun Machine Learning Toolbox")
set(CPACK_COMPONENT_libraries_GROUP "Runtime")
set(CPACK_COMPONENT_libraries_REQUIRED On)

set(CPACK_COMPONENT_headers_DESCRIPTION "Development files for Shogun Machine Learning Toolbox")
set(CPACK_COMPONENT_headers_GROUP "Development")

set(CPACK_COMPONENT_GROUP_DEVELOPMENT_EXPANDED ON)
set(CPACK_COMPONENT_GROUP_DEVELOPMENT_DESCRIPTION
   "All of the packages that's required for developing with Shogun")

set(CPACK_COMPONENT_headers_DEPENDS libraries)

set(CPACK_COMPONENT_python_DESCRIPTION "Python modular interface of Shogun")
set(CPACK_COMPONENT_python_GROUP "Runtime")
set(CPACK_COMPONENT_python_DEPENDS libraries)

set(CPACK_COMPONENT_r_DESCRIPTION "R modular interface of Shogun")
set(CPACK_COMPONENT_r_GROUP "Runtime")
set(CPACK_COMPONENT_r_DEPENDS libraries)

set(CPACK_COMPONENT_ruby_DESCRIPTION "Ruby modular interface of Shogun")
set(CPACK_COMPONENT_ruby_GROUP "Runtime")
set(CPACK_COMPONENT_ruby_DEPENDS libraries)

set(CPACK_COMPONENT_csharp_DESCRIPTION "C# modular interface of Shogun")
set(CPACK_COMPONENT_csharp_GROUP "Runtime")
set(CPACK_COMPONENT_csharp_DEPENDS libraries)

set(CPACK_COMPONENT_java_DESCRIPTION "Java modular interface of Shogun")
set(CPACK_COMPONENT_java_GROUP "Runtime")
set(CPACK_COMPONENT_java_DEPENDS libraries)

set(CPACK_COMPONENT_octave_DESCRIPTION "OCTAVE modular interface of Shogun")
set(CPACK_COMPONENT_octave_GROUP "Runtime")
set(CPACK_COMPONENT_octave_DEPENDS libraries)

set(CPACK_COMPONENT_lua_DESCRIPTION "Lua modular interface of Shogun")
set(CPACK_COMPONENT_lua_GROUP "Runtime")
set(CPACK_COMPONENT_lua_DEPENDS libraries)

set(CPACK_COMPONENT_cmdline_DISPLAY_NAME "Command Line Interface")
set(CPACK_COMPONENT_cmdline_DESCRIPTION "Command Line interface of Shogun")
set(CPACK_COMPONENT_cmdline_GROUP "Runtime")
set(CPACK_COMPONENT_cmdline_DEPENDS libraries)

INCLUDE(CPack)
