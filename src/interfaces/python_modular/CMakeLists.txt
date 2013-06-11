include_directories(${CMAKE_SOURCE_DIR}/src)

# create symlinks to ../modular/* files
set(modular_files)

FILE(GLOB_RECURSE MODULAR_FILES ${COMMON_MODULAR_SRC_DIR}/*.i)
FILE(GLOB_RECURSE PYTHON_MODULAR_FILES *.i)
LIST(APPEND MODULAR_FILES ${PYTHON_MODULAR_FILES})
FOREACH(file ${MODULAR_FILES})
	STRING(REGEX REPLACE ".*/(.*)$" "\\1" fname "${file}")
	list(APPEND modular_files ${fname})
	ADD_CUSTOM_COMMAND(OUTPUT ${fname}
		COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${file} ${fname}
	)
ENDFOREACH()

ADD_CUSTOM_TARGET(modular_interface 
	DEPENDS ${modular_files} 
	COMMENT "") 

INCLUDE(${SWIG_USE_FILE})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${NUMPY_INCLUDE_DIRS})

SET_SOURCE_FILES_PROPERTIES(modshogun.i PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(modshogun.i PROPERTIES SWIG_FLAGS ${TARGET_SWIGFLAGS})
SWIG_ADD_MODULE(modshogun python modshogun.i sg_print_functions.cpp)
SWIG_LINK_LIBRARIES(modshogun ${PYTHON_LIBRARIES} shogun)
ADD_DEPENDENCIES(_modshogun modular_interface)

IF(DOXYGEN_FOUND)
	configure_file(${COMMON_MODULAR_SRC_DIR}/modshogun.doxy.in modshogun.doxy)

	ADD_CUSTOM_COMMAND( 
    	OUTPUT    modshogun
    	COMMAND   ${DOXYGEN_EXECUTABLE}
    	ARGS	  modshogun.doxy
    	COMMENT   "Generating doxygen doc"
	)

	ADD_CUSTOM_COMMAND( 
    	OUTPUT    modshogun_doxygen.i
    	COMMAND   ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/src/.doxy2swig.py
    	ARGS	  --quiet --no-function-definition modshogun/doxygen_xml/index.xml modshogun_doxygen.i
    	DEPENDS   modshogun
	)
	ADD_CUSTOM_TARGET(doxy2swig DEPENDS modshogun_doxygen.i)
	ADD_DEPENDENCIES(_modshogun doxy2swig)
ELSE()

ENDIF()

SET(PYTHON_SITE_PACKGES_INSTALL_DIR lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
IF (${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.${PYTHON_VERSION_PATCH} VERSION_GREATER "2.6.2")
	SET(PYTHON_SITE_PACKGES_INSTALL_DIR "${PYTHON_SITE_PACKGES_INSTALL_DIR}/dist-packages")
ELSE()
	SET(PYTHON_SITE_PACKGES_INSTALL_DIR "${PYTHON_SITE_PACKGES_INSTALL_DIR}/site-packages")
ENDIF()

INSTALL(TARGETS _modshogun DESTINATION ${PYTHON_SITE_PACKGES_INSTALL_DIR})
INSTALL(FILES modshogun.py DESTINATION ${PYTHON_SITE_PACKGES_INSTALL_DIR})

# create the __init.py__ files for modules and the main
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py.in "from modshogun import *")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/main_init.py.in "__all__= [\n")
configure_file(${CMAKE_CURRENT_BINARY_DIR}/__init__.py.in __init__.py COPYONLY)
SET(MODULES Kernel Distance Features Classifier Regression Converter
	Loss Clustering Evaluation IO Library Mathematics
	ModelSelection Preprocessor Structure Distribution Statistics Latent)

FOREACH(module ${MODULES})
	INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py 
		DESTINATION ${PYTHON_SITE_PACKGES_INSTALL_DIR}/shogun/${module})
	file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/main_init.py.in "\"${module}\",\n")
ENDFOREACH()

file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/main_init.py.in "]")
configure_file(${CMAKE_CURRENT_BINARY_DIR}/main_init.py.in main_init.py COPYONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/main_init.py
	DESTINATION ${PYTHON_SITE_PACKGES_INSTALL_DIR}/shogun
	RENAME __init__.py)
