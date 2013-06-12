include(CommonModularInterface)
include_directories(${CMAKE_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${NUMPY_INCLUDE_DIRS})

GENERATE_MODULAR_TARGET(python ${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON_LIBRARIES})
SET_TARGET_PROPERTIES(_python_modular PROPERTIES OUTPUT_NAME _modshogun)

SET(PYTHON_SITE_PACKGES_INSTALL_DIR lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
IF (${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.${PYTHON_VERSION_PATCH} VERSION_GREATER "2.6.2")
	SET(PYTHON_SITE_PACKGES_INSTALL_DIR "${PYTHON_SITE_PACKGES_INSTALL_DIR}/dist-packages")
ELSE()
	SET(PYTHON_SITE_PACKGES_INSTALL_DIR "${PYTHON_SITE_PACKGES_INSTALL_DIR}/site-packages")
ENDIF()

INSTALL(TARGETS _python_modular 
		DESTINATION ${PYTHON_SITE_PACKGES_INSTALL_DIR})
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/modshogun.py 
		DESTINATION ${PYTHON_SITE_PACKGES_INSTALL_DIR})

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
