INCLUDE(PythonEnvironment)
GET_PYTHON_ENV()

FILE(GLOB PYTHON_EXAMPLES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

IF (NOT USE_GPL_SHOGUN)
	LIST(REMOVE_ITEM PYTHON_EXAMPLES
	"classifier_svmlin.py"
	"kernel_distantsegments.py"
	"variational_classifier.py"
	"evaluation_cross_validation_multiclass_storage.py"
	"classifier_larank.py"
	"classifier_svmsgd.py"
	)
ENDIF()

FOREACH(EXAMPLE_PY ${PYTHON_EXAMPLES})
	STRING(REGEX REPLACE ".py\$" "" EXAMPLE "${EXAMPLE_PY}")
	
	add_test(NAME python_legacy-${EXAMPLE}
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			COMMAND ${PYTHON_EXECUTABLE} ${EXAMPLE_PY})
	set_property(TEST python_legacy-${EXAMPLE} PROPERTY
				ENVIRONMENT "${PYTHON_ENV_VARS}")
ENDFOREACH()
