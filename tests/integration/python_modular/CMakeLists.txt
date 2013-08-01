INCLUDE(PythonEnvironment)
GET_PYTHON_ENV()

FOREACH(TEST_SRC ${INTEGRATION_TESTS})
	STRING(REGEX REPLACE ".*/(.*).m" "\\1" TEST_NAME ${TEST_SRC})
	add_test(NAME integration-python_modular-${TEST_NAME}
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			COMMAND ${PYTHON_EXECUTABLE} test_one.py ${TEST_SRC})
	set_property(TEST integration-python_modular-${TEST_NAME} PROPERTY
				ENVIRONMENT "${PYTHON_ENV_VARS}")
ENDFOREACH()

add_test(NAME integration-python_modular-tester
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMAND ${PYTHON_EXECUTABLE} tester.py)
set_property(TEST integration-python_modular-tester PROPERTY
			ENVIRONMENT "${PYTHON_ENV_VARS}")
