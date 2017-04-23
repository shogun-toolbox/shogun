execute_process(
    COMMAND ${UNIT_TEST_CMD} --gtest_list_tests
    COMMAND ${DISCOVER_CMD} ${UNIT_TEST_CMD}
    WORKING_DIRECTORY ${WORKING_DIR}
)
