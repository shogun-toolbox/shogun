execute_process(COMMAND ${EXECUTABLE} ${FILE} ${ARGS} RESULT_VARIABLE RESULT)
if (RESULT)
	if (GDB_COMMAND AND GDB_SCRIPT)
		execute_process(COMMAND ${GDB_COMMAND} 
			--command=${GDB_SCRIPT} 
			--args ${EXECUTABLE} ${FILE} ${ARGS} 
			RESULT_VARIABLE GDB_RESULT)
	endif()
endif()
