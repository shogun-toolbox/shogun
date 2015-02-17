FILE(GLOB R_EXAMPLES *.R)
FOREACH(EXAMPLE ${R_EXAMPLES})
	get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)
	add_test(NAME generated_r_modular-${EXAMPLE_NAME}
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			COMMAND ${R_EXECUTABLE} --no-restore --no-save --no-readline --slave -f ${EXAMPLE})
ENDFOREACH()

# For now, the generated examples are not installed in the R library
# When the switch to the example generation is complete, the following lines may be uncommented 
#install(FILES ${R_EXAMPLES}
#	DESTINATION ${R_COMPONENT_LIB_PATH}/shogun/demo
#	COMPONENT r)