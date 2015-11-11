Macro(IsAnyTrue LIST RESULT)
	Set(${RESULT} "FALSE")
	ForEach(Element ${LIST})
		If(${Element})
			Set(${RESULT} "TRUE")
		EndIf()
	EndForEach(Element)
EndMacro()

MACRO(MergeCFLAGS)
	SET(MERGED_C_FLAGS ${CMAKE_C_FLAGS})
	SET(MERGED_CXX_FLAGS ${CMAKE_CXX_FLAGS})

	IF (CMAKE_BUILD_TYPE MATCHES Release)
		SET(MERGED_C_FLAGS "${MERGED_C_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
		SET(MERGED_CXX_FLAGS "${MERGED_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
	ELSEIF (CMAKE_BUILD_TYPE MATCHES Distribution)
		SET(MERGED_C_FLAGS "${MERGED_C_FLAGS} ${CMAKE_C_FLAGS_DISTRIBUTION}")
		SET(MERGED_CXX_FLAGS "${MERGED_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DISTRIBUTION}")
	ELSEIF (CMAKE_BUILD_TYPE MATCHES Debug)
		SET(MERGED_C_FLAGS "${MERGED_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
		SET(MERGED_CXX_FLAGS "${MERGED_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
	ENDIF()
ENDMACRO()

MACRO(GetCompilers)
	IF(CCACHE_FOUND AND ENABLE_CCACHE)
		SET(C_COMPILER ${CMAKE_C_COMPILER_ARG1})
		SET(CXX_COMPILER ${CMAKE_CXX_COMPILER_ARG1})
	ELSE()
		SET(C_COMPILER ${CMAKE_C_COMPILER})
		SET(CXX_COMPILER ${CMAKE_CXX_COMPILER})
	ENDIF()
ENDMACRO()

MACRO(PrintInterfaceStatus INTERFACE_NAME INTERFACE_FLAG)
	IF ( ${INTERFACE_FLAG} )
		message(STATUS "  ${INTERFACE_NAME} is ON")
	ELSE()
		STRING(LENGTH ${INTERFACE_NAME} IFACE_NAME_LENGTH)
		IF (IFACE_NAME_LENGTH LESS 10)
			message(STATUS "  ${INTERFACE_NAME} is OFF \t\t - enable with -D${INTERFACE_FLAG}=ON")
		ELSE ()
			message(STATUS "  ${INTERFACE_NAME} is OFF \t - enable with -D${INTERFACE_FLAG}=ON")
		ENDIF ()
	ENDIF()
ENDMACRO()

# based on compiz_discover_tests
function (shogun_discover_tests EXECUTABLE)

        add_dependencies (${EXECUTABLE} discover_gtest_tests)

        add_custom_command (TARGET ${EXECUTABLE}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -D UNIT_TEST_CMD=${CMAKE_BINARY_DIR}/bin/${EXECUTABLE}
                     -D DISCOVER_CMD=${CMAKE_BINARY_DIR}/bin/discover_gtest_tests
                     -D WORKING_DIR=${CMAKE_CURRENT_BINARY_DIR}
                     -P ${CMAKE_MODULE_PATH}/discover_unit_tests.cmake
            COMMENT "Discovering Tests in ${EXECUTABLE}"
            DEPENDS
            VERBATIM)
endfunction ()

MACRO(AddMetaIntegrationTest META_TARGET CONDITION)
    IF (${CONDITION})
        add_test(NAME integration_meta_${META_TARGET}-${NAME_WITH_DIR}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        COMMAND meta_example_integration_tester ${REL_DIR} ${NAME}.dat ${META_TARGET} generated_results reference_results)
                    set_tests_properties(
                        integration_meta_${META_TARGET}-${NAME_WITH_DIR}
	                        PROPERTIES
	                        DEPENDS generated_${META_TARGET}-${NAME_WITH_DIR}
                    )
    ENDIF()
ENDMACRO()

function(PrintLine)
	message(STATUS "===================================================================================================================")
endfunction()

function(PrintStatus MSG)
	message(STATUS "${MSG}")
endfunction()
