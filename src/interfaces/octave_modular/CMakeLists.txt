include(CommonModularInterface)
include_directories(${OCTAVE_INCLUDE_DIRS})

GENERATE_MODULAR_TARGET(octave ${CMAKE_CURRENT_SOURCE_DIR} ${OCTAVE_LIBRARIES})
set_target_properties(octave_modular PROPERTIES PREFIX "")
set_target_properties(octave_modular PROPERTIES SUFFIX ".oct")

SET(OCTAVE_MODULAR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
install (TARGETS octave_modular
    DESTINATION ${OCTAVE_OCT_LOCAL_API_FILE_DIR}/shogun
	COMPONENT octave)
