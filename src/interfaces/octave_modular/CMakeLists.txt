include(CommonModularInterface)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${OCTAVE_INCLUDE_DIR})

GENERATE_MODULAR_TARGET(octave ${CMAKE_CURRENT_SOURCE_DIR} ${OCTAVE_LIBRARIES})

INSTALL(TARGETS octave_modular
	LIBRARY DESTINATION lib/lua/)