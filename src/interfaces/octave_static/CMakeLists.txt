include_directories(${CMAKE_SOURCE_DIR}/src/shogun
		${CMAKE_BINARY_DIR}/src/shogun
		${OCTAVE_INCLUDE_DIRS})

add_library(octave_static SHARED OctaveInterface.cpp)
target_link_libraries(octave_static shogun ${OCTAVE_LIBRARIES})
