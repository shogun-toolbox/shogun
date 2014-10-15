include_directories(${OCTAVE_INCLUDE_DIRS} ${SYSTEM_INCLUDES})

add_library(octave_static SHARED OctaveInterface.cpp)
target_link_libraries(octave_static shogun ${OCTAVE_LIBRARIES})
