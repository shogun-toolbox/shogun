include_directories(${CMAKE_SOURCE_DIR}/src/shogun
		${CMAKE_BINARY_DIR}/src/shogun
		${MATLAB_INCLUDE_DIR})

add_library(matlab_static SHARED MatlabInterface.cpp)
target_link_libraries(matlab_static shogun ${MATLAB_LIBRARIES})
