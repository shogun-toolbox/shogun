include_directories(${CMAKE_SOURCE_DIR}/src ${MATLAB_INCLUDE_DIR})

add_library(matlab_static SHARED MatlabInterface.cpp)
target_link_libraries(matlab_static shogun ${MATLAB_LIBRARIES})
