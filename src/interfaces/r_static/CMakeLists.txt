include_directories(${CMAKE_SOURCE_DIR}/src)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${R_CXX_FLAGS}")

add_library(r_static SHARED RInterface.cpp)
target_link_libraries(r_static shogun ${R_LIBRARIES})
