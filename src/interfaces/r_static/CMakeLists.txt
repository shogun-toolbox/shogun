include_directories(${CMAKE_SOURCE_DIR}/src)

add_library(r_static SHARED RInterface.cpp)
target_link_libraries(r_static shogun ${R_LIBRARIES})
