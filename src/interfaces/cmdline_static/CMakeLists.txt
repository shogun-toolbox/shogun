include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(shogun_cmd cmdline.cpp CmdLineInterface.cpp)
target_link_libraries(shogun_cmd shogun)