include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(cmdline_static cmdline.cpp CmdLineInterface.cpp)
target_link_libraries(cmdline_static shogun)
