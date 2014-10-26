include_directories(${SYSTEM_INCLUDES})
SET(CMDLINE_STATIC_EXEC "${CMAKE_CURRENT_BINARY_DIR}/cmdline_static" PARENT_SCOPE)
add_executable(cmdline_static cmdline.cpp CmdLineInterface.cpp)
target_link_libraries(cmdline_static shogun)

INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/cmdline_static
		DESTINATION bin
		COMPONENT cmdline
		RENAME shogun)
