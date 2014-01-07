include_directories(${CMAKE_SOURCE_DIR}/src/shogun
		${CMAKE_BINARY_DIR}/src/shogun)

SET(CMDLINE_STATIC_EXEC "${CMAKE_CURRENT_BINARY_DIR}/cmdline_static" PARENT_SCOPE)
add_executable(cmdline_static cmdline.cpp CmdLineInterface.cpp)
target_link_libraries(cmdline_static shogun)

INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/cmdline_static
		DESTINATION bin
		COMPONENT cmdline
		RENAME shogun)

set(CPACK_COMPONENT_CMDLINE_DISPLAY_NAME "Command Line Interface" PARENT_SCOPE)
set(CPACK_COMPONENT_CMDLINE_DESCRIPTION "Command Line interface of Shogun" PARENT_SCOPE)
set(CPACK_COMPONENT_CMDLINE_GROUP "Runtime" PARENT_SCOPE)
set(CPACK_COMPONENT_CMDLINE_DEPENDS libraries PARENT_SCOPE)
SET(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} cmdline PARENT_SCOPE)
