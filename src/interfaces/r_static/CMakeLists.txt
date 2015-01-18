include_directories(${SYSTEM_INCLUDES})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${R_CXX_FLAGS}")

add_library(r_static SHARED RInterface.cpp)
target_link_libraries(r_static shogun ${R_LIBRARIES})
set_target_properties(r_static PROPERTIES PREFIX "")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/r_static.so
       DESTINATION ${R_COMPONENT_LIB_PATH}/sg/libs
       COMPONENT r
   	   RENAME sg.so)

install(CODE "FILE(MAKE_DIRECTORY \$ENV{DESTDIR}/${R_COMPONENT_LIB_PATH}/sg/Meta)")
install(CODE "FILE(MAKE_DIRECTORY \$ENV{DESTDIR}/${R_COMPONENT_LIB_PATH}/sg/R)")
install(CODE "EXECUTE_PROCESS(COMMAND ${CMAKE_SOURCE_DIR}/src/.r-install.sh ${R_COMPONENT_LIB_PATH} sg so saveRDS WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
