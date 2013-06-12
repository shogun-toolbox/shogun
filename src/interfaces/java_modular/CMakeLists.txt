include(CommonModularInterface)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${JNI_INCLUDE_DIRS})
GENERATE_MODULAR_TARGET(java ${CMAKE_CURRENT_SOURCE_DIR} "")