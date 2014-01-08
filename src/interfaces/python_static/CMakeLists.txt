include_directories(${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_DIRS})

add_library(python_static SHARED PythonInterface.cpp)
target_link_libraries(python_static shogun ${PYTHON_LIBRARIES})
