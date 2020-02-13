set(SHOGUN_GRAPH_HEADERS
	Allocator.h
	Graph.h
	LinalgNodes.h
	OperationImplementation.h
	Shape.h
	Tensor.h
	Types.h
	ops/Add.h
	ops/Input.h
	)

set(SHOGUN_GRAPH_SOURCES
	Graph.cpp
	Tensor.cpp
	ops/Add.cpp
	ops/Input.cpp
	)

add_library(shogun-graph SHARED ${SHOGUN_GRAPH_HEADERS} ${SHOGUN_GRAPH_SOURCES} ${SHOGUN_CORE_HEADERS} ${CMAKE_BINARY_DIR}/src/shogun/lib/config.h)

target_link_libraries(shogun-graph shogun-core)
set_property(TARGET shogun-graph PROPERTY POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(
  shogun-graph PROPERTIES
  VERSION ${LIBSHOGUNVER}
  SOVERSION ${LIBSHOGUNSO}
  INTERFACE_shogun_MAJOR_VERSION ${SHOGUN_VERSION_MAJOR})
set_property(TARGET shogun-graph APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${SHOGUN_VERSION_MAJOR})