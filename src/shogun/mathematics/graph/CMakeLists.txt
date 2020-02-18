set(SHOGUN_CORE_GRAPH_HEADERS
	${CMAKE_CURRENT_SOURCE_DIR}/Allocator.h
	${CMAKE_CURRENT_SOURCE_DIR}/Graph.h
	${CMAKE_CURRENT_SOURCE_DIR}/GraphExecutor.h
	${CMAKE_CURRENT_SOURCE_DIR}/Shape.h
	${CMAKE_CURRENT_SOURCE_DIR}/Tensor.h
	${CMAKE_CURRENT_SOURCE_DIR}/Types.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Add.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Multiply.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Divide.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Subtract.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Input.h
	${CMAKE_CURRENT_SOURCE_DIR}/ops/abstract/Operator.h
	${CMAKE_CURRENT_SOURCE_DIR}/ops/abstract/BinaryOperator.h
	)

set(SHOGUN_CORE_GRAPH_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/Graph.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/GraphExecutor.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Tensor.cpp
	)

set(SHOGUN_GRAPH_IMPLEMENTATION_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Add.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Divide.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Multiply.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Input.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Subtract.h
	${CMAKE_CURRENT_SOURCE_DIR}/ShogunGraph.h
	)
set(SHOGUN_GRAPH_IMPLEMENTATION_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/ShogunGraph.cpp
	)

set(NGRAPH_IMPLEMENTATION_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Add.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Divide.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Multiply.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Input.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Subtract.h

	${CMAKE_CURRENT_SOURCE_DIR}/NGraph.h
 	)
set(NGRAPH_IMPLEMENTATION_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/NGraph.cpp
 	)

add_library(shogun-engine SHARED ${SHOGUN_GRAPH_HEADERS}
	${SHOGUN_GRAPH_SOURCES}
	${SHOGUN_CORE_GRAPH_SOURCES}
	${SHOGUN_CORE_GRAPH_HEADERS}
)

target_link_libraries(shogun-engine shogun-core)
set_property(TARGET shogun-engine PROPERTY POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(
  shogun-engine PROPERTIES
  VERSION ${LIBSHOGUNVER}
  SOVERSION ${LIBSHOGUNSO}
  INTERFACE_shogun_MAJOR_VERSION ${SHOGUN_VERSION_MAJOR}
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/src/shogun)
set_property(TARGET shogun-engine APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${SHOGUN_VERSION_MAJOR})

add_library(shogun-vanilla-executor SHARED
	${SHOGUN_GRAPH_IMPLEMENTATION_HEADERS}
	${SHOGUN_GRAPH_IMPLEMENTATION_SOURCES}
)
target_link_libraries(shogun-vanilla-executor shogun-engine)
# todo create a macro/function for this
set_property(TARGET shogun-vanilla-executor PROPERTY POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(
  shogun-vanilla-executor PROPERTIES
  VERSION ${LIBSHOGUNVER}
  SOVERSION ${LIBSHOGUNSO}
  INTERFACE_shogun_MAJOR_VERSION ${SHOGUN_VERSION_MAJOR}
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/src/shogun)
set_property(TARGET shogun-vanilla-executor APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${SHOGUN_VERSION_MAJOR})
LIST(APPEND EXCUTORS shogun-vanilla-executor)

find_package(NGraph)
if (NGraph_FOUND)
	add_library(shogun-ngraph-executor SHARED
		${NGRAPH_IMPLEMENTATION_HEADERS}
		${NGRAPH_IMPLEMENTATION_SOURCES}
		)
	target_include_directories(shogun-ngraph-executor PRIVATE ${NGraph_INCLUDE_DIRS})
	target_link_libraries(shogun-ngraph-executor shogun-engine ${NGraph_LIBRARIES})
	set_property(TARGET shogun-ngraph-executor PROPERTY POSITION_INDEPENDENT_CODE TRUE)
	set_target_properties(
		shogun-ngraph-executor PROPERTIES
		VERSION ${LIBSHOGUNVER}
		SOVERSION ${LIBSHOGUNSO}
		INTERFACE_shogun_MAJOR_VERSION ${SHOGUN_VERSION_MAJOR}
		LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/src/shogun)
	set_property(TARGET shogun-ngraph-executor APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${SHOGUN_VERSION_MAJOR})
	LIST(APPEND EXCUTORS shogun-ngraph-executor)
endif()

IF (ENABLE_TESTING)
	set(SHOGUN_TEST_LINK_LIBS shogun_test_main shogun-engine)

	ADD_SHOGUN_TEST(nodes/Add_test)
	ADD_SHOGUN_TEST(nodes/Subtract_test)
	ADD_SHOGUN_TEST(nodes/Multiply_test)
	ADD_SHOGUN_TEST(nodes/Divide_test)

	if (UNIX AND NOT APPLE)
		set_property(TEST Add_test PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src/shogun")
		set_property(TEST Subtract_test PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src/shogun")
		set_property(TEST Multiply_test PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src/shogun")
		set_property(TEST Divide_test PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src/shogun")
	endif()
	add_dependencies(Add_test ${EXCUTORS})
	add_dependencies(Subtract_test ${EXCUTORS})
	add_dependencies(Multiply_test ${EXCUTORS})
	add_dependencies(Divide_test ${EXCUTORS})
ENDIF()
