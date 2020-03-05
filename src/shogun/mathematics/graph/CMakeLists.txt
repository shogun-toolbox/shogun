set(SHOGUN_CORE_GRAPH_HEADERS
	${CMAKE_CURRENT_SOURCE_DIR}/Allocator.h
	${CMAKE_CURRENT_SOURCE_DIR}/Graph.h
	${CMAKE_CURRENT_SOURCE_DIR}/GraphExecutor.h
	${CMAKE_CURRENT_SOURCE_DIR}/Shape.h
	${CMAKE_CURRENT_SOURCE_DIR}/Tensor.h
	${CMAKE_CURRENT_SOURCE_DIR}/Types.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Add.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Equal.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/LogicalAnd.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/LogicalOr.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/LogicalXor.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Multiply.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Divide.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Subtract.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Input.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Dot.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Reshape.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/MatMul.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Cast.h
	#${CMAKE_CURRENT_SOURCE_DIR}/ops/abstract/UnaryOperator.h
	)

set(SHOGUN_CORE_GRAPH_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/Graph.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/GraphExecutor.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Tensor.cpp
	)

set(SHOGUN_GRAPH_IMPLEMENTATION_HEADERS
		${CMAKE_CURRENT_SOURCE_DIR}/ops/abstract/Operator.h
		${CMAKE_CURRENT_SOURCE_DIR}/ops/abstract/BinaryOperator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Add.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Equal.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/LogicalAnd.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/LogicalOr.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/LogicalXor.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Divide.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Input.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Multiply.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Dot.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Reshape.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/MatMul.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Subtract.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/ShogunBinaryNode.h
		${CMAKE_CURRENT_SOURCE_DIR}/ShogunGraph.h
		${PROJECT_SOURCE_DIR}/src/shogun/io/SGIO.h
		)
set(SHOGUN_GRAPH_IMPLEMENTATION_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/ShogunGraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Cast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Divide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Dot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/LogicalAnd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/LogicalOr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/LogicalXor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/MatMul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Multiply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Subtract.cpp
	)

set(NGRAPH_IMPLEMENTATION_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Add.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Dot.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Equal.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalAnd.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalOr.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalXor.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Divide.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Multiply.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Input.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Subtract.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Reshape.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/MatMul.h
		)

set(NGRAPH_IMPLEMENTATION_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/NGraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Cast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Divide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Dot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalAnd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalOr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalXor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/MatMul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Multiply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Subtract.cpp
 	)

add_library(shogun-engine SHARED ${SHOGUN_GRAPH_HEADERS}
	${SHOGUN_GRAPH_SOURCES}
	${SHOGUN_CORE_GRAPH_SOURCES}
	${SHOGUN_CORE_GRAPH_HEADERS}
)
generate_export_header(shogun-engine)

target_link_libraries(shogun-engine shogun-core)
PLUGIN_COMPATIBILITY(shogun-engine)
CommonOutputDir(shogun-engine)

function(add_executor EXECUTOR_NAME)

    cmake_parse_arguments(
        EXECUTOR
        ""
        "ADDITIONAL_SHARED_LIBRARIES;ADDITIONAL_INCLUDE_DIRS"
        "IMPLEMENTATION_HEADERS;IMPLEMENTATION_SOURCES"
        ${ARGN}
    )

    add_library(${EXECUTOR_NAME} SHARED
		${EXECUTOR_IMPLEMENTATION_HEADERS}
		${EXECUTOR_IMPLEMENTATION_SOURCES}
	)

	target_include_directories(${EXECUTOR_NAME} PRIVATE ${EXECUTOR_ADDITIONAL_INCLUDE_DIRS})
	target_link_libraries(${EXECUTOR_NAME} shogun-engine ${EXECUTOR_ADDITIONAL_SHARED_LIBRARIES})
	PLUGIN_COMPATIBILITY(${EXECUTOR_NAME})
	set(EXECUTORS ${EXECUTORS} ${EXECUTOR_NAME} PARENT_SCOPE)
	CommonOutputDir(${EXECUTOR_NAME})
endfunction()

function (add_graph_test TEST_FILE)
	get_filename_component(TARGET_NAME ${TEST_FILE} NAME)
	ADD_SHOGUN_TEST(${TEST_FILE})
	add_dependencies(${TARGET_NAME} ${EXECUTORS})
endfunction()

# add executors
add_executor(shogun-vanilla-executor
	IMPLEMENTATION_HEADERS ${SHOGUN_GRAPH_IMPLEMENTATION_HEADERS}
	IMPLEMENTATION_SOURCES ${SHOGUN_GRAPH_IMPLEMENTATION_SOURCES}
	)

find_package(NGraph)
if (NGraph_FOUND)
	add_executor(shogun-ngraph-executor
		IMPLEMENTATION_HEADERS ${NGRAPH_IMPLEMENTATION_HEADERS}
		IMPLEMENTATION_SOURCES ${NGRAPH_IMPLEMENTATION_SOURCES}
		ADDITIONAL_SHARED_LIBRARIES ${NGraph_LIBRARIES}
		ADDITIONAL_INCLUDE_DIRS ${NGraph_INCLUDE_DIRS}
		)
endif()

# add graph tests for available executors
IF (ENABLE_TESTING)
	enable_testing()
	set(SHOGUN_TEST_LINK_LIBS shogun_test_main shogun-engine)

	add_graph_test(nodes/Add_test)
	add_graph_test(nodes/Equal_test)
	add_graph_test(nodes/Subtract_test)
	add_graph_test(nodes/Multiply_test)
	add_graph_test(nodes/Divide_test)
	add_graph_test(nodes/LogicalAnd_test)
	add_graph_test(nodes/LogicalOr_test)
	add_graph_test(nodes/LogicalXor_test)
	add_graph_test(nodes/Dot_test)
	add_graph_test(nodes/Reshape_test)
	add_graph_test(nodes/MatMul_test)
	add_graph_test(nodes/Cast_test)

ENDIF()
