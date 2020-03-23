set(SHOGUN_CORE_GRAPH_HEADERS
	${CMAKE_CURRENT_SOURCE_DIR}/Array.h
	${CMAKE_CURRENT_SOURCE_DIR}/CPUArch.h
	${CMAKE_CURRENT_SOURCE_DIR}/Graph.h
	${CMAKE_CURRENT_SOURCE_DIR}/GraphExecutor.h
	${CMAKE_CURRENT_SOURCE_DIR}/Shape.h
	${CMAKE_CURRENT_SOURCE_DIR}/Tensor.h
	${CMAKE_CURRENT_SOURCE_DIR}/Types.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Add.h
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/BinaryNode.h
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
	)

set(SHOGUN_CORE_GRAPH_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/CPUArch.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Graph.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/GraphExecutor.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Tensor.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Types.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Shape.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Storage.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/BinaryNode.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/nodes/Dot.cpp
	# ${CMAKE_SOURCE_DIR}/src/shogun/features/DataGenerator.cpp
	# ${CMAKE_SOURCE_DIR}/src/shogun/distributions/Gaussian.cpp
	# ${CMAKE_SOURCE_DIR}/src/shogun/distributions/Distribution.cpp
	# ${CMAKE_SOURCE_DIR}/src/shogun/features/DotFeatures.cpp
	# ${CMAKE_SOURCE_DIR}/src/shogun/mathematics/linalg/backend/eigen/BasicOps.cpp
	# ${CMAKE_SOURCE_DIR}/src/shogun/mathematics/linalg/backend/eigen/Decompositions.cpp
	# ${CMAKE_SOURCE_DIR}/src/shogun/mathematics/linalg/backend/eigen/EigenSolvers.cpp
	# ${CMAKE_SOURCE_DIR}/src/shogun/mathematics/linalg/backend/eigen/Functions.cpp
	# ${CMAKE_SOURCE_DIR}/src/shogun/mathematics/linalg/backend/eigen/Misc.cpp
	# ${CMAKE_SOURCE_DIR}/src/shogun/mathematics/linalg/backend/eigen/Solvers.cpp
	# ${CMAKE_SOURCE_DIR}/src/shogun/mathematics/linalg/backend/eigen/SpecialPurposes.cpp
	)

set(SHOGUN_GRAPH_IMPLEMENTATION_HEADERS
		${CMAKE_CURRENT_SOURCE_DIR}/ops/abstract/UnaryOperator.h
		${CMAKE_CURRENT_SOURCE_DIR}/ops/abstract/Operator.h
		${CMAKE_CURRENT_SOURCE_DIR}/ops/abstract/BinaryOperator.h
		${CMAKE_CURRENT_SOURCE_DIR}/ops/abstract/utils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Add.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Equal.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Exp.h
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
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Sign.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/ShogunBinaryNode.h
		${CMAKE_CURRENT_SOURCE_DIR}/ShogunGraph.h
		${PROJECT_SOURCE_DIR}/src/shogun/io/SGIO.h
		)

set (SHOGUN_GRAPH_IMPLEMENTATION_AVX512F
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/add_avx512f.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/divide_avx512f.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/multiply_avx512f.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/subtract_avx512f.cpp
	)

set (SHOGUN_GRAPH_IMPLEMENTATION_AVX
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/add_avx.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/divide_avx.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/multiply_avx.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/subtract_avx.cpp
	)

set (SHOGUN_GRAPH_IMPLEMENTATION_SSE2
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/add_sse2.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/divide_sse2.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/multiply_sse2.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/subtract_sse2.cpp
	)

set(SHOGUN_GRAPH_IMPLEMENTATION_SSE4_1
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/multiply_sse4_1.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/details/subtract_sse4_1.cpp
	)

set(SHOGUN_GRAPH_IMPLEMENTATION_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/ShogunGraph.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/abstract/Operator.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/abstract/utils.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/Dot.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ops/shogun/MatMul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Cast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Divide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Dot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Exp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/LogicalAnd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/LogicalOr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/LogicalXor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/MatMul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Multiply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Sign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/shogun/Subtract.cpp
	)

set(NGRAPH_IMPLEMENTATION_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Add.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Dot.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Equal.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Exp.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalAnd.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalOr.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalXor.h
		${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Divide.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Multiply.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Input.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Subtract.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Reshape.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Sign.h
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/MatMul.h
		)

set(NGRAPH_IMPLEMENTATION_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/NGraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Cast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Divide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Dot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Exp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalAnd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalOr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/LogicalXor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/MatMul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Multiply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Sign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ngraph/Subtract.cpp
 	)

FIND_PACKAGE(Eigen3 ${EIGEN_VERSION_MINIMUM})
IF(NOT EIGEN3_FOUND)
  include(external/Eigen3)
  set(EIGEN_INCLUDE_DIR $<INSTALL_INTERFACE:include/shogun/third_party/eigen>)
ELSE()
  # eigen3 bug related to aliasing operators and self-storing.
  IF(${EIGEN_VERSION} VERSION_GREATER 3.2.90 AND ${EIGEN_VERSION} VERSION_LESS 3.3.0)
      MESSAGE(WARNING "The system Eigen3 version ${EIGEN_VERSION} contains a lot of bugs, which could cause some models failing!")
  ENDIF()
ENDIF()

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

add_library(shogun-vanilla-executor-avx512f STATIC ${SHOGUN_GRAPH_IMPLEMENTATION_AVX512F})
target_include_directories(shogun-vanilla-executor-avx512f PRIVATE ${EIGEN_INCLUDE_DIR})
target_compile_options(shogun-vanilla-executor-avx512f PRIVATE -mavx512f)
set_property(TARGET shogun-vanilla-executor-avx512f PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(shogun-vanilla-executor-avx STATIC ${SHOGUN_GRAPH_IMPLEMENTATION_AVX})
target_include_directories(shogun-vanilla-executor-avx PRIVATE ${EIGEN_INCLUDE_DIR})
target_compile_options(shogun-vanilla-executor-avx PRIVATE -mavx)
set_property(TARGET shogun-vanilla-executor-avx PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(shogun-vanilla-executor-sse2 STATIC ${SHOGUN_GRAPH_IMPLEMENTATION_SSE2})
target_include_directories(shogun-vanilla-executor-sse2 PRIVATE ${EIGEN_INCLUDE_DIR})
target_compile_options(shogun-vanilla-executor-sse2 PRIVATE -msse2)
set_property(TARGET shogun-vanilla-executor-sse2 PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(shogun-vanilla-executor-sse41 STATIC ${SHOGUN_GRAPH_IMPLEMENTATION_SSE4_1})
target_include_directories(shogun-vanilla-executor-sse41 PRIVATE ${EIGEN_INCLUDE_DIR})
target_compile_options(shogun-vanilla-executor-sse41 PRIVATE -msse4.1)
set_property(TARGET shogun-vanilla-executor-sse41 PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(shogun-vanilla-executor shogun-vanilla-executor-sse2)
target_link_libraries(shogun-vanilla-executor shogun-vanilla-executor-sse41)
target_link_libraries(shogun-vanilla-executor shogun-vanilla-executor-avx)
target_link_libraries(shogun-vanilla-executor shogun-vanilla-executor-avx512f)


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
	add_graph_test(nodes/Sign_test)
	add_graph_test(nodes/Exp_test)
	add_graph_test(Graph_test)
	add_graph_test(Tensor_test)
	add_graph_test(Array_test)

ENDIF()
