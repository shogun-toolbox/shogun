// #include <gtest/gtest.h>

// #include <shogun/mathematics/graph/Graph.h>
// #include <shogun/mathematics/graph/nodes/MatMul.h>
// #include <shogun/mathematics/graph/nodes/Input.h>

// #include "../test/GraphTest.h"

// using namespace shogun;
// using namespace shogun::graph;
// using namespace std;

// TYPED_TEST(GraphTest, matrix_2D_2D_no_transpose)
// {
// 	using NumericType = TypeParam;

//     if constexpr(std::is_same_v<TypeParam, bool>)
//     	return;

// 	SGMatrix<NumericType> X1 {{1,3,5},{2,4,6}};
// 	SGMatrix<NumericType> X2 {{1,2},{3,4}, {5,6}};
// 	SGMatrix<NumericType> expected_result{{5,11,17},{11,25,39},{17,39,61}};

// 	auto A = make_shared<node::Input>(
// 	    Shape{Shape::Dynamic, Shape::Dynamic},
// get_enum_from_type<NumericType>::type); 	auto B = make_shared<node::Input>(
// 	    Shape{Shape::Dynamic, Shape::Dynamic},
// get_enum_from_type<NumericType>::type);

// 	auto output = make_shared<node::MatMul>(A, B);

// 	auto graph = make_shared<Graph>(
// 	    vector{A, B},
// 	    vector<shared_ptr<node::Node>>{intermediate, output});

// 	for (auto&& backend : this->m_backends)
// 	{
// 		if (backend == GRAPH_BACKEND::SHOGUN)
// 			continue;
// 		graph->build(backend);

// 		vector<shared_ptr<Tensor>> result = graph->evaluate(
// 		    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});

// 		auto result1 = result[0]->as<SGMatrix<NumericType>>();

// 		for (const auto& [expected_i, result_i] :
// 		     zip_iterator(expected_result1, result1))
// 		{
// 			EXPECT_EQ(expected_i, result_i);
// 		}

// 		for (const auto& [expected_i, result_i] :
// 		     zip_iterator(expected_result2, result2))
// 		{
// 			EXPECT_EQ(expected_i, result_i);
// 		}
// 	}
// }