#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/nodes/Reshape.h>
#include <shogun/mathematics/graph/nodes/Input.h>

#include "../test/GraphTest.h"

using namespace shogun;
using namespace shogun::graph;
using namespace std;


TYPED_TEST(GraphTest, vector_reshape)
{
	using NumericType = TypeParam;

	auto X1 = SGVector<NumericType>(10);

	auto expected_result1 = SGMatrix<NumericType>(10, 1);
	auto expected_result2 = SGMatrix<NumericType>(1, 10);

	auto input1 = make_shared<node::Input>(
	    Shape{Shape::Dynamic}, get_enum_from_type<NumericType>::type);
	auto input2 = make_shared<node::Input>(
	    Shape{10}, get_enum_from_type<NumericType>::type);

	auto output1 = make_shared<node::Reshape>(input1, Shape{10, 1});
	auto output2 = make_shared<node::Reshape>(input2, Shape{1, 10});

	EXPECT_THROW(make_shared<node::Reshape>(input2, Shape{10, 2}), ShogunException);

	auto graph = make_shared<Graph>(
	    vector{input1, input2},
	    vector<shared_ptr<node::Node>>{output1, output2});

	for (auto&& backend : this->m_backends)
	{
		graph->build(backend);

		vector<shared_ptr<Tensor>> result = graph->evaluate(
		    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X1)});

		EXPECT_THROW(result[0]->as<SGVector<NumericType>>(), ShogunException);
		auto result1 = result[0]->as<SGMatrix<NumericType>>();
		auto result2 = result[1]->as<SGMatrix<NumericType>>();

		EXPECT_EQ(result1.num_rows, expected_result1.num_rows);
		EXPECT_EQ(result1.num_cols, expected_result1.num_cols);
		EXPECT_EQ(result2.num_rows, expected_result2.num_rows);
		EXPECT_EQ(result2.num_cols, expected_result2.num_cols);
	}
}

TYPED_TEST(GraphTest, matrix_reshape)
{
	using NumericType = TypeParam;

	auto X1 = SGMatrix<NumericType>(10, 1);

	auto expected_result1 = SGMatrix<NumericType>(5, 2);
	auto expected_result2 = SGVector<NumericType>(10);

	auto input1 = make_shared<node::Input>(
	    Shape{Shape::Dynamic, Shape::Dynamic}, get_enum_from_type<NumericType>::type);
	auto input2 = make_shared<node::Input>(
	    Shape{10, 1}, get_enum_from_type<NumericType>::type);

	auto output1 = make_shared<node::Reshape>(input1, Shape{5, 2});
	auto output2 = make_shared<node::Reshape>(input2, Shape{10});

	EXPECT_THROW(make_shared<node::Reshape>(input2, Shape{10, 2}), ShogunException);

	auto graph = make_shared<Graph>(
	    vector{input1, input2},
	    vector<shared_ptr<node::Node>>{output1, output2});

	for (auto&& backend : this->m_backends)
	{
		graph->build(backend);

		vector<shared_ptr<Tensor>> result = graph->evaluate(
		    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X1)});

		auto result1 = result[0]->as<SGMatrix<NumericType>>();
		auto result2 = result[1]->as<SGVector<NumericType>>();

		EXPECT_EQ(result1.num_rows, expected_result1.num_rows);
		EXPECT_EQ(result1.num_cols, expected_result1.num_cols);
		EXPECT_EQ(result2.vlen, expected_result2.vlen);
	}
}
