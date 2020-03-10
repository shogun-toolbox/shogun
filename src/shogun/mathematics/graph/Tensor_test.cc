#include <gtest/gtest.h>

#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/UniformRealDistribution.h>
#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/nodes/Add.h>
#include <shogun/mathematics/graph/nodes/Input.h>

#include "test/GraphTest.h"

#include <random>

using namespace shogun;
using namespace shogun::graph;
using namespace std;

TYPED_TEST(GraphTest, tensor_lvalue)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;

	auto X1 = SGVector<NumericType>(10);
	auto X2 = SGVector<NumericType>(10);

	X1.range_fill();
	X2.range_fill();

	auto expected_result1 = X1 + X2;
	auto expected_result2 = expected_result1 + X2;

	auto input1 =
	    make_shared<node::Input>(Shape{Shape::Dynamic}, TypeParam::type_id);
	auto input2 = make_shared<node::Input>(Shape{10}, TypeParam::type_id);

	auto intermediate = input1 + input1;

	auto output = intermediate + input2;

	auto graph = make_shared<Graph>(
	    vector{input1, input2},
	    vector<shared_ptr<node::Node>>{intermediate, output});

	for (auto&& backend : this->m_backends)
	{
		graph->build(backend);

		vector<shared_ptr<Tensor>> result = graph->evaluate(
		    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});

		auto result1 = result[0]->as<SGVector<NumericType>>();
		auto result2 = result[1]->as<SGVector<NumericType>>();

		for (const auto& [expected_i, result_i] :
		     zip_iterator(expected_result1, result1))
		{
			EXPECT_EQ(expected_i, result_i);
		}

		for (const auto& [expected_i, result_i] :
		     zip_iterator(expected_result2, result2))
		{
			EXPECT_EQ(expected_i, result_i);
		}
	}
}

TYPED_TEST(GraphTest, tensor_rvalue)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;

	auto input1 =
	    make_shared<node::Input>(Shape{Shape::Dynamic}, TypeParam::type_id);
	auto input2 = make_shared<node::Input>(Shape{10}, TypeParam::type_id);

	auto intermediate = input1 + input1;

	auto output = intermediate + input2;

	auto graph = make_shared<Graph>(
	    vector{input1, input2},
	    vector<shared_ptr<node::Node>>{intermediate, output});

	for (auto&& backend : this->m_backends)
	{
		// vectors reinstantiated in the loop because of the move
		auto X1 = SGVector<NumericType>(10);
		auto X2 = SGVector<NumericType>(10);

		X1.range_fill();
		X2.range_fill();

		auto expected_result1 = X1 + X2;
		auto expected_result2 = expected_result1 + X2;

		graph->build(backend);

		vector<shared_ptr<Tensor>> result =
		    graph->evaluate(vector{make_shared<Tensor>(std::move(X1)),
		                           make_shared<Tensor>(std::move(X2))});

		auto result1 = result[0]->as<SGVector<NumericType>>();
		auto result2 = result[1]->as<SGVector<NumericType>>();

		for (const auto& [expected_i, result_i] :
		     zip_iterator(expected_result1, result1))
		{
			EXPECT_EQ(expected_i, result_i);
		}

		for (const auto& [expected_i, result_i] :
		     zip_iterator(expected_result2, result2))
		{
			EXPECT_EQ(expected_i, result_i);
		}
	}
}

TYPED_TEST(GraphTest, tensor_view)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;

	auto X1 = SGVector<NumericType>(10);
	auto X2 = SGVector<NumericType>(10);

	X1.range_fill();
	X2.range_fill();

	auto expected_result1 = X1 + X2;
	auto expected_result2 = expected_result1 + X2;

	auto input1 =
	    make_shared<node::Input>(Shape{Shape::Dynamic}, TypeParam::type_id);
	auto input2 = make_shared<node::Input>(Shape{10}, TypeParam::type_id);

	auto intermediate = input1 + input1;

	auto output = intermediate + input2;

	auto graph = make_shared<Graph>(
	    vector{input1, input2},
	    vector<shared_ptr<node::Node>>{intermediate, output});

	for (auto&& backend : this->m_backends)
	{
		graph->build(backend);

		vector<shared_ptr<Tensor>> result = graph->evaluate(
		    std::vector{Tensor::create_view(X1), Tensor::create_view(X2)});

		auto result1 = result[0]->as<SGVector<NumericType>>();
		auto result2 = result[1]->as<SGVector<NumericType>>();

		for (const auto& [expected_i, result_i] :
		     zip_iterator(expected_result1, result1))
		{
			EXPECT_EQ(expected_i, result_i);
		}

		for (const auto& [expected_i, result_i] :
		     zip_iterator(expected_result2, result2))
		{
			EXPECT_EQ(expected_i, result_i);
		}
	}
}