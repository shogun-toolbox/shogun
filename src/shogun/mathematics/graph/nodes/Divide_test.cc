#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/nodes/Divide.h>
#include <shogun/mathematics/graph/nodes/Input.h>

#include "../test/GraphTest.h"

using namespace shogun;
using namespace shogun::graph;
using namespace std;

TYPED_TEST(GraphTest, divide)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;

	auto X1 = SGVector<NumericType>(10);
	auto X2 = SGVector<NumericType>(10);

	X1.range_fill(1);
	X2.range_fill(1);

	auto expected_result1 = SGVector<NumericType>(10);
	std::transform(
	    X1.data(), X1.data() + X1.size(), X2.data(), expected_result1.data(),
	    std::divides<NumericType>{});
	auto expected_result2 = SGVector<NumericType>(10);
	std::transform(
	    expected_result1.data(),
	    expected_result1.data() + expected_result1.size(), X2.data(),
	    expected_result2.data(), std::divides<NumericType>{});

	auto input =
	    make_shared<node::Input>(Shape{Shape::Dynamic}, TypeParam::type_id);
	auto input1 = make_shared<node::Input>(Shape{10}, TypeParam::type_id);

	auto intermediate = input / input;

	auto output = intermediate / input1;

	auto graph = make_shared<Graph>(
	    vector{input, input1},
	    vector<shared_ptr<node::Node>>{intermediate, output});

	this->test_binary_op_results(
	    graph, X1, X2, expected_result1, expected_result2);
}

TYPED_TEST(GraphTest, vector_scalar_divide)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;
	else
	{
		SGVector<NumericType> X1{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
		NumericType X2{2};

		auto expected_result1 = SGVector<NumericType>(10);
		expected_result1.range_fill();

		auto input1 =
		    make_shared<node::Input>(Shape{Shape::Dynamic}, TypeParam::type_id);
		auto input2 = make_shared<node::Input>(Shape{}, TypeParam::type_id);

		auto output = input1 / input2;

		auto graph = make_shared<Graph>(
		    vector{input1, input2}, vector<shared_ptr<node::Node>>{output});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			std::vector<std::shared_ptr<shogun::graph::Tensor>> result =
			    graph->evaluate(
			        std::vector{std::make_shared<shogun::graph::Tensor>(X1),
			                    std::make_shared<shogun::graph::Tensor>(X2)});

			auto result1 = result[0]->as<shogun::SGVector<NumericType>>();

			for (const auto& [expected_i, result_i] :
			     shogun::zip_iterator(expected_result1, result1))
			{
				EXPECT_EQ(expected_i, result_i);
			}
		}
	}
}

TYPED_TEST(GraphTest, scalar_vector_divide)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;
	else
	{
		NumericType X1{100};
		SGVector<NumericType> X2{1, 2, 4, 5, 10, 20, 25, 50, 100};

		SGVector<NumericType> expected_result1{100, 50, 25, 20, 10, 5, 4, 2, 1};

		auto input1 = make_shared<node::Input>(Shape{}, TypeParam::type_id);
		auto input2 =
		    make_shared<node::Input>(Shape{Shape::Dynamic}, TypeParam::type_id);

		auto output = input1 / input2;

		auto graph = make_shared<Graph>(
		    vector{input1, input2}, vector<shared_ptr<node::Node>>{output});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			std::vector<std::shared_ptr<shogun::graph::Tensor>> result =
			    graph->evaluate(
			        std::vector{std::make_shared<shogun::graph::Tensor>(X1),
			                    std::make_shared<shogun::graph::Tensor>(X2)});

			auto result1 = result[0]->as<shogun::SGVector<NumericType>>();

			for (const auto& [expected_i, result_i] :
			     shogun::zip_iterator(expected_result1, result1))
			{
				EXPECT_EQ(expected_i, result_i);
			}
		}
	}
}