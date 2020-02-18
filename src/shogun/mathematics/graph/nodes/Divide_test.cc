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
	using NumericType = TypeParam;

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

	auto input = make_shared<node::Input>(
	    Shape{Shape::Dynamic}, get_enum_from_type<NumericType>::type);
	auto input1 = make_shared<node::Input>(
	    Shape{10}, get_enum_from_type<NumericType>::type);

	auto intermediate = make_shared<node::Divide>(input, input);

	auto output = make_shared<node::Divide>(intermediate, input1);

	auto graph = make_shared<Graph>(
	    vector{input, input1},
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