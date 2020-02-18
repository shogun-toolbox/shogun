#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/nodes/Input.h>
#include <shogun/mathematics/graph/nodes/LogicalXor.h>

#include "../test/GraphTest.h"

using namespace shogun;
using namespace shogun::graph;
using namespace std;

TYPED_TEST(GraphTest, xor)
{
	using NumericType = TypeParam;

	SGVector<NumericType> X1{true, false, true, false};
	SGVector<NumericType> X2{true, false, false, true};

	auto input = make_shared<node::Input>(
	    Shape{Shape::Dynamic}, get_enum_from_type<NumericType>::type);
	auto input1 = make_shared<node::Input>(
	    Shape{4}, get_enum_from_type<NumericType>::type);

	if constexpr (std::is_same_v<NumericType, bool>)
	{
		auto output = make_shared<node::LogicalXor>(input, input1);

		auto graph = make_shared<Graph>(
		    vector{input, input1}, vector<shared_ptr<node::Node>>{output});

		for (auto&& backend : this->m_backends)
		{
			// need to figure out how to handle logical operators
			graph->build(backend);

			vector<shared_ptr<Tensor>> result = graph->evaluate(
			    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});

			EXPECT_EQ(result.size(), 1);

			auto result1 = result[0]->as<SGVector<bool>>();

			for (const auto& [result_i, el1, el2] :
			     zip_iterator(result1, X1, X2))
			{
				EXPECT_EQ(result_i, static_cast<bool>(!el1 != !el2));
			}

			EXPECT_THROW(result[0]->as<SGVector<float32_t>>(), ShogunException);
		}
	}
	else
	{
		EXPECT_THROW(
		    make_shared<node::LogicalXor>(input, input1), ShogunException);
	}
}