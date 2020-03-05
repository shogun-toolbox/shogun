#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/nodes/Input.h>
#include <shogun/mathematics/graph/nodes/Sign.h>

#include "../test/GraphTest.h"

using namespace shogun;
using namespace shogun::graph;
using namespace std;

TYPED_TEST(GraphTest, sign)
{
	using NumericType = TypeParam;

	if constexpr (std::is_unsigned_v<NumericType>)
	{
		auto input = make_shared<node::Input>(
		    Shape{Shape::Dynamic}, get_enum_from_type<NumericType>::type);
		EXPECT_THROW(make_shared<node::Sign>(input), ShogunException);
	}
	else
	{
		SGVector<NumericType> X{-1, 0, 1, -10, 10};

		SGVector<NumericType> expected_result{-1, 0, 1, -1, 1};

		auto input = make_shared<node::Input>(
		    Shape{Shape::Dynamic}, get_enum_from_type<NumericType>::type);

		auto output = make_shared<node::Sign>(input);

		auto graph = make_shared<Graph>(
		    vector{input}, vector<shared_ptr<node::Node>>{output});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			vector<shared_ptr<Tensor>> result =
			    graph->evaluate(vector{make_shared<Tensor>(X)});

			auto result1 = result[0]->as<SGVector<NumericType>>();

			for (const auto& [result_i, el1] :
			     zip_iterator(result1, expected_result))
			{
				EXPECT_EQ(result_i, el1);
			}
		}
	}
}