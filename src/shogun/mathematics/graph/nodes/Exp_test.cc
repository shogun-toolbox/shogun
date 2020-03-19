#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/nodes/Exp.h>
#include <shogun/mathematics/graph/nodes/Input.h>

#include "../test/GraphTest.h"

using namespace shogun;
using namespace shogun::graph;
using namespace std;

TYPED_TEST(GraphTest, exp)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (!std::is_same_v<NumericType, bool>)
	{
		SGVector<NumericType> X{0, 1, 10};

		SGVector<NumericType> expected_result(X.size());

		std::transform(
		    X.begin(), X.end(), expected_result.begin(),
		    [](const NumericType& el) { return std::exp(el); });

		auto input =
		    make_shared<node::Input>(Shape{Shape::Dynamic}, TypeParam::type_id);

		auto output = make_shared<node::Exp>(input);

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