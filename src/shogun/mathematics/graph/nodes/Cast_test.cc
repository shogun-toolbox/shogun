#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/nodes/Cast.h>
#include <shogun/mathematics/graph/nodes/Input.h>

#include "../test/GraphTest.h"

using namespace shogun;
using namespace shogun::graph;
using namespace std;

TYPED_TEST(GraphTest, cast)
{
	using NumericType = TypeParam;

	auto X1 = SGVector<NumericType>(10);

	X1.range_fill();

	auto input = make_shared<node::Input>(
	    Shape{Shape::Dynamic}, get_enum_from_type<NumericType>::type);

	auto output = make_shared<node::Cast>(input, element_type::FLOAT64);

	auto graph = make_shared<Graph>(
	    vector{input}, vector<shared_ptr<node::Node>>{output});

	for (auto&& backend : this->m_backends)
	{
		graph->build(backend);

		vector<shared_ptr<Tensor>> result =
		    graph->evaluate(vector{make_shared<Tensor>(X1)});

		auto result1 = result[0]->as<SGVector<float64_t>>();
		if constexpr (!std::is_same_v<NumericType, float64_t>)
		{
			EXPECT_THROW(
			    result[0]->as<SGVector<NumericType>>(), ShogunException);
		}

		for (const auto& [result_i, el1] : zip_iterator(result1, X1))
		{
			EXPECT_EQ(static_cast<float64_t>(result_i), el1);
		}
	}
}