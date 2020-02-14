#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/nodes/Add.h>
#include <shogun/mathematics/graph/nodes/Input.h>

#include <shogun/mathematics/graph/operator_list.h>

using namespace shogun;
using namespace std;

template <typename T>
class GraphTest : public ::testing::Test {};

using NumericTypes = ::testing::Types<float32_t, float64_t>;
TYPED_TEST_CASE(GraphTest, NumericTypes);

TYPED_TEST(GraphTest, add)
{
	auto X1 = SGVector<TypeParam>(10);
	auto X2 = SGVector<TypeParam>(10);

	X1.range_fill();
	X2.range_fill();

	auto expected_result1 = X1 + X2;
	auto expected_result2 = expected_result1 + X2;


	auto input = make_shared<Input>(Shape{Shape::Dynamic}, get_enum_from_type<TypeParam>::type);
	auto input1 = make_shared<Input>(Shape{10}, get_enum_from_type<TypeParam>::type);

	auto intermediate = make_shared<Add>(input, input);

	auto output = make_shared<Add>(intermediate, input1);

	auto graph = make_shared<Graph>(vector{input, input1}, vector<shared_ptr<Node>>{intermediate, output});

	graph->build();

	vector<shared_ptr<Tensor>> result = graph->evaluate(vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});

	auto result1 = result[0]->as<SGVector<TypeParam>>();
	auto result2 = result[1]->as<SGVector<TypeParam>>();

	for (const auto& [expected_i, result_i]: zip_iterator(expected_result1, result1))
	{
		EXPECT_EQ(expected_i, result_i);
	}

	for (const auto& [expected_i, result_i]: zip_iterator(expected_result2, result2))
	{
		EXPECT_EQ(expected_i, result_i);
	}
}