#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/ops/Add.h>
#include <shogun/mathematics/graph/ops/Input.h>

#include <shogun/mathematics/graph/operator_list.h>

using namespace shogun;
using namespace std;

template <typename T>
class GraphTest : public ::testing::Test {};

using NumericTypes = ::testing::Types<float32_t, float64_t>;
TYPED_TEST_CASE(GraphTest, NumericTypes);

TYPED_TEST(GraphTest, add)
{
	auto X = SGVector<TypeParam>(10);

	auto input = make_shared<Input>(Shape{Shape::Dynamic}, get_enum_from_type<TypeParam>::type);
	auto input1 = make_shared<Input>(Shape{2}, get_enum_from_type<TypeParam>::type);

	auto intermediate = make_shared<Add>(input, input);

	cout << intermediate->get_tensors()[0] << '\n';
	cout << *intermediate << '\n';

	auto output = make_shared<Add>(input, input1);

	cout << output->get_tensors()[0] << '\n';
	cout << *output << '\n';

	auto graph = make_shared<Graph>(vector{input, input1}, vector<shared_ptr<Node>>{intermediate, output});

	graph->build();

	/*vector<shared_ptr<Tensor>> result =*/ graph->evaluate(vector{make_shared<Tensor>(X)});
}