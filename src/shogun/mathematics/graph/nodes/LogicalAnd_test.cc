#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/nodes/Input.h>
#include <shogun/mathematics/graph/nodes/LogicalAnd.h>

#include "../test/GraphTest.h"

using namespace shogun;
using namespace shogun::graph;
using namespace std;

TYPED_TEST(GraphTest, and)
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
		auto output = make_shared<node::LogicalAnd>(input, input1);

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
				EXPECT_EQ(result_i, static_cast<bool>(el1 && el2));
			}

			EXPECT_THROW(result[0]->as<SGVector<float32_t>>(), ShogunException);
		}
	}
	else
	{
		// LogicalAnd only works when both inputs are bool
		EXPECT_THROW(
		    make_shared<node::LogicalAnd>(input, input1), ShogunException);
	}
}


TYPED_TEST(GraphTest, vector_scalar_and)
{
	using NumericType = TypeParam;

	auto input1 = make_shared<node::Input>(
	    Shape{Shape::Dynamic}, get_enum_from_type<NumericType>::type);
	auto input2 = make_shared<node::Input>(
	    Shape{}, get_enum_from_type<NumericType>::type);

	if constexpr (std::is_same_v<NumericType, bool>)
	{
		SGVector<NumericType> X1{true, false, true, false};
		NumericType X2{true};
		
		auto output = make_shared<node::LogicalAnd>(input1, input2);

		auto graph = make_shared<Graph>(
		    vector{input1, input2}, vector<shared_ptr<node::Node>>{output});

		for (auto&& backend : this->m_backends)
		{
			// need to figure out how to handle logical operators
			graph->build(backend);

			vector<shared_ptr<Tensor>> result = graph->evaluate(
			    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});

			EXPECT_EQ(result.size(), 1);

			auto result1 = result[0]->as<SGVector<bool>>();

			for (const auto& [result_i, el1] :
			     zip_iterator(result1, X1))
			{
				EXPECT_EQ(result_i, static_cast<bool>(el1 && X2));
			}

			EXPECT_THROW(result[0]->as<SGVector<float32_t>>(), ShogunException);
		}
	}
	else
	{
		// LogicalAnd only works when both inputs are bool
		EXPECT_THROW(
		    make_shared<node::LogicalAnd>(input1, input2), ShogunException);
	}
}


TYPED_TEST(GraphTest, scalar_vector_and)
{
	using NumericType = TypeParam;

	auto input1 = make_shared<node::Input>(
	    Shape{}, get_enum_from_type<NumericType>::type);
	auto input2 = make_shared<node::Input>(
	    Shape{Shape::Dynamic}, get_enum_from_type<NumericType>::type);

	if constexpr (std::is_same_v<NumericType, bool>)
	{
		NumericType X1{true};
		SGVector<NumericType> X2{true, false, true, false};

		auto output = make_shared<node::LogicalAnd>(input1, input2);

		auto graph = make_shared<Graph>(
		    vector{input1, input2}, vector<shared_ptr<node::Node>>{output});

		for (auto&& backend : this->m_backends)
		{
			// need to figure out how to handle logical operators
			graph->build(backend);

			vector<shared_ptr<Tensor>> result = graph->evaluate(
			    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});

			EXPECT_EQ(result.size(), 1);

			auto result1 = result[0]->as<SGVector<bool>>();

			for (const auto& [result_i, el1] :
			     zip_iterator(result1, X2))
			{
				EXPECT_EQ(result_i, static_cast<bool>(el1 && X1));
			}

			EXPECT_THROW(result[0]->as<SGVector<float32_t>>(), ShogunException);
		}
	}
	else
	{
		// LogicalAnd only works when both inputs are bool
		EXPECT_THROW(
		    make_shared<node::LogicalAnd>(input1, input2), ShogunException);
	}
}