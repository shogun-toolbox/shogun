#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Array.h>

#include "test/GraphTest.h"

#include <random>

using namespace shogun;
using namespace shogun::graph;
using namespace std;

TYPED_TEST(GraphTest, array_lvalue)
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

	for (auto&& backend : this->m_backends)
	{
		auto input1 = make_shared<Array>(X1);
		auto input2 = make_shared<Array>(X2);
		auto intermediate = input1 + input1;
		auto output = intermediate + input2;

		ShogunEnv::instance()->set_graph_backend(backend);

		auto result1 =
		    intermediate->evaluate()->template as<SGVector<NumericType>>();
		auto result2 = output->evaluate()->template as<SGVector<NumericType>>();

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


TYPED_TEST(GraphTest, array_rvalue)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;

	for (auto&& backend : this->m_backends)
	{
		auto X1 = SGVector<NumericType>(10);
		auto X2 = SGVector<NumericType>(10);

		X1.range_fill();
		X2.range_fill();

		auto expected_result1 = X1 + X2;
		auto expected_result2 = expected_result1 + X2;

		auto input1 = make_shared<Array>(std::move(X1));
		auto input2 = make_shared<Array>(std::move(X2));
		auto intermediate = input1 + input1;
		auto output = intermediate + input2;

		ShogunEnv::instance()->set_graph_backend(backend);

		auto result1 =
		    intermediate->evaluate()->template as<SGVector<NumericType>>();
		auto result2 = output->evaluate()->template as<SGVector<NumericType>>();

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


TYPED_TEST(GraphTest, array_view)
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

	for (auto&& backend : this->m_backends)
	{
		auto input1 = Array::create_view(X1);
		auto input2 = Array::create_view(X2);
		auto intermediate = input1 + input1;
		auto output = intermediate + input2;

		ShogunEnv::instance()->set_graph_backend(backend);

		auto result1 =
		    intermediate->evaluate()->template as<SGVector<NumericType>>();
		auto result2 = output->evaluate()->template as<SGVector<NumericType>>();

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