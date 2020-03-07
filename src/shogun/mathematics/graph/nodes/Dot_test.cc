#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/nodes/Add.h>
#include <shogun/mathematics/graph/nodes/Dot.h>
#include <shogun/mathematics/graph/nodes/Input.h>

#include "../test/GraphTest.h"

using namespace shogun;
using namespace shogun::graph;
using namespace std;

TYPED_TEST(GraphTest, vector_scalar_dot)
{
	using NumericType = TypeParam;

	if constexpr (std::is_same_v<TypeParam, bool>)
		return;
	else
	{
		SGVector<NumericType> X1{1, 2, 3};
		NumericType X2 = 2;
		SGMatrix<NumericType> X3{{1, 2, 3}, {4, 5, 6}};
		SGVector<NumericType> expected_result1 = {2, 4, 6};
		SGMatrix<NumericType> expected_result2{{2, 4, 6}, {8, 10, 12}};

		auto A = make_shared<node::Input>(
		    Shape{3}, get_enum_from_type<NumericType>::type);
		auto B = make_shared<node::Input>(
		    Shape{}, get_enum_from_type<NumericType>::type);
		auto C = make_shared<node::Input>(
		    Shape{3, 2}, get_enum_from_type<NumericType>::type);

		auto output1 = make_shared<node::Dot>(A, B);
		auto output2 = make_shared<node::Dot>(B, C);

		auto graph = make_shared<Graph>(
		    vector{A, B, C}, vector<shared_ptr<node::Node>>{output1, output2});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			vector<shared_ptr<Tensor>> result = graph->evaluate(
			    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2),
			           make_shared<Tensor>(X3)});

			auto result1 = result[0]->as<SGVector<NumericType>>();
			auto result2 = result[1]->as<SGMatrix<NumericType>>();

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
}

TYPED_TEST(GraphTest, vector_vector_dot)
{
	using NumericType = TypeParam;

	if constexpr (std::is_same_v<TypeParam, bool>)
		return;
	else
	{
		SGVector<NumericType> X1{1, 2, 3};
		SGVector<NumericType> X2{4, 5, 6};
		NumericType expected_result = 32;

		auto A = make_shared<node::Input>(
		    Shape{3}, get_enum_from_type<NumericType>::type);
		auto B = make_shared<node::Input>(
		    Shape{3}, get_enum_from_type<NumericType>::type);

		auto output = make_shared<node::Dot>(A, B);

		auto graph = make_shared<Graph>(
		    vector{A, B}, vector<shared_ptr<node::Node>>{output});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			vector<shared_ptr<Tensor>> result = graph->evaluate(
			    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});

			auto result1 = result[0]->as<SGVector<NumericType>>();

			EXPECT_EQ(result1[0], expected_result);
		}
	}
}

TYPED_TEST(GraphTest, matrix_vector_dot)
{
	using NumericType = TypeParam;

	if constexpr (std::is_same_v<TypeParam, bool>)
		return;
	else
	{
		SGMatrix<NumericType> X1{{1, 3, 5}, {2, 4, 6}};
		SGVector<NumericType> X2{1, 2};
		SGVector<NumericType> expected_result = {5, 11, 17};

		auto A = make_shared<node::Input>(
		    Shape{3, 2}, get_enum_from_type<NumericType>::type);
		auto B = make_shared<node::Input>(
		    Shape{2}, get_enum_from_type<NumericType>::type);

		auto output = make_shared<node::Dot>(A, B);

		auto graph = make_shared<Graph>(
		    vector{A, B}, vector<shared_ptr<node::Node>>{output});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			vector<shared_ptr<Tensor>> result = graph->evaluate(
			    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});

			auto result1 = result[0]->as<SGVector<NumericType>>();

			for (const auto& [expected_i, result_i] :
			     zip_iterator(expected_result, result1))
			{
				EXPECT_EQ(expected_i, result_i);
			}
		}
	}
}

TYPED_TEST(GraphTest, vector_matrix_dot)
{
	using NumericType = TypeParam;

	if constexpr (std::is_same_v<TypeParam, bool>)
		return;
	else
	{
		SGVector<NumericType> X1{1, 3, 5};
		SGMatrix<NumericType> X2{{1, 3, 5}, {2, 4, 6}};
		SGVector<NumericType> expected_result{35, 44};

		auto A = make_shared<node::Input>(
		    Shape{3}, get_enum_from_type<NumericType>::type);
		auto B = make_shared<node::Input>(
		    Shape{3, 2}, get_enum_from_type<NumericType>::type);

		auto output = make_shared<node::Dot>(A, B);

		auto graph = make_shared<Graph>(
		    vector{A, B}, vector<shared_ptr<node::Node>>{output});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			vector<shared_ptr<Tensor>> result = graph->evaluate(
			    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});

			auto result1 = result[0]->as<SGVector<NumericType>>();

			for (const auto& [expected_i, result_i] :
			     zip_iterator(expected_result, result1))
			{
				EXPECT_EQ(expected_i, result_i);
			}
		}
	}
}

TYPED_TEST(GraphTest, matrix_matrix_dot1)
{
	using NumericType = TypeParam;

	if constexpr (std::is_same_v<TypeParam, bool>)
		return;
	else
	{
		SGMatrix<NumericType> X1{{1, 3, 5}, {2, 4, 6}};
		SGMatrix<NumericType> X2{{1, 2}, {3, 4}, {5, 6}};

		SGMatrix<NumericType> expected_result = {
		    {5, 11, 17}, {11, 25, 39}, {17, 39, 61}};

		auto A = make_shared<node::Input>(
		    Shape{3, 2}, get_enum_from_type<NumericType>::type);
		auto B = make_shared<node::Input>(
		    Shape{2, 3}, get_enum_from_type<NumericType>::type);

		auto output = make_shared<node::Dot>(A, B);

		auto graph = make_shared<Graph>(
		    vector{A, B}, vector<shared_ptr<node::Node>>{output});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			vector<shared_ptr<Tensor>> result = graph->evaluate(
			    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});

			auto result1 = result[0]->as<SGMatrix<NumericType>>();

			for (const auto& [expected_i, result_i] :
			     zip_iterator(expected_result, result1))
			{
				EXPECT_EQ(expected_i, result_i);
			}
		}
	}
}

TYPED_TEST(GraphTest, matrix_matrix_dot2)
{
	using NumericType = TypeParam;

	if constexpr (
	    std::is_same_v<NumericType, bool> ||
	    std::is_same_v<NumericType, uint8_t> ||
	    std::is_same_v<NumericType, int8_t>)
		return;
	else
	{
		SGMatrix<NumericType> X1{{1, 3, 5}, {2, 4, 6}};
		SGMatrix<NumericType> X2{{1, 2}, {3, 4}, {5, 6}};

		SGMatrix<NumericType> expected_result1 = {
		    {5, 11, 17}, {11, 25, 39}, {17, 39, 61}};
		SGMatrix<NumericType> expected_result2 = {{123, 281, 439},
		                                          {156, 356, 556}};

		auto A = make_shared<node::Input>(
		    Shape{3, 2}, get_enum_from_type<NumericType>::type);
		auto B = make_shared<node::Input>(
		    Shape{2, 3}, get_enum_from_type<NumericType>::type);

		auto output1 = make_shared<node::Dot>(A, B);
		auto output2 = make_shared<node::Dot>(output1, A);

		auto graph = make_shared<Graph>(
		    vector{A, B}, vector<shared_ptr<node::Node>>{output1, output2});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			vector<shared_ptr<Tensor>> result = graph->evaluate(
			    vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});

			auto result1 = result[0]->as<SGMatrix<NumericType>>();
			auto result2 = result[1]->as<SGMatrix<NumericType>>();

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
}

TYPED_TEST(GraphTest, simple_perceptron_inference)
{
	using NumericType = TypeParam;

	if constexpr (std::is_same_v<TypeParam, bool>)
		return;
	else
	{
		SGMatrix<NumericType> features{{1, 3, 5}, {2, 4, 6}};
		SGVector<NumericType> weights{1, 2};
		NumericType bias = 1;
		SGVector<NumericType> expected_result = {6, 12, 18};

		auto X = make_shared<node::Input>(
		    Shape{Shape::Dynamic, 2}, get_enum_from_type<NumericType>::type);
		auto w = make_shared<node::Input>(
		    Shape{2}, get_enum_from_type<NumericType>::type);
		auto b = make_shared<node::Input>(
		    Shape{}, get_enum_from_type<NumericType>::type);

		auto prediction = make_shared<node::Dot>(X, w) + b;

		auto graph = make_shared<Graph>(
		    vector{X, w, b}, vector<shared_ptr<node::Node>>{prediction});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			vector<shared_ptr<Tensor>> result = graph->evaluate(vector{
			    make_shared<Tensor>(features), make_shared<Tensor>(weights),
			    make_shared<Tensor>(bias)});

			auto result1 = result[0]->as<SGVector<NumericType>>();

			for (const auto& [expected_i, result_i] :
			     zip_iterator(expected_result, result1))
			{
				EXPECT_EQ(expected_i, result_i);
			}
		}
	}
}
