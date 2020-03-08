#include <gtest/gtest.h>

#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/nodes/Input.h>
#include <shogun/mathematics/graph/nodes/MatMul.h>

#include "../test/GraphTest.h"

using namespace shogun;
using namespace shogun::graph;
using namespace std;

TYPED_TEST(GraphTest, scalar_matmul)
{
	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;
	else
	{
		auto A = make_shared<node::Input>(
		    Shape{3}, TypeParam::type_id);
		auto B = make_shared<node::Input>(
		    Shape{}, TypeParam::type_id);

		// MatMul doesn't work with scalar, unlike Dot. See MatMul docs.
		EXPECT_THROW(make_shared<node::MatMul>(A, B), ShogunException);
	}
}

TYPED_TEST(GraphTest, vector_vector_matmul)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;
	else
	{
		SGVector<NumericType> X1{1, 2, 3};
		SGVector<NumericType> X2{4, 5, 6};
		NumericType expected_result = 32;

		auto A = make_shared<node::Input>(
		    Shape{3}, TypeParam::type_id);
		auto B = make_shared<node::Input>(
		    Shape{3}, TypeParam::type_id);

		auto output = make_shared<node::MatMul>(A, B);

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

TYPED_TEST(GraphTest, matrix_vector_matmul)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;
	else
	{
		SGMatrix<NumericType> X1{{1, 3, 5}, {2, 4, 6}};
		SGVector<NumericType> X2{1, 2};
		SGVector<NumericType> expected_result = {5, 11, 17};

		auto A = make_shared<node::Input>(
		    Shape{3, 2}, TypeParam::type_id);
		auto B = make_shared<node::Input>(
		    Shape{2}, TypeParam::type_id);

		auto output = make_shared<node::MatMul>(A, B);

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

TYPED_TEST(GraphTest, vector_matrix_matmul)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;
	else
	{
		SGVector<NumericType> X1{1, 3, 5};
		SGMatrix<NumericType> X2{{1, 3, 5}, {2, 4, 6}};
		SGVector<NumericType> expected_result{35, 44};

		auto A = make_shared<node::Input>(
		    Shape{3}, TypeParam::type_id);
		auto B = make_shared<node::Input>(
		    Shape{3, 2}, TypeParam::type_id);

		auto output = make_shared<node::MatMul>(A, B);

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

TYPED_TEST(GraphTest, matrix_matrix_matmul1)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (std::is_same_v<TypeParam, BooleanType>)
		return;
	else
	{
		SGMatrix<NumericType> X1{{1, 3, 5}, {2, 4, 6}};
		SGMatrix<NumericType> X2{{1, 2}, {3, 4}, {5, 6}};

		SGMatrix<NumericType> expected_result = {
		    {5, 11, 17}, {11, 25, 39}, {17, 39, 61}};

		auto A = make_shared<node::Input>(
		    Shape{3, 2}, TypeParam::type_id);
		auto B = make_shared<node::Input>(
		    Shape{2, 3}, TypeParam::type_id);

		auto output = make_shared<node::MatMul>(A, B);

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

TYPED_TEST(GraphTest, matrix_matrix_matmul2)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (
	    std::is_same_v<TypeParam, BooleanType> ||
	    std::is_same_v<TypeParam, UInt8Type> ||
	    std::is_same_v<TypeParam, Int8Type>)
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
		    Shape{3, 2}, TypeParam::type_id);
		auto B = make_shared<node::Input>(
		    Shape{2, 3}, TypeParam::type_id);

		auto output1 = make_shared<node::MatMul>(A, B);
		auto output2 = make_shared<node::MatMul>(output1, A);

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

TYPED_TEST(GraphTest, matrix_matrixT_matmul)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (
	    std::is_same_v<TypeParam, BooleanType> ||
	    std::is_same_v<TypeParam, UInt8Type> ||
	    std::is_same_v<TypeParam, Int8Type>)
		return;
	else
	{
		SGMatrix<NumericType> X1{{1, 3, 5}, {2, 4, 6}};

		SGMatrix<NumericType> expected_result1 = {
		    {5, 11, 17}, {11, 25, 39}, {17, 39, 61}};
		SGMatrix<NumericType> expected_result2 = {{123, 281, 439},
		                                          {156, 356, 556}};

		auto A = make_shared<node::Input>(
		    Shape{3, 2}, TypeParam::type_id);

		auto output1 = make_shared<node::MatMul>(A, A, false, true);
		EXPECT_THROW(make_shared<node::MatMul>(A, A), ShogunException);
		auto output2 = make_shared<node::MatMul>(output1, A, true, false);

		auto graph = make_shared<Graph>(
		    vector{A}, vector<shared_ptr<node::Node>>{output1, output2});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			vector<shared_ptr<Tensor>> result =
			    graph->evaluate(vector{make_shared<Tensor>(X1)});

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

TYPED_TEST(GraphTest, matrixT_matrix_matmul)
{
	using NumericType = typename TypeParam::c_type;

	if constexpr (
	    std::is_same_v<TypeParam, BooleanType> ||
	    std::is_same_v<TypeParam, UInt8Type> ||
	    std::is_same_v<TypeParam, Int8Type>)
		return;
	else
	{
		SGMatrix<NumericType> X1{{1, 3, 5}, {2, 4, 6}};

		SGMatrix<NumericType> expected_result1 = {{35, 44}, {44, 56}};
		SGMatrix<NumericType> expected_result2 = {
		    {123, 156}, {281, 356}, {439, 556}};

		auto A = make_shared<node::Input>(
		    Shape{3, 2}, TypeParam::type_id);

		auto output1 = make_shared<node::MatMul>(A, A, true, false);
		EXPECT_THROW(make_shared<node::MatMul>(output1, A), ShogunException);
		auto output2 = make_shared<node::MatMul>(output1, A, false, true);

		auto graph = make_shared<Graph>(
		    vector{A}, vector<shared_ptr<node::Node>>{output1, output2});

		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			vector<shared_ptr<Tensor>> result =
			    graph->evaluate(vector{make_shared<Tensor>(X1)});

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
