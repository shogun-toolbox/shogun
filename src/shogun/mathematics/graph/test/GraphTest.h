/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <gtest/gtest.h>

#include <shogun/mathematics/graph/GraphExecutor.h>
#include <shogun/util/zip_iterator.h>

template <typename T>
class GraphTest : public ::testing::Test
{
protected:
	GraphTest() : m_backends(shogun::graph::available_backends())
	{
	}

	void test_binary_op_results(
	    const std::shared_ptr<shogun::graph::Graph>& graph,
	    const shogun::SGVector<typename T::c_type>& X1,
	    const shogun::SGVector<typename T::c_type>& X2,
	    const shogun::SGVector<typename T::c_type>& expected_result1,
	    const shogun::SGVector<typename T::c_type>& expected_result2)
	{
		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			std::vector<std::shared_ptr<shogun::graph::Tensor>> result =
			    graph->evaluate(
			        std::vector{std::make_shared<shogun::graph::Tensor>(X1),
			                    std::make_shared<shogun::graph::Tensor>(X2)});

			auto result1 =
			    result[0]->template as<shogun::SGVector<typename T::c_type>>();
			auto result2 =
			    result[1]->template as<shogun::SGVector<typename T::c_type>>();

			for (const auto& [expected_i, result_i] :
			     shogun::zip_iterator(expected_result1, result1))
			{
				EXPECT_EQ(expected_i, result_i);
			}

			for (const auto& [expected_i, result_i] :
			     shogun::zip_iterator(expected_result2, result2))
			{
				EXPECT_EQ(expected_i, result_i);
			}
		}
	}

	std::set<GRAPH_BACKEND> m_backends;
};

using GraphTypes = ::testing::Types<
    shogun::graph::BooleanType, shogun::graph::UInt8Type,
    shogun::graph::Int8Type, shogun::graph::UInt16Type,
    shogun::graph::Int16Type, shogun::graph::UInt32Type,
    shogun::graph::Int32Type, shogun::graph::Int64Type,
    shogun::graph::UInt64Type, shogun::graph::Float32Type,
    shogun::graph::Float64Type>;

TYPED_TEST_CASE(GraphTest, GraphTypes);
