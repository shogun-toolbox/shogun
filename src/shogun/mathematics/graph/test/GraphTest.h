#include <gtest/gtest.h>

#include <shogun/mathematics/graph/GraphExecutor.h>
#include <shogun/util/zip_iterator.h>

#include <tuple>

template <typename T>
class GraphTest : public ::testing::Test
{
protected:
	GraphTest() : m_backends(shogun::graph::available_backends())
	{
	}

	void test_binary_op_results(
	    const std::shared_ptr<shogun::graph::Graph>& graph,
	    const shogun::SGVector<T>& X1, const shogun::SGVector<T>& X2,
	    const shogun::SGVector<T>& expected_result1,
	    const shogun::SGVector<T>& expected_result2)
	{
		for (auto&& backend : this->m_backends)
		{
			graph->build(backend);

			std::vector<std::shared_ptr<shogun::graph::Tensor>> result =
			    graph->evaluate(
			        std::vector{std::make_shared<shogun::graph::Tensor>(X1),
			                    std::make_shared<shogun::graph::Tensor>(X2)});

			auto result1 = result[0]->template as<shogun::SGVector<T>>();
			auto result2 = result[1]->template as<shogun::SGVector<T>>();

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
    bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t,
    uint64_t, float32_t, float64_t>;

TYPED_TEST_CASE(GraphTest, GraphTypes);
