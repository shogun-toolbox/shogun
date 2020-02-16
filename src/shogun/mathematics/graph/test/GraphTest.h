#include <gtest/gtest.h>

#include <shogun/mathematics/graph/GraphExecutor.h>

#include <tuple>

template <typename T>
class GraphTest : public ::testing::Test
{
protected:
	GraphTest() : m_backends(shogun::graph::available_backends())
	{
	}

	std::set<GRAPH_BACKEND> m_backends;
};

using GraphTypes = ::testing::Types<float32_t, float64_t>;
TYPED_TEST_CASE(GraphTest, GraphTypes);
