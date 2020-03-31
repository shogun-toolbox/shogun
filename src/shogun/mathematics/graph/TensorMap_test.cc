#include <gtest/gtest.h>

#include <shogun/mathematics/graph/TensorMap.h>

#include <vector>

using namespace shogun::graph;
using namespace std;

TEST(TensorMapTest, vector)
{
	vector<double> input{1.0, 2.0, 3.0, 4.0};
	TensorMap mapped(input.data(), Shape{4});

	EXPECT_EQ(input.size(), mapped.size());
}

TEST(TensorMapTest, matrix)
{
	vector<double> input{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	TensorMap mapped(input.data(), Shape{4, 2});

	EXPECT_EQ(input.size(), mapped.size());
}
