#include <shogun/lib/ParameterObserverScalar.h>
#include <vector>
#include <gtest/gtest.h>

using namespace shogun;

std::vector<std::string> test_params = {"a", "b", "c", "d"};

TEST(ParameterObserverScalar, filter_empty)
{
    ParameterObserverScalar tmp;
    EXPECT_TRUE(tmp.filter("a"));
}

TEST(ParameterObserverScalar, filter_found)
{
    ParameterObserverScalar tmp {test_params};
    EXPECT_TRUE(tmp.filter("a"));
    EXPECT_TRUE(tmp.filter("b"));
    EXPECT_TRUE(tmp.filter("c"));
    EXPECT_TRUE(tmp.filter("d"));
}

TEST(ParameterObserverScalar, filter_not_found)
{
    ParameterObserverScalar tmp {test_params};
    EXPECT_FALSE(tmp.filter("k"));
}
