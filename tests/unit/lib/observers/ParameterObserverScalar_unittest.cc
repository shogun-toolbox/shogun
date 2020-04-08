/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */
#include <gtest/gtest.h>

#include <shogun/lib/config.h>
#ifdef HAVE_TFLOGGER

#include <shogun/lib/observers/ParameterObserverScalar.h>
#include <vector>

std::vector<std::string> test_params = {"a", "b", "c", "d"};

using namespace shogun;

TEST(ParameterObserverScalar, filter_empty)
{
	ParameterObserverScalar tmp;
	EXPECT_TRUE(tmp.filter("a"));
}

TEST(ParameterObserverScalar, filter_found)
{
	ParameterObserverScalar tmp{test_params};
	EXPECT_TRUE(tmp.filter("a"));
	EXPECT_TRUE(tmp.filter("b"));
	EXPECT_TRUE(tmp.filter("c"));
	EXPECT_TRUE(tmp.filter("d"));
}

TEST(ParameterObserverScalar, filter_not_found)
{
	ParameterObserverScalar tmp{test_params};
	EXPECT_FALSE(tmp.filter("k"));
}

#endif // HAVE_TFLOGGER
