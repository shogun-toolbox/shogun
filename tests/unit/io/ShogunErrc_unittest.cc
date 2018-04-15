#include <gtest/gtest.h>
#include <shogun/io/ShogunErrc.h>

using namespace shogun::io;

TEST(ShogunErrc, error_conditions)
{
	auto ec = make_error_condition(ShogunErrc::OutOfRange);
	ASSERT_TRUE(is_out_of_range(ec));
}
