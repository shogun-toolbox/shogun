#include <gtest/gtest.h>
#include <shogun/io/SGIO.h>
#include <stdexcept>

using namespace shogun;

TEST(SGIO, exception)
{
	EXPECT_THROW(error("Error"), ShogunException);
	EXPECT_THROW(
	    error<std::invalid_argument>("Error"), std::invalid_argument);
	EXPECT_THROW(require(0, "Error"), ShogunException);
	EXPECT_THROW(
	    require<std::invalid_argument>(0, "Error"), std::invalid_argument);
}
