#include <gtest/gtest.h>
#include <shogun/io/SGIO.h>
#include <stdexcept>

using namespace shogun;

TEST(SGIO, exception)
{
	EXPECT_THROW(SG_ERROR("Error"), ShogunException);
	EXPECT_THROW(
	    SG_THROW(std::invalid_argument, "Error"), std::invalid_argument);
	EXPECT_THROW(REQUIRE(0, "Error"), ShogunException);
	EXPECT_THROW(
	    REQUIRE_E(0, std::invalid_argument, "Error"), std::invalid_argument);
}
