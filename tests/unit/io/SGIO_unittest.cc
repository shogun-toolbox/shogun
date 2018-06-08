#include <gtest/gtest.h>
#include <shogun/io/SGIO.h>
#include <stdexcept>

using namespace shogun;

TEST(SGIO, exception)
{
	EXPECT_THROW(SG_SERROR("Error"), ShogunException);
	EXPECT_THROW(
	    SG_STHROW(std::invalid_argument, "Error"), std::invalid_argument);
	EXPECT_THROW(REQUIRE(0, "Error"), ShogunException);
	EXPECT_THROW(
	    REQUIRE_E(0, std::invalid_argument, "Error"), std::invalid_argument);
}
