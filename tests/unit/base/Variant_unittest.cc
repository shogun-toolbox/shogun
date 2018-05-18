#include <gtest/gtest.h>
#include <shogun/base/variant.h>
#include <string>

using namespace shogun;

TEST(Variant, get)
{
	auto s = "123";
	variant<int, std::string> v(s);
	EXPECT_EQ(s, get<std::string>(v));
	EXPECT_THROW(get<int>(v), bad_variant_access);
}
