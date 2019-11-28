#include <gtest/gtest.h>

#include "sg_gtest_utilities.h"

#include <shogun/lib/sg_types.h>
#include <shogun/util/converters.h>

#include <cstdint>
#include <iostream>
#include <limits>

using namespace shogun;

template <typename T>
class Converters : public ::testing::Test
{
};

SG_TYPED_TEST_CASE(Converters, sg_non_complex_types);

TYPED_TEST(Converters, canDoCorrectConvertionOfUnsignedArg)
{
	TypeParam result = 0;
	uint64_t value = 120ull;

	ASSERT_NO_THROW(result = utils::safe_convert<TypeParam>(value));
	EXPECT_EQ(static_cast<TypeParam>(value), result);
}

TYPED_TEST(Converters, canDoCorrectConvertionOfSignedArg)
{
	TypeParam result = 0;
	int64_t value = 120ull;

	ASSERT_NO_THROW(result = utils::safe_convert<TypeParam>(value));
	EXPECT_EQ(static_cast<TypeParam>(value), result);
}

TYPED_TEST(Converters, cannotConvertOutOfRangeOfSignedArg)
{
	// skip testing of long double since it cannot be overflowed
	// this way
	if (typeid(TypeParam) != typeid(long double))
	{
		auto upperLimit = std::numeric_limits<long double>::max();
		auto lowerLimit = std::numeric_limits<long double>::lowest();

		EXPECT_THROW(
		    utils::safe_convert<TypeParam>(upperLimit), std::overflow_error);
		EXPECT_THROW(
		    utils::safe_convert<TypeParam>(lowerLimit), std::overflow_error);
	}
}

TYPED_TEST(Converters, cannotConvertOutOfRangeOfUnsignedArg)
{
	// skip testing of real types, unsigned long
	// since they cannot be overflowed this way
	if (std::is_integral<TypeParam>::value &&
	    typeid(TypeParam) != typeid(unsigned long long) &&
	    typeid(TypeParam) != typeid(unsigned long))
	{
		auto upperLimit = std::numeric_limits<unsigned long long>::max();

		EXPECT_THROW(
		    utils::safe_convert<TypeParam>(upperLimit), std::overflow_error);
	}
}

TYPED_TEST(Converters, canConvertBool)
{
	TypeParam result{0};
	ASSERT_NO_THROW(result = utils::safe_convert<TypeParam>(true));
	EXPECT_EQ(static_cast<TypeParam>(1), result);

	ASSERT_NO_THROW(result = utils::safe_convert<TypeParam>(false));
	EXPECT_EQ(static_cast<TypeParam>(0), result);
}

// two tests below are skipped for non-integer types
TYPED_TEST(Converters, cannotConvertFloatInfinityToInteger)
{
	if (std::is_integral<TypeParam>::value)
	{
		EXPECT_THROW(
		    utils::safe_convert<TypeParam>(
		        std::numeric_limits<float>::infinity()),
		    std::overflow_error);
	}
}

TYPED_TEST(Converters, cannotConvertFloatNaNToInteger)
{
	if (std::is_integral<TypeParam>::value)
	{
		EXPECT_THROW(
		    utils::safe_convert<TypeParam>(
		        std::numeric_limits<float>::quiet_NaN()),
		    std::overflow_error);
	}
}

// the tests below are skipped for integer types
TYPED_TEST(Converters, canConvertFloatInfinityToReal)
{
	if (!std::is_integral<TypeParam>::value)
	{
		TypeParam result{0};
		ASSERT_NO_THROW(
		    result = utils::safe_convert<TypeParam>(
		        std::numeric_limits<float>::infinity()));
		EXPECT_TRUE(std::isinf(result));
	}
}

TYPED_TEST(Converters, canConvertFloatNaNToReal)
{
	if (!std::is_integral<TypeParam>::value)
	{
		TypeParam result{0};
		ASSERT_NO_THROW(
		    result = utils::safe_convert<TypeParam>(
		        std::numeric_limits<float>::quiet_NaN()));
		EXPECT_TRUE(std::isnan(result));
	}
}
