/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Tommy Nguyen
 *
 */
#include "sg_gtest_utilities.h"

#include <cmath>
#include <gtest/gtest.h>
#include <shogun/util/converters.h>
#include <vector>

using namespace shogun;

template <typename T>
class SafeConvertRealTypesTest : public ::testing::Test
{
};

template <typename T>
class SafeConvertNonComplexTest : public ::testing::Test
{
};

template <typename T>
class SafeConvertNonfiniteTest : public ::testing::Test
{
};

SG_TYPED_TEST_CASE(SafeConvertNonComplexTest, sg_non_complex_types);
SG_TYPED_TEST_CASE(SafeConvertRealTypesTest, sg_real_types);
SG_TYPED_TEST_CASE(SafeConvertNonfiniteTest, sg_real_types);

TYPED_TEST(SafeConvertNonfiniteTest, identity_convert)
{
	const TypeParam iNAN = NAN;
	const TypeParam iINF = INFINITY;
	const TypeParam iMinusINF = -INFINITY;

	EXPECT_TRUE(std::isnan(static_cast<TypeParam>(utils::safe_convert<TypeParam>(iNAN))));
	EXPECT_TRUE(std::isinf(static_cast<TypeParam>(utils::safe_convert<TypeParam>(iINF))));
	EXPECT_TRUE(std::isinf(static_cast<TypeParam>(utils::safe_convert<TypeParam>(iMinusINF))));
}

TYPED_TEST(SafeConvertRealTypesTest, successfully_converts)
{
	const TypeParam f = 1.0f;
	const TypeParam d = 2.0;
	const TypeParam ld = 3.0l;
	const TypeParam ui = 4u;

	EXPECT_EQ(utils::safe_convert<TypeParam>(f), f);
	EXPECT_EQ(utils::safe_convert<TypeParam>(d), d);
	EXPECT_EQ(utils::safe_convert<TypeParam>(ld), ld);

	// Handle the case where std::make_unsigned would be called for a real
	// type.
	EXPECT_EQ(utils::safe_convert<TypeParam>(ui), ui);
}

TYPED_TEST(SafeConvertNonComplexTest, successfully_converts)
{
	const TypeParam i = 1;
	const TypeParam ui = 1u;
	const TypeParam ul = 1ul;
	const TypeParam ull = 1ull;
	const TypeParam il = 1l;
	const TypeParam ill = 1ll;

	EXPECT_EQ(utils::safe_convert<TypeParam>(i), i);
	EXPECT_EQ(utils::safe_convert<TypeParam>(ui), ui);
	EXPECT_EQ(utils::safe_convert<TypeParam>(ul), ul);
	EXPECT_EQ(utils::safe_convert<TypeParam>(ull), ull);
	EXPECT_EQ(utils::safe_convert<TypeParam>(il), il);
	EXPECT_EQ(utils::safe_convert<TypeParam>(ill), ill);
}

TEST(SafeConvertOutOfRange, out_of_range)
{
	// std::is_signed<I>::value && std::is_signed<J>::value
	EXPECT_THROW(
	    utils::safe_convert<std::int32_t>(
	        std::numeric_limits<std::int64_t>::max()),
	    std::overflow_error);
	// std::is_signed<I>::value && std::is_unsigned<J>::value
	EXPECT_THROW(
	    utils::safe_convert<std::int32_t>(
	        static_cast<std::uint64_t>(
	            std::numeric_limits<std::int64_t>::max())),
	    std::overflow_error);
	// std::is_unsigned<I>::value && std::is_signed<J>::value
	EXPECT_THROW(utils::safe_convert<std::uint32_t>(-1), std::overflow_error);
	// std::is_unsigned<I>::value && std::is_unsigned<J>::value
	EXPECT_THROW(
	    utils::safe_convert<std::uint32_t>(
	        std::numeric_limits<std::uint64_t>::max()),
	    std::overflow_error);
}
