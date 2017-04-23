/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2016 Sanuj Sharma
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 */
#include <shogun/base/SGObject.h>
#include <shogun/lib/any.h>
#include <gtest/gtest.h>
#include <shogun/lib/config.h>
#include <stdexcept>

using namespace shogun;

TEST(Any, as)
{
	int32_t integer = 10;
	auto any = Any(integer);
	EXPECT_EQ(any.as<int32_t>(), integer);
	EXPECT_THROW(any.as<float64_t>(), std::logic_error);
}

TEST(Any, same_type)
{
	int32_t integer = 10;
	auto any = Any(integer);
	EXPECT_EQ(any.same_type<int32_t>(), true);
	EXPECT_EQ(any.same_type<float64_t>(), false);
}

TEST(Any, empty)
{
	int32_t integer = 10;
	auto any = Any(integer);
	auto empty_any = Any();
	EXPECT_EQ(any.empty(), false);
	EXPECT_EQ(empty_any.empty(), true);
}

TEST(Any, same_type_fallback)
{
	int32_t integer = 10;
	auto any = Any(integer);
	EXPECT_EQ(any.same_type_fallback<int32_t>(), true);
	EXPECT_EQ(any.same_type_fallback<float64_t>(), false);
}

// TODO(lisitsyn): Windows being unstable here, unclear yet
#ifndef _MSC_VER
TEST(Any, erase_type)
{
	int32_t integer = 10;
	float64_t float_pt = 10.0;
	auto int_any = Any(integer);
	auto empty_any = Any();
	auto float_any = Any(float_pt);
	auto erased_int = erase_type(integer);
	EXPECT_EQ(erased_int, int_any);
	EXPECT_NE(erased_int, empty_any);
	EXPECT_NE(erased_int, float_any);
}
#endif

TEST(Any, recall_type)
{
	int32_t integer = 10;
	auto any = Any(integer);
	auto empty_any = Any();
	EXPECT_EQ(recall_type<int32_t>(any), integer);
	EXPECT_THROW(recall_type<float64_t>(any), std::logic_error);
	EXPECT_THROW(recall_type<int32_t>(empty_any), std::logic_error);
}

TEST(Any, erase_type_non_owning)
{
	int32_t integer = 10;
	auto any = erase_type_non_owning(&integer);
	EXPECT_EQ(recall_type<int32_t>(any), integer);
	integer++;
	EXPECT_EQ(recall_type<int32_t>(any), integer);
}

TEST(Any, mixing_policies)
{
	int32_t integer = 10;
	auto owning_any = erase_type(integer);
	auto non_owning_any = erase_type_non_owning(&integer);
	EXPECT_THROW(owning_any = non_owning_any, std::logic_error);
	EXPECT_THROW(non_owning_any = owning_any, std::logic_error);
}
