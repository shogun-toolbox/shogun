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
#include <gtest/gtest.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/any.h>
#include <shogun/lib/config.h>
#include <stdexcept>

using namespace shogun;

struct Simple
{
public:
	bool equals(const Simple* other) const
	{
		return true;
	}
	bool equals(const Simple& other) const
	{
		return true;
	}
};

TEST(Any, as)
{
	int32_t integer = 10;
	auto any = erase_type(integer);
	EXPECT_EQ(any.as<int32_t>(), integer);
	EXPECT_THROW(any.as<float64_t>(), std::logic_error);
}

TEST(Any, same_type)
{
	int32_t integer = 10;
	auto any = erase_type(integer);
	EXPECT_EQ(any.same_type<int32_t>(), true);
	EXPECT_EQ(any.same_type<float64_t>(), false);
}

TEST(Any, empty)
{
	int32_t integer = 10;
	auto any = erase_type(integer);
	auto empty_any = Any();
	EXPECT_EQ(any.empty(), false);
	EXPECT_EQ(empty_any.empty(), true);
}

TEST(Any, same_type_fallback)
{
	int32_t integer = 10;
	auto any = erase_type(integer);
	EXPECT_EQ(any.same_type_fallback<int32_t>(), true);
	EXPECT_EQ(any.same_type_fallback<float64_t>(), false);
}

// TODO(lisitsyn): Windows being unstable here, unclear yet
#ifndef _MSC_VER
TEST(Any, erase_type)
{
	int32_t integer = 10;
	float64_t float_pt = 10.0;
	auto int_any = erase_type(integer);
	auto empty_any = Any();
	auto float_any = erase_type(float_pt);
	auto erased_int = erase_type(integer);
	EXPECT_EQ(erased_int, int_any);
	EXPECT_NE(erased_int, empty_any);
	EXPECT_NE(erased_int, float_any);
}
#endif

TEST(Any, recall_type)
{
	int32_t integer = 10;
	auto any = erase_type(integer);
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

TEST(Any, assign_non_owning)
{
	int32_t integer = 10;
	Any any;
	any = erase_type_non_owning(&integer);
	EXPECT_EQ(any.as<int32_t>(), integer);
}

TEST(Any, assign_into_non_owning)
{
	int32_t integer = 10;
	int32_t other = 42;
	auto any = erase_type_non_owning(&integer);
	EXPECT_EQ(any.as<int32_t>(), integer);
	any = erase_type(other);
	EXPECT_EQ(any.as<int32_t>(), other);
	EXPECT_EQ(integer, other);
}

TEST(Any, assign_non_owning_into_non_owning_then_owning)
{
	int32_t first = 111;
	int32_t second = 222;
	int32_t third = 333;
	auto first_any = erase_type_non_owning(&first);
	auto second_any = erase_type_non_owning(&second);
	EXPECT_EQ(first_any.as<int32_t>(), first);
	EXPECT_EQ(second_any.as<int32_t>(), second);
	first_any = second_any;
	EXPECT_EQ(first_any.as<int32_t>(), second);
	EXPECT_EQ(second_any.as<int32_t>(), second);
	first_any = erase_type(third);
	EXPECT_EQ(first_any.as<int32_t>(), third);
	EXPECT_EQ(second_any.as<int32_t>(), third);
}

TEST(Any, assign_non_owning_into_non_owning_then_non_owning)
{
	int32_t first = 111;
	int32_t second = 222;
	int32_t third = 333;
	auto first_any = erase_type_non_owning(&first);
	auto second_any = erase_type_non_owning(&second);
	EXPECT_EQ(first_any.as<int32_t>(), first);
	EXPECT_EQ(second_any.as<int32_t>(), second);
	first_any = second_any;
	EXPECT_EQ(first_any.as<int32_t>(), second);
	EXPECT_EQ(second_any.as<int32_t>(), second);
	first_any = erase_type_non_owning(&third);
	EXPECT_EQ(first_any.as<int32_t>(), third);
	EXPECT_EQ(second_any.as<int32_t>(), second);
}

TEST(Any, assign_wrong_type_into_owning)
{
	int32_t integer = 10;
	auto any = erase_type(integer);
	EXPECT_THROW(any = erase_type(3.14), std::logic_error);
}

TEST(Any, assign_wrong_type_into_non_owning)
{
	int32_t integer = 10;
	auto any = erase_type_non_owning(&integer);
	EXPECT_THROW(any = erase_type(3.14), std::logic_error);
}

TEST(Any, compare_owning_and_non_owning)
{
	int32_t integer = 10;
	auto owning = erase_type(integer);
	auto non_owning = erase_type_non_owning(&integer);
	EXPECT_EQ(owning, non_owning);
}

TEST(Any, compare_non_owning_and_non_owning)
{
	int32_t integer = 10;
	auto first_non_owning = erase_type_non_owning(&integer);
	auto second_non_owning = erase_type_non_owning(&integer);
	EXPECT_EQ(first_non_owning, second_non_owning);
}

TEST(Any, compare_different_types)
{
	int32_t an_integer = 10;
	float a_float = 10;
	EXPECT_NE(erase_type(an_integer), erase_type(a_float));
	EXPECT_NE(erase_type(a_float), erase_type(an_integer));
}

TEST(Any, compare_different_types_non_owning)
{
	int32_t an_integer = 10;
	float a_float = 10;
	EXPECT_NE(
	    erase_type_non_owning(&an_integer), erase_type_non_owning(&a_float));
	EXPECT_NE(
	    erase_type_non_owning(&a_float), erase_type_non_owning(&an_integer));
}

TEST(Any, copy_owning)
{
	int32_t integer = 10;
	int32_t other = 12;
	auto any(erase_type(integer));
	EXPECT_EQ(any.as<int32_t>(), integer);
	any = erase_type(other);
	EXPECT_EQ(any.as<int32_t>(), other);
}

TEST(Any, copy_non_owning)
{
	int32_t integer = 10;
	int32_t other = 12;
	auto any(erase_type_non_owning(&integer));
	EXPECT_EQ(any.as<int32_t>(), integer);
	any = erase_type(other);
	EXPECT_EQ(any.as<int32_t>(), other);
	EXPECT_EQ(integer, other);
}

TEST(Any, type_info)
{
	int32_t integer = 10;
	auto any = erase_type(integer);
	EXPECT_EQ(any.type_info().hash_code(), typeid(integer).hash_code());
}

TEST(Any, store_in_map)
{
	int32_t integer = 10;
	std::map<std::string, Any> map;
	map["something"] = erase_type(integer);
	EXPECT_EQ(map.at("something").as<int32_t>(), integer);
	integer = 13;
	EXPECT_NE(map.at("something").as<int32_t>(), integer);
}

TEST(Any, store_non_owning_in_map)
{
	int32_t integer = 10;
	std::map<std::string, Any> map;
	map["something"] = erase_type_non_owning(&integer);
	EXPECT_EQ(map.at("something").as<int32_t>(), integer);
	integer = 13;
	EXPECT_EQ(map.at("something").as<int32_t>(), integer);
}

TEST(Any, equals_int)
{
	int32_t a = 1;
	int32_t b = 1;
	EXPECT_EQ(erase_type(a), erase_type(b));
	EXPECT_EQ(erase_type(b), erase_type(a));
}

TEST(Any, equals_pointer)
{
	Simple* a = new Simple;
	Simple* b = new Simple;
	EXPECT_EQ(erase_type(a), erase_type(b));
	EXPECT_EQ(erase_type(b), erase_type(a));
	delete a;
	delete b;
}

TEST(Any, equals_value)
{
	Simple a;
	Simple b;
	EXPECT_EQ(erase_type(a), erase_type(b));
	EXPECT_EQ(erase_type(b), erase_type(a));
}
