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
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/any.h>
#include <shogun/lib/config.h>
#include <stdexcept>

using namespace shogun;

struct Simple
{
public:
	Simple* clone() const
	{
		Simple* copy = new Simple;
		copy->cloned = true;
		return copy;
	}
	bool equals(const Simple* other) const
	{
		return true;
	}
	bool equals(const Simple& other) const
	{
		return true;
	}

	bool cloned = false;
};

TEST(Any, as)
{
	int32_t integer = 10;
	auto any = make_any(integer);
	EXPECT_EQ(any.as<int32_t>(), integer);
	EXPECT_THROW(any.as<float64_t>(), std::logic_error);
}

TEST(Any, same_type)
{
	int32_t integer = 10;
	auto any = make_any(integer);
	EXPECT_EQ(any.same_type<int32_t>(), true);
	EXPECT_EQ(any.same_type<float64_t>(), false);
}

TEST(Any, empty)
{
	int32_t integer = 10;
	auto any = make_any(integer);
	auto empty_any = Any();
	EXPECT_EQ(any.empty(), false);
	EXPECT_EQ(empty_any.empty(), true);
}

TEST(Any, same_type_fallback)
{
	int32_t integer = 10;
	auto any = make_any(integer);
	EXPECT_EQ(any.same_type_fallback<int32_t>(), true);
	EXPECT_EQ(any.same_type_fallback<float64_t>(), false);
}

// TODO(lisitsyn): Windows being unstable here, unclear yet
#ifndef _MSC_VER
TEST(Any, make_any)
{
	int32_t integer = 10;
	float64_t float_pt = 10.0;
	auto int_any = make_any(integer);
	auto empty_any = Any();
	auto float_any = make_any(float_pt);
	auto erased_int = make_any(integer);
	EXPECT_EQ(erased_int, int_any);
	EXPECT_NE(erased_int, empty_any);
	EXPECT_NE(erased_int, float_any);
}
#endif

TEST(Any, any_cast)
{
	int32_t integer = 10;
	auto any = make_any(integer);
	auto empty_any = Any();
	EXPECT_EQ(any_cast<int32_t>(any), integer);
	EXPECT_THROW(any_cast<float64_t>(any), std::logic_error);
	EXPECT_THROW(any_cast<int32_t>(empty_any), std::logic_error);
}

TEST(Any, make_any_ref)
{
	int32_t integer = 10;
	auto any = make_any_ref(&integer);
	EXPECT_EQ(any_cast<int32_t>(any), integer);
	integer++;
	EXPECT_EQ(any_cast<int32_t>(any), integer);
}

TEST(Any, assign_non_owning)
{
	int32_t integer = 10;
	Any any;
	any = make_any_ref(&integer);
	EXPECT_EQ(any.as<int32_t>(), integer);
}

TEST(Any, assign_into_non_owning)
{
	int32_t integer = 10;
	int32_t other = 42;
	auto any = make_any_ref(&integer);
	EXPECT_EQ(any.as<int32_t>(), integer);
	any = make_any(other);
	EXPECT_EQ(any.as<int32_t>(), other);
	EXPECT_EQ(integer, other);
}

TEST(Any, assign_non_owning_into_non_owning_then_owning)
{
	int32_t first = 111;
	int32_t second = 222;
	int32_t third = 333;
	auto first_any = make_any_ref(&first);
	auto second_any = make_any_ref(&second);
	EXPECT_EQ(first_any.as<int32_t>(), first);
	EXPECT_EQ(second_any.as<int32_t>(), second);
	first_any = second_any;
	EXPECT_EQ(first_any.as<int32_t>(), second);
	EXPECT_EQ(second_any.as<int32_t>(), second);
	first_any = make_any(third);
	EXPECT_EQ(first_any.as<int32_t>(), third);
	EXPECT_EQ(second_any.as<int32_t>(), third);
}

TEST(Any, assign_non_owning_into_non_owning_then_non_owning)
{
	int32_t first = 111;
	int32_t second = 222;
	int32_t third = 333;
	auto first_any = make_any_ref(&first);
	auto second_any = make_any_ref(&second);
	EXPECT_EQ(first_any.as<int32_t>(), first);
	EXPECT_EQ(second_any.as<int32_t>(), second);
	first_any = second_any;
	EXPECT_EQ(first_any.as<int32_t>(), second);
	EXPECT_EQ(second_any.as<int32_t>(), second);
	first_any = make_any_ref(&third);
	EXPECT_EQ(first_any.as<int32_t>(), third);
	EXPECT_EQ(second_any.as<int32_t>(), second);
}

TEST(Any, assign_wrong_type_into_owning)
{
	int32_t integer = 10;
	auto any = make_any(integer);
	EXPECT_THROW(any = make_any(3.14), std::logic_error);
}

TEST(Any, assign_wrong_type_into_non_owning)
{
	int32_t integer = 10;
	auto any = make_any_ref(&integer);
	EXPECT_THROW(any = make_any(3.14), std::logic_error);
}

TEST(Any, compare_owning_and_non_owning)
{
	int32_t integer = 10;
	auto owning = make_any(integer);
	auto non_owning = make_any_ref(&integer);
	EXPECT_EQ(owning, non_owning);
}

TEST(Any, compare_non_owning_and_non_owning)
{
	int32_t integer = 10;
	auto first_non_owning = make_any_ref(&integer);
	auto second_non_owning = make_any_ref(&integer);
	EXPECT_EQ(first_non_owning, second_non_owning);
}

TEST(Any, compare_different_types)
{
	int32_t an_integer = 10;
	float a_float = 10;
	EXPECT_NE(make_any(an_integer), make_any(a_float));
	EXPECT_NE(make_any(a_float), make_any(an_integer));
}

TEST(Any, compare_different_types_non_owning)
{
	int32_t an_integer = 10;
	float a_float = 10;
	EXPECT_NE(make_any_ref(&an_integer), make_any_ref(&a_float));
	EXPECT_NE(make_any_ref(&a_float), make_any_ref(&an_integer));
}

TEST(Any, copy_owning)
{
	int32_t integer = 10;
	int32_t other = 12;
	auto any(make_any(integer));
	EXPECT_EQ(any.as<int32_t>(), integer);
	any = make_any(other);
	EXPECT_EQ(any.as<int32_t>(), other);
}

TEST(Any, copy_non_owning)
{
	int32_t integer = 10;
	int32_t other = 12;
	auto any(make_any_ref(&integer));
	EXPECT_EQ(any.as<int32_t>(), integer);
	any = make_any(other);
	EXPECT_EQ(any.as<int32_t>(), other);
	EXPECT_EQ(integer, other);
}

TEST(Any, type_info)
{
	int32_t integer = 10;
	auto any = make_any(integer);
	EXPECT_EQ(any.type_info().hash_code(), typeid(integer).hash_code());
}

TEST(Any, store_in_map)
{
	int32_t integer = 10;
	std::map<std::string, Any> map;
	map["something"] = make_any(integer);
	EXPECT_EQ(map.at("something").as<int32_t>(), integer);
	integer = 13;
	EXPECT_NE(map.at("something").as<int32_t>(), integer);
}

TEST(Any, store_non_owning_in_map)
{
	int32_t integer = 10;
	std::map<std::string, Any> map;
	map["something"] = make_any_ref(&integer);
	EXPECT_EQ(map.at("something").as<int32_t>(), integer);
	integer = 13;
	EXPECT_EQ(map.at("something").as<int32_t>(), integer);
}

TEST(Any, equals_int)
{
	int32_t a = 1;
	int32_t b = 1;
	EXPECT_EQ(make_any(a), make_any(b));
	EXPECT_EQ(make_any(b), make_any(a));
}

TEST(Any, equals_pointer)
{
	Simple* a = new Simple;
	Simple* b = new Simple;
	EXPECT_EQ(make_any(a), make_any(b));
	EXPECT_EQ(make_any(b), make_any(a));
	delete a;
	delete b;
}

TEST(Any, equals_null_pointer)
{
	Simple* a = nullptr;
	Simple* b = nullptr;
	EXPECT_EQ(make_any(a), make_any(b));
	EXPECT_EQ(make_any(b), make_any(a));

	b = new Simple;
	EXPECT_NE(make_any(a), make_any(b));
	EXPECT_NE(make_any(b), make_any(a));
	delete b;
}

TEST(Any, equals_value)
{
	Simple a;
	Simple b;
	EXPECT_EQ(make_any(a), make_any(b));
	EXPECT_EQ(make_any(b), make_any(a));
}

TEST(Any, clone_into_non_owning_via_clone)
{
	Simple* a = nullptr;
	Simple* other = new Simple;
	auto a_any = make_any_ref(&a);
	Simple* old_a = a;
	a_any.clone_from(make_any(other));
	auto cloned = a_any.as<Simple*>();
	EXPECT_NE(cloned, nullptr);
	EXPECT_EQ(cloned->cloned, true);
	delete cloned;
	delete old_a;
	delete other;
}

TEST(Any, clone_into_non_owning_via_copy)
{
	int a = 3;
	int other = 5;
	auto a_any = make_any_ref(&a);
	a_any.clone_from(make_any(other));
	EXPECT_EQ(a_any.as<int>(), other);
	EXPECT_EQ(a, other);
}

TEST(Any, clone_wrong_type)
{
	Simple* a = nullptr;
	int other = 5;
	auto any = make_any_ref(&a);
	EXPECT_THROW(any.clone_from(make_any(other)), std::logic_error);
}

TEST(Any, clone_into_owning_via_copy)
{
	int a = 3;
	int other = 5;
	auto a_any = make_any(a);
	a_any.clone_from(make_any(other));
	EXPECT_EQ(a_any.as<int>(), other);
}

TEST(Any, clone_sgvector)
{
	auto a = SGVector<float64_t>(3);
	a.range_fill();
	SGVector<float64_t> b;
	ASSERT_FALSE(a.equals(b));

	auto a_any = make_any(a);
	auto b_any = make_any(b);

	auto cloned_b = b_any.clone_from(a_any).as<SGVector<float64_t>>();

	EXPECT_NE(a.vector, cloned_b.vector);
	EXPECT_TRUE(a.equals(cloned_b));
}

TEST(Any, clone_sgmatrix)
{
	auto a = SGMatrix<float64_t>(3, 4);
	SGVector<float64_t>(a.matrix, a.num_rows * a.num_cols, false).range_fill();
	SGMatrix<float64_t> b;
	ASSERT_FALSE(a.equals(b));

	auto a_any = make_any(a);
	auto b_any = make_any(b);

	auto cloned_b = b_any.clone_from(a_any).as<SGMatrix<float64_t>>();

	EXPECT_NE(a.matrix, cloned_b.matrix);
	EXPECT_TRUE(a.equals(cloned_b));
}
