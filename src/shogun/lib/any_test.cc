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

#include <numeric>
#include <shogun/base/SGObject.h>
#include <shogun/lib/any.h>
#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>
#include <stdexcept>
#include <functional>
#include <map>

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

struct SimpleValue
{
public:
	SimpleValue clone() const
	{
		SimpleValue copy;
		copy.cloned = true;
		return copy;
	}
	bool equals(const SimpleValue& other) const
	{
		return true;
	}

	std::string to_string() const
	{
		return "abracadabra";
	}

	bool cloned = false;
};

class Object : public SGObject
{
public:
	virtual const char* get_name() const override
	{
		return "Object";
	}

	std::shared_ptr<SGObject> create_empty() const override
	{
		return std::make_shared<Object>();
	}

	int computed_member() const
	{
		return 90210;
	}
};

TEST(Any, as)
{
	int32_t integer = 10;
	auto any = make_any(integer);
	EXPECT_EQ(any.as<int32_t>(), integer);
	EXPECT_THROW(any.as<float64_t>(), TypeMismatchException);
}

TEST(Any, has_type)
{
	int32_t integer = 10;
	auto any = make_any(integer);
	EXPECT_EQ(any.has_type<int32_t>(), true);
	EXPECT_EQ(any.has_type<float64_t>(), false);
}

TEST(Any, empty)
{
	int32_t integer = 10;
	auto any = make_any(integer);
	auto empty_any = Any();
	EXPECT_EQ(any.empty(), false);
	EXPECT_EQ(empty_any.empty(), true);
}

TEST(Any, has_type_fallback)
{
	int32_t integer = 10;
	auto any = make_any(integer);
	EXPECT_EQ(any.has_type_fallback<int32_t>(), true);
	EXPECT_EQ(any.has_type_fallback<float64_t>(), false);
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
	EXPECT_THROW(any_cast<float64_t>(any), TypeMismatchException);
	EXPECT_THROW(any_cast<int32_t>(empty_any), TypeMismatchException);
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
	EXPECT_THROW(any = make_any(3.14), TypeMismatchException);
}

TEST(Any, assign_wrong_type_into_non_owning)
{
	int32_t integer = 10;
	auto any = make_any_ref(&integer);
	EXPECT_THROW(any = make_any(3.14), TypeMismatchException);
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
	SimpleValue a;
	SimpleValue b;
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
	EXPECT_THROW(any.clone_from(make_any(other)), TypeMismatchException);
}

TEST(Any, clone_into_owning_via_copy)
{
	int a = 3;
	int other = 5;
	auto a_any = make_any(a);
	a_any.clone_from(make_any(other));
	EXPECT_EQ(a_any.as<int>(), other);
}

TEST(Any, clone_value)
{
	auto a_any = make_any(SimpleValue());
	auto b_any = make_any(SimpleValue());
	auto cloned_b = b_any.clone_from(a_any).as<SimpleValue>();
}

TEST(Any, array_ref)
{
	int src_len = 5;
	float* src = new float[src_len];
	std::iota(src, src + src_len, 9);
	int dst_len = 8;
	float* dst = new float[dst_len];
	std::iota(dst, dst + dst_len, 5);
	int other_len = src_len;
	float* other = new float[other_len];
	std::iota(other, other + other_len, 9);

	auto any_src = make_any_ref(&src, &src_len);
	auto any_dst = make_any_ref(&dst, &dst_len);
	auto any_other = make_any_ref(&other, &other_len);

	EXPECT_EQ(any_src, any_src);
	EXPECT_EQ(any_dst, any_dst);

	EXPECT_NE(any_src, any_dst);
	EXPECT_NE(any_dst, any_src);

	EXPECT_EQ(any_src, any_other);
	EXPECT_EQ(any_other, any_src);

	EXPECT_NE(any_dst, any_other);
	EXPECT_NE(any_other, any_dst);

	delete[] src;
	delete[] dst;
	delete[] other;
}

TEST(Any, array2d_ref)
{
	int src_rows = 5;
	int src_cols = 4;
	int src_size = src_rows * src_cols;
	float* src = new float[src_size];
	std::iota(src, src + src_size, 9);

	int dst_rows = 3;
	int dst_cols = 2;
	int dst_size = dst_rows * dst_cols;
	float* dst = new float[dst_size];
	std::iota(dst, dst + dst_size, 5);

	int other_rows = src_rows;
	int other_cols = src_cols;
	int other_size = other_rows * other_cols;
	float* other = new float[other_size];
	std::iota(other, other + other_size, 9);

	auto any_src = make_any_ref(&src, &src_rows, &src_cols);
	auto any_dst = make_any_ref(&dst, &dst_rows, &dst_cols);
	auto any_other = make_any_ref(&other, &other_rows, &other_cols);

	EXPECT_EQ(any_src, any_src);
	EXPECT_EQ(any_dst, any_dst);

	EXPECT_NE(any_src, any_dst);
	EXPECT_NE(any_dst, any_src);

	EXPECT_EQ(any_src, any_other);
	EXPECT_EQ(any_other, any_src);

	EXPECT_NE(any_dst, any_other);
	EXPECT_NE(any_other, any_dst);

	delete[] src;
	delete[] dst;
	delete[] other;
}

TEST(Any, clone_array)
{
	int src_size = 3;
	float* src = new float[src_size];
	std::iota(src, src + src_size, 9);
	auto any_src = make_any_ref(&src, &src_size);

	float* dst = nullptr;
	int dst_size = 0;
	auto any_dst = make_any_ref(&dst, &dst_size);
	any_dst.clone_from(any_src);

	EXPECT_EQ(src_size, dst_size);
	EXPECT_NE(src, dst);
	EXPECT_NE(dst, nullptr);
	EXPECT_EQ(any_src, any_dst);

	delete[] src;
}

TEST(Any, clone_array2d)
{
	int src_rows = 5;
	int src_cols = 4;
	int src_size = src_rows * src_cols;
	float* src = new float[src_size];
	std::iota(src, src + src_size, 9);
	auto any_src = make_any_ref(&src, &src_rows, &src_cols);

	float* dst = nullptr;
	int dst_rows = 0;
	int dst_cols = 0;
	auto any_dst = make_any_ref(&dst, &dst_rows, &dst_cols);
	any_dst.clone_from(any_src);

	EXPECT_EQ(src_rows, dst_rows);
	EXPECT_EQ(src_cols, dst_cols);
	EXPECT_NE(src, dst);
	EXPECT_NE(dst, nullptr);
	EXPECT_EQ(any_src, any_dst);

	delete[] src;
}

TEST(Any, clone_array_sgobject)
{
	int src_len = 3;
	std::shared_ptr<Object>* src = new std::shared_ptr<Object>[src_len];
	std::generate(src, src + src_len, []() { return std::make_shared<Object>(); });
	auto any_src = make_any_ref(&src, &src_len);

	std::shared_ptr<Object>* dst = nullptr;
	int dst_len = 0;
	auto any_dst = make_any_ref(&dst, &dst_len);
	any_dst.clone_from(any_src);

	EXPECT_EQ(src_len, dst_len);
	EXPECT_NE(src, dst);
	for (int i = 0; i < dst_len; i++)
	{
		EXPECT_NE(src[i], dst[i]);
		EXPECT_TRUE(src[i]->equals(dst[i]));
	}
	EXPECT_NE(dst, nullptr);
	EXPECT_EQ(any_src, any_dst);

	delete[] src;
}

TEST(Any, clone_array2d_sgobject)
{
	int src_rows = 5;
	int src_cols = 4;
	int src_size = src_rows * src_cols;
	std::shared_ptr<Object>* src = new std::shared_ptr<Object>[src_size];
	std::generate(src, src + src_size, []() { return std::make_shared<Object>(); });
	auto any_src = make_any_ref(&src, &src_rows, &src_cols);

	int dst_rows = 0;
	int dst_cols = 0;
	std::shared_ptr<Object>* dst = nullptr;
	auto any_dst = make_any_ref(&dst, &dst_rows, &dst_cols);
	any_dst.clone_from(any_src);

	EXPECT_EQ(src_rows, dst_rows);
	EXPECT_EQ(src_cols, dst_cols);
	EXPECT_NE(src, dst);
	for (int i = 0; i < dst_rows * dst_cols; i++)
	{
		EXPECT_NE(src[i], dst[i]);
		EXPECT_TRUE(src[i]->equals(dst[i]));
	}
	EXPECT_NE(dst, nullptr);
	EXPECT_EQ(any_src, any_dst);

	delete[] src;
}

TEST(Any, free_array_simple)
{
	auto size = 4;
	auto array = SG_MALLOC(float, size);
	any_detail::free_array(array, size);
}

/*
 * FIXME
TEST(Any, free_array_sgobject)
{
	auto obj = std::make_shared<Object>();

	auto size = 4;
	auto array = SG_MALLOC(SGObject*, size);
	for (auto i = 0; i < size; ++i)
	{
		array[i] = obj;

	}
	EXPECT_EQ(obj.use_count(), size + 1);
	any_detail::free_array(array, size);
	EXPECT_EQ(obj.use_count(), 1);

}

TEST(Any, reset_array_reference)
{
	auto obj = std::make_shared<Object>();

	auto size = 4;
	auto array = SG_MALLOC(Object*, size);
	for (auto i = 0; i < size; ++i)
	{
		array[i] = obj;

	}
	EXPECT_EQ(obj.use_count(), size + 1);

	Object** src_array = nullptr;
	int src_size = 0;

	auto array_ref = ArrayReference<Object*, int>(&array, &size);
	array_ref.reset(ArrayReference<Object*, int>(&src_array, &src_size));
	EXPECT_EQ(obj.use_count(), 1);

}

TEST(Any, reset_array2d_reference)
{
	auto obj = std::make_shared<Object>();

	auto rows = 4;
	auto cols = 3;
	auto array = SG_MALLOC(Object*, rows * cols);
	for (auto i = 0; i < rows * cols; ++i)
	{
		array[i] = obj;

	}
	EXPECT_EQ(obj.use_count(), rows * cols + 1);

	Object** src_array = nullptr;
	int src_rows = 0;
	int src_cols = 0;

	auto array_ref = Array2DReference<Object*, int>(&array, &rows, &cols);
	array_ref.reset(
	    Array2DReference<Object*, int>(&src_array, &src_rows, &src_cols));
	EXPECT_EQ(obj.use_count(), 1);
}
*/

TEST(Any, lazy_simple)
{
	auto v = 9;
	auto any = make_any<int>([=]() { return v; });
	EXPECT_EQ(any.as<int32_t>(), v);
}

TEST(Any, lazy_assignment_into_with_value)
{
	auto v = 3;
	auto any = make_any<int>([=]() { return 111; });
	EXPECT_THROW(any = make_any<int>(v), TypeMismatchException);
}

TEST(Any, lazy_assignment_into_with_function)
{
	auto v = 3;
	auto any = make_any<int>([=]() { return 111; });
	any = make_any<int>([=]() { return v; });
	EXPECT_EQ(any.as<int32_t>(), v);
}

TEST(Any, lazy_assignment_from)
{
	auto v = 3;
	auto any = make_any(0);
	EXPECT_THROW(
	    any = make_any<int>([=]() { return v; }), TypeMismatchException);
}

TEST(Any, lazy_member_function)
{
	auto obj = std::make_shared<Object>();
	auto any = make_any<int>(std::bind(&Object::computed_member, obj));
	EXPECT_EQ(any.as<int>(), obj->computed_member());
}

TEST(Any, lazy_cloneable_visitable)
{
	Any any;
	EXPECT_THROW(
	    any.clone_from(make_any<void>([=]() { return; })), std::logic_error);
	EXPECT_TRUE(any.cloneable());
	EXPECT_TRUE(any.visitable());
	any = make_any<void>([=]() { return; });
	EXPECT_FALSE(any.cloneable());
	EXPECT_FALSE(any.visitable());
	EXPECT_THROW(any.visit(nullptr), std::logic_error);
}

TEST(Any, lazy_compare_no_args)
{
	bool called = false;
	auto l = make_any<void>([&]() { called = true; });
	auto r = make_any<void>([&]() { called = true; });
	EXPECT_NE(l, r);
	EXPECT_EQ(called, false);
}

TEST(Any, hash_empty)
{
	Any empty;
	EXPECT_EQ(empty.hash(), 0UL);
}

TEST(Any, hash_number)
{
	size_t number = 91;
	EXPECT_EQ(make_any(number).hash(), std::hash<decltype(number)>{}(number));
}

TEST(Any, hash_lazy)
{
	auto v = 9;
	auto lazy = make_any<int>([=]() { return v; });
	EXPECT_EQ(lazy.hash(), 0UL);
}

TEST(Any, special_compare)
{
	EXPECT_EQ(any_detail::has_special_compare<float32_t>::value, true);
	EXPECT_EQ(any_detail::has_special_compare<float64_t>::value, true);
	EXPECT_EQ(any_detail::has_special_compare<floatmax_t>::value, true);
	EXPECT_EQ(any_detail::has_special_compare<complex128_t>::value, true);
	EXPECT_EQ(any_detail::has_special_compare<int>::value, false);
	EXPECT_EQ(any_detail::has_special_compare<Object>::value, false);
}

TEST(Any, hash_vector)
{
	std::vector<int> values = {1,2,3};
	EXPECT_EQ(make_any(values).hash(), std::hash<int>{}(1) ^ std::hash<int>{}(2) ^ std::hash<int>{}(3));
}

TEST(Any, hash_vector_vector)
{
	std::vector<std::vector<int>> values = {{1,2,3}};
	EXPECT_EQ(make_any(values).hash(), std::hash<int>{}(1) ^ std::hash<int>{}(2) ^ std::hash<int>{}(3));
}

TEST(AnyParameterProperties, api_default)
{
	AnyParameterProperties params = AnyParameterProperties();

	EXPECT_TRUE(params.compare_mask(ParameterProperties::NONE));
}

TEST(AnyParameterProperties, has_property)
{
	AnyParameterProperties params =
		AnyParameterProperties("test", ParameterProperties::HYPER | ParameterProperties::MODEL);
	EXPECT_TRUE(params.has_property(ParameterProperties::HYPER));
	EXPECT_TRUE(params.has_property(ParameterProperties::MODEL));
	EXPECT_TRUE(params.compare_mask(ParameterProperties::HYPER | ParameterProperties::MODEL));
}

TEST(AnyParameterProperties, remove_property)
{
	AnyParameterProperties params =
			AnyParameterProperties("test", ParameterProperties::HYPER | ParameterProperties::MODEL);
	params.remove_property(ParameterProperties::HYPER);
	EXPECT_FALSE(params.has_property(ParameterProperties::HYPER));
	EXPECT_TRUE(params.has_property(ParameterProperties::MODEL));
	EXPECT_TRUE(params.compare_mask(ParameterProperties::MODEL));
}

TEST(Any, compare_string_vectors)
{
        std::string str("some string");
        std::vector<std::string> lhs{str};
        std::vector<std::string> rhs{str};
        Any any_lhs = make_any(lhs);
        Any any_rhs = make_any(rhs);
        EXPECT_EQ(any_lhs, any_rhs);
}

TEST(Any, compare_object_vectors)
{
        auto lhs_obj = std::make_shared<Object>();
        auto rhs_obj = std::make_shared<Object>();
        EXPECT_TRUE(lhs_obj->equals(rhs_obj.get()));
        std::vector<std::shared_ptr<SGObject>> lhs{lhs_obj};
        std::vector<std::shared_ptr<SGObject>> rhs{rhs_obj};
        Any any_lhs = make_any(lhs);
        Any any_rhs = make_any(rhs);
        EXPECT_EQ(any_lhs, any_rhs);
}

TEST(Any, compare_string_maps)
{
        std::string key("key");
        std::string value("value");
        std::map<std::string, std::string> lhs = {{key, value}};
        std::map<std::string, std::string> rhs = {{key, value}};
        Any any_lhs = make_any(lhs);
        Any any_rhs = make_any(rhs);
        EXPECT_EQ(any_lhs, any_rhs);
}

TEST(Any, compare_object_maps)
{
        auto lhs_obj = std::make_shared<Object>();
        auto rhs_obj = std::make_shared<Object>();
        EXPECT_TRUE(lhs_obj->equals(rhs_obj));
        std::map<std::string, std::shared_ptr<SGObject>> lhs = {{"test", lhs_obj}};
        std::map<std::string, std::shared_ptr<SGObject>> rhs = {{"test", rhs_obj}};
        Any any_lhs = make_any(lhs);
        Any any_rhs = make_any(rhs);
        EXPECT_EQ(any_lhs, any_rhs);
}

TEST(Any, simple_casting)
{
	float value = 36.0;
	auto any = make_any<float>(value);
	EXPECT_EQ(any.cast<int>(), static_cast<int>(value));
}

TEST(Any, fail_to_cast)
{
	auto any = make_any<std::string>("dsads");
	EXPECT_THROW(any.cast<float64_t>(), std::logic_error);
}

TEST(Any, cast_into_sgobject)
{
	auto ptr = std::make_shared<Object>();
	auto any = make_any(ptr);
	EXPECT_EQ(any.cast<std::shared_ptr<Object>>(), ptr);
	EXPECT_EQ(any.cast<std::shared_ptr<SGObject>>(), ptr);
	EXPECT_THROW(any.cast<int>(), std::logic_error);
}

TEST(Any, simple_visit)
{
	int extracted_value{};
	struct ExtractVisitor
	{
	};
	Any::register_visitor<int, ExtractVisitor>([&] (auto value, auto visitor) { extracted_value = value; });
	auto any = make_any<int>(42);
	any.visit_with<ExtractVisitor>();
	EXPECT_EQ(any.as<int>(), extracted_value);
}

TEST(Any, stateful_visit)
{
	const std::string initial_value = "hello, world!";
	struct StringStreamVisitor
	{
		std::stringstream ss;
	};
	Any::register_visitor<std::string, StringStreamVisitor>([&] (auto value, auto visitor) {
		(visitor->ss) << value;
	});
	auto any = make_any(initial_value);
	StringStreamVisitor visitor;
	any.visit_with(&visitor);
	any.visit_with(&visitor);
	EXPECT_EQ(visitor.ss.str(), initial_value + initial_value);
}

TEST(Any, to_string_using_std)
{
	EXPECT_EQ(std::to_string(static_cast<int>(0)), make_any<int>(0).cast<std::string>());
	EXPECT_EQ(std::to_string(static_cast<long>(0)), make_any<long>(0).cast<std::string>());
	EXPECT_EQ(std::to_string(static_cast<long long>(0)), make_any<long long>(0).cast<std::string>());
	EXPECT_EQ(std::to_string(static_cast<unsigned>(0)), make_any<unsigned>(0).cast<std::string>());
	EXPECT_EQ(std::to_string(static_cast<unsigned long>(0)), make_any<unsigned long>(0).cast<std::string>());
	EXPECT_EQ(std::to_string(static_cast<unsigned long long>(0)), make_any<unsigned long long>(0).cast<std::string>());
	EXPECT_EQ(std::to_string(static_cast<float>(0)), make_any<float>(0).cast<std::string>());
	EXPECT_EQ(std::to_string(static_cast<double>(0)), make_any<double>(0).cast<std::string>());
	EXPECT_EQ(std::to_string(static_cast<long double>(0)), make_any<long double>(0).cast<std::string>());
}

TEST(Any, own_to_string)
{
	SimpleValue value{};
	EXPECT_EQ(value.to_string(), make_any(value).cast<std::string>());
}
