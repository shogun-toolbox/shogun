/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <gtest/gtest.h>
#include <shogun/lib/type_case.h>

using namespace shogun;

template <typename T>
struct Simple
{
public:
	Simple(T value_) : value(value_){};
	T get_value()
	{
		return value;
	};
	void add_value(T value_)
	{
		value += value_;
	};

private:
	T value;
};

TEST(Type_case, positional_lambdas)
{
	float32_t a_scalar = 42.0;
	auto a_vector = SGVector<float32_t>({a_scalar});
	auto a_matrix = SGMatrix<float32_t>(a_vector);
	int counter = 0;

	auto any_scalar = make_any(a_scalar);
	auto any_vector = make_any(a_vector);
	auto any_matrix = make_any(a_matrix);

	auto f_scalar = [&counter](auto type) { counter++; };
	auto f_vector = [&counter](auto type) { counter++; };
	auto f_matrix = [&counter](auto type) { counter++; };

	sg_for_each_type(any_scalar, sg_all_types, f_scalar);
	EXPECT_EQ(counter, 1);

	sg_for_each_type(any_vector, sg_all_types, nullptr, f_vector);
	EXPECT_EQ(counter, 2);

	sg_for_each_type(any_matrix, sg_all_types, nullptr, nullptr, f_matrix);
	EXPECT_EQ(counter, 3);

	sg_for_each_type(any_scalar, sg_all_types, nullptr, f_vector, f_matrix);
	EXPECT_EQ(counter, 3);
}

TEST(Type_case, exception)
{
	int32_t a_scalar = 42;
	int counter = 0;
	auto any_scalar = make_any(a_scalar);

	auto f_scalar = [&counter](auto type) { counter += 1; };

	EXPECT_THROW(
		sg_for_each_type(any_scalar, sg_real_types, f_scalar), ShogunException);
	EXPECT_EQ(counter, 0);
}

TEST(Type_case, modify_struct)
{
	auto a_struct = Simple<float32_t>(42);
	int32_t a_int = 42;
	float32_t a_float = 42.0;

	auto any_int = make_any(a_int);
	auto any_float = make_any(a_float);

	auto f_int = [&a_struct, &any_int](auto type) {
		a_struct.add_value(any_cast<decltype(type)>(any_int));
	};

	auto f_float = [&a_struct, &any_float](auto type) {
		a_struct.add_value(any_cast<decltype(type)>(any_float));
	};

	sg_for_each_type(any_float, sg_real_types, f_float);
	EXPECT_EQ(a_struct.get_value(), 84);
	EXPECT_THROW(
		sg_for_each_type(any_int, sg_real_types, f_int), ShogunException);
	EXPECT_EQ(a_struct.get_value(), 84);
}

TEST(Type_case, custom_map)
{
	int counter = 0;
	int32_t a_int = 42;
	float32_t a_float = 42.0;

	auto any_int = make_any(a_int);
	auto any_float = make_any(a_float);

#define ADD_TYPE_TO_MAP(TYPENAME, TYPE_ENUM)                                   \
	{std::type_index(typeid(TYPENAME)), TYPE_ENUM},

	typemap my_int_map = {
			ADD_TYPE_TO_MAP(int8_t, TYPE::PT_INT8)
			ADD_TYPE_TO_MAP(int16_t, TYPE::PT_INT16)
			ADD_TYPE_TO_MAP(int32_t, TYPE::PT_INT32)
			ADD_TYPE_TO_MAP(int64_t, TYPE::PT_INT64)
	};

#undef ADD_TYPE_TO_MAP

	auto f = [&counter](auto type) { counter++; };

	sg_for_each_type(any_int, my_int_map, f);
	EXPECT_THROW(sg_for_each_type(any_float, my_int_map, f), ShogunException);
	EXPECT_EQ(counter, 1);
}