/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <gtest/gtest.h>
#include <shogun/lib/type_case.h>

using namespace shogun;
using testing::StaticAssertTypeEq;

//template <typename T>
//struct Simple
//{
//public:
//	Simple(T value_) : value(value_){};
//	T get_value()
//	{
//		return value;
//	};
//	void add_value(T value_)
//	{
//		value += value_;
//	};
//
//private:
//	T value;
//};
//
//template <typename T>
//class StaticAssertReturnTypeEqTestHelper
//{
//public:
//	StaticAssertReturnTypeEqTestHelper()
//	{
//		StaticAssertTypeEq<type_internal::assert_return_type_is_valid, T>();
//	}
//};
//
//template <typename T>
//class StaticAssertArityEqTestHelper
//{
//public:
//	StaticAssertArityEqTestHelper()
//	{
//		StaticAssertTypeEq<type_internal::assert_arity_is_valid, T>();
//	}
//};
//
//TEST(Type_case, positional_lambdas)
//{
//	float32_t a_scalar = 42.0;
//	SGVector<float32_t> a_vector = {a_scalar};
//	auto a_matrix = SGMatrix<float32_t>(a_vector);
//	int counter = 0;
//
//	auto any_scalar = make_any(a_scalar);
//	auto any_vector = make_any(a_vector);
//	auto any_matrix = make_any(a_matrix);
//
//	auto f_scalar = [&counter](auto value) { counter++; };
//	auto f_vector = [&counter](auto value) { counter++; };
//	auto f_matrix = [&counter](auto value) { counter++; };
//
//	sg_any_dispatch(any_scalar, sg_all_types, f_scalar);
//	EXPECT_EQ(counter, 1);
//
//	sg_any_dispatch(any_vector, sg_all_types, None{}, f_vector);
//	EXPECT_EQ(counter, 2);
//
//	sg_any_dispatch(any_matrix, sg_all_types, None{}, None{}, f_matrix);
//	EXPECT_EQ(counter, 3);
//
//	sg_any_dispatch(any_scalar, sg_all_types, None{}, f_vector, f_matrix);
//	EXPECT_EQ(counter, 3);
//}
//
//TEST(Type_case, exception)
//{
//	int32_t a_scalar = 42;
//	int counter = 0;
//	auto any_scalar = make_any(a_scalar);
//
//	auto f_scalar = [&counter](auto value) { counter++; };
//
//	EXPECT_THROW(
//		sg_any_dispatch(any_scalar, sg_real_types, f_scalar), ShogunException);
//	EXPECT_EQ(counter, 0);
//}
//
//TEST(Type_case, modify_struct)
//{
//	auto a_struct = Simple<float32_t>(42);
//	int32_t a_int = 42;
//	float32_t a_float = 42.0;
//
//	auto any_int = make_any(a_int);
//	auto any_float = make_any(a_float);
//
//	auto f_int = [&a_struct](auto value) {
//		a_struct.add_value(value);
//	};
//
//	auto f_float = [&a_struct](auto value) {
//		a_struct.add_value(value);
//	};
//
//	sg_any_dispatch(any_float, sg_real_types, f_float);
//	EXPECT_EQ(a_struct.get_value(), 84);
//	EXPECT_THROW(
//		sg_any_dispatch(any_int, sg_real_types, f_int), ShogunException);
//	EXPECT_EQ(a_struct.get_value(), 84);
//}
//
//TEST(Type_case, custom_map)
//{
//	int counter = 0;
//	int32_t a_int = 42;
//	float32_t a_float = 42.0;
//
//	auto any_int = make_any(a_int);
//	auto any_float = make_any(a_float);
//
//#define ADD_TYPE_TO_MAP(TYPENAME, TYPE_ENUM)                                   \
//	{std::type_index(typeid(TYPENAME)), TYPE_ENUM},
//
//	typemap my_int_map = {
//			ADD_TYPE_TO_MAP(int8_t, TYPE::T_INT8)
//			ADD_TYPE_TO_MAP(int16_t, TYPE::T_INT16)
//			ADD_TYPE_TO_MAP(int32_t, TYPE::T_INT32)
//			ADD_TYPE_TO_MAP(int64_t, TYPE::T_INT64)
//	};
//
//#undef ADD_TYPE_TO_MAP
//
//	auto f = [&counter](auto a) { counter++; };
//
//	sg_any_dispatch(any_int, my_int_map, f);
//	EXPECT_EQ(counter, 1);
//	EXPECT_THROW(sg_any_dispatch(any_float, my_int_map, f), ShogunException);
//	EXPECT_EQ(counter, 1);
//}

//TEST(Type_case, static_asserts)
//{
//	float32_t a_float = 42.0;
//	auto any_float = make_any(a_float);
//
//	auto f_return_fail = [](auto a) { return 1; };
//	auto f_arity_fail = [](auto a, float b) {};
//
//	StaticAssertReturnTypeEqTestHelper<decltype(
//		sg_any_dispatch(any_float, sg_all_types, f_return_fail))>();
//	StaticAssertArityEqTestHelper<decltype(
//		sg_any_dispatch(any_float, sg_all_types, f_arity_fail))>();
//}