/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_SG_GTEST_UTILITIES_H
#define SHOGUN_SG_GTEST_UTILITIES_H

#include <gtest/gtest.h>
#include <shogun/lib/sg_types.h>

template <typename Types>
struct TypesGoogleTestWrapper;

template <template <typename...> class TypesT, typename... Args>
struct TypesGoogleTestWrapper<TypesT<Args...>>
{
	using type = ::testing::Types<Args...>;
};

template <typename T1, typename... Args>
struct PopTypesGoogleTestWrapper
{
};

template <template <typename...> class TypesT, typename... Args>
struct PopTypesGoogleTestWrapper<TypesT<Args...>>
{
	using type = typename TypesGoogleTestWrapper<Types<Args...>>::type;
};

template <
    template <typename...> class TypesT, typename... Args, typename... Args1>
struct PopTypesGoogleTestWrapper<TypesT<Args...>, Args1...>
{
	using type = typename TypesGoogleTestWrapper<
	    typename popTypesByTypes<Types<Args...>, Types<Args1...>>::type>::type;
};

#define RANDOM_NAME(a, b) RANDOM_NAME_I(a, b)
#define RANDOM_NAME_I(a, b) RANDOM_NAME_II(~, a##b)
#define RANDOM_NAME_II(p, res) res

#define SG_TYPED_TEST_CASE(class_name, types, ...)                             \
	using RANDOM_NAME(class_name, __LINE__) =                                  \
	    typename PopTypesGoogleTestWrapper<types, ##__VA_ARGS__>::type;        \
	TYPED_TEST_CASE(class_name, RANDOM_NAME(class_name, __LINE__))

#endif // SHOGUN_SG_GTEST_UTILITIES_H