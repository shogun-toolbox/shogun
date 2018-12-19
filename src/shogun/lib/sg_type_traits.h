/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saatvik Shah
 */

#include <type_traits>

#ifndef SHOGUN_SG_TYPE_TRAITS_H
#define SHOGUN_SG_TYPE_TRAITS_H
/**
 * A collection of useful type traits
 */

// Checks if any one of the types matches
// Ref: https://stackoverflow.com/a/17032517/3656081
template <typename T, typename... Rest>
struct is_any_of : std::false_type
{
};

template <typename T, typename First>
struct is_any_of<T, First> : std::is_same<T, First>
{
};

template <typename T, typename First, typename... Rest>
struct is_any_of<T, First, Rest...>
    : std::integral_constant<bool, std::is_same<T, First>::value ||
                                       is_any_of<T, Rest...>::value>
{
};

template <typename T, typename First, typename... Rest>
constexpr bool is_any_of_v = is_any_of<T, First, Rest...>::value;
#endif // SHOGUN_SG_TYPE_TRAITS_H
