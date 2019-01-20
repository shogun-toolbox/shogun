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

namespace shogun {
template <typename T1, typename T2>
constexpr bool sg_is_same_v = std::is_same<T1, T2>::value;
// Checks if any one of the types matches
// Ref: https://stackoverflow.com/a/17032517/3656081
template<typename T, typename... Rest>
struct sg_is_any_of : std::false_type {
};

template<typename T, typename First>
struct sg_is_any_of<T, First> : std::is_same<T, First> {
};

template<typename T, typename First, typename... Rest>
struct sg_is_any_of<T, First, Rest...>
 : std::integral_constant<bool, std::is_same<T, First>::value ||
     sg_is_any_of<T, Rest...>::value> {
};

template<typename T, typename First, typename... Rest>
constexpr bool sg_is_any_of_v = sg_is_any_of<T, First, Rest...>::value;

template <typename T>
concept Floating = std::is_floating_point<T>::value;
template <typename T>
concept Integral = std::is_integral<T>::value;

// Anything that uses the fill_array* macros
template<typename T>
concept RngVectorizable = requires(T* container, index_t len, CRandom* r){
{ r->fill_array(container, len) };
} || requires(T* container, index_t len, CRandom* r){
{ r->fill_array_co(container, len) };
};

// all shogun base classes for put/add templates
class CMachine;
class CKernel;
class CDistance;
class CFeatures;
class CLabels;
class CECOCEncoder;
class CECOCDecoder;
class CMulticlassStrategy;
class CNeuralLayer;
// type trait to enable certain methods only for shogun base types
template<class T>
struct is_sg_base
 : std::integral_constant<bool,
                          sg_is_any_of_v<T, CMachine, CKernel,
                                         CDistance, CFeatures,
                                         CLabels, CECOCEncoder,
                                         CECOCDecoder, CMulticlassStrategy,
                                         CNeuralLayer>> {
};
}
#endif // SHOGUN_SG_TYPE_TRAITS_H
