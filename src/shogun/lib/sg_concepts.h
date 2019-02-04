/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saatvik Shah
 */

#include <type_traits>

#ifndef SHOGUN_SG_CONCEPTS_H
#define SHOGUN_SG_CONCEPTS_H
/**
 * A collection of useful concepts
 */

namespace shogun{

template <typename T>
concept Floating = std::is_floating_point<T>::value;
template <typename T>
concept Integral = std::is_integral<T>::value;

template<typename T>
concept Rngable = requires(CRandom* r){
{ r->random(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()) };
};

// Anything that uses the fill_array* macros
template<typename T>
concept RngVectorizable = requires(T* container, index_t len, CRandom* r){
{ r->fill_array(container, len) };
} || requires(T* container, index_t len, CRandom* r){
{ r->fill_array_co(container, len) };
};
}
#endif // SHOGUN_SG_CONCEPTS_H
