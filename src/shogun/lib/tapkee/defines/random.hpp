/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DEFINES_RANDOM_H_
#define TAPKEE_DEFINES_RANDOM_H_

#include <cstdlib>
#include <algorithm>
#include <limits>

namespace tapkee
{

inline IndexType uniform_random_index()
{
#ifdef CUSTOM_UNIFORM_RANDOM_INDEX_FUNCTION
	return CUSTOM_UNIFORM_RANDOM_INDEX_FUNCTION % std::numeric_limits<IndexType>::max();
#else
	return std::rand();
#endif
}

inline IndexType uniform_random_index_bounded(IndexType upper)
{
	return uniform_random_index() % upper;
}

inline ScalarType uniform_random()
{
#ifdef CUSTOM_UNIFORM_RANDOM_FUNCTION
	return CUSTOM_UNIFORM_RANDOM_FUNCTION;
#else
	return std::rand()/((double)RAND_MAX+1);
#endif
}

inline ScalarType gaussian_random()
{
#ifdef CUSTOM_GAUSSIAN_RANDOM_FUNCTION
	return CUSTOM_GAUSSIAN_RANDOM_FUNCTION;
#else
	ScalarType x, y, radius;
	do {
		x = 2*(std::rand()/((double)RAND_MAX+1)) - 1;
		y = 2*(std::rand()/((double)RAND_MAX+1)) - 1;
		radius = (x * x) + (y * y);
	} while ((radius >= 1.0) || (radius == 0.0));
	radius = std::sqrt(-2 * std::log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
#endif
}

template <class RAI>
inline void random_shuffle(RAI first, RAI last)
{
	std::random_shuffle(first,last,uniform_random_index_bounded);
}

}

#endif


