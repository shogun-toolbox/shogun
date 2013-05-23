/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DEFINES_RANDOM_H_
#define TAPKEE_DEFINES_RANDOM_H_

#include <cstdlib>
#include <algorithm>

namespace tapkee
{

IndexType uniform_random_index() 
{
#ifdef CUSTOM_UNIFORM_RANDOM_INDEX_FUNCTION
	return CUSTOM_UNIFORM_RANDOM_INDEX_FUNCTION;
#else
	return rand();
#endif
}

IndexType uniform_random_index_bounded(IndexType upper)
{
	return uniform_random_index() % upper;
}

ScalarType uniform_random()
{
#ifdef CUSTOM_UNIFORM_RANDOM_FUNCTION
	return CUSTOM_UNIFORM_RANDOM_FUNCTION;
#else
	return rand()/static_cast<ScalarType>(RAND_MAX);
#endif
}

ScalarType gaussian_random()
{
#ifdef CUSTOM_GAUSSIAN_RANDOM_FUNCTION
	return CUSTOM_GAUSSIAN_RANDOM_FUNCTION;
#else
	ScalarType x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while ((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
#endif
}

template <class RAI>
void random_shuffle(RAI first, RAI last)
{
	std::random_shuffle(first,last);
}

}

#endif


