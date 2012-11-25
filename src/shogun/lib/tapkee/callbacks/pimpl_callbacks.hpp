/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 *
 */

#ifndef TAPKEE_PIMPL_CALLBACKS_H_
#define TAPKEE_PIMPL_CALLBACKS_H_

// Kernel function callback that computes
// similarity function values on vectors 
// given by their indices. This impl. computes 
// kernel i.e. dot product between two vectors.
template<class Implementation>
struct pimpl_kernel_callback
{
	pimpl_kernel_callback(Implementation* i) : impl(i) {};
	inline DefaultScalarType operator()(int a, int b) const
	{
		return impl->kernel(a,b);
	}
	Implementation* impl;
};
// That's mandatory to specify that kernel_callback
// is a kernel (and it is good to know that it is linear).
TAPKEE_CALLBACK_IS_KERNEL(pimpl_kernel_callback);

// Distance function callback that provides
// dissimilarity function values on vectors
// given by their indices. This impl. computes
// euclidean distance between two vectors.
template<class Implementation>
struct pimpl_distance_callback
{
	pimpl_distance_callback(Implementation* i) : impl(i) {};
	inline DefaultScalarType operator()(int a, int b) const
	{
		return impl->distance(a,b);
	}
	Implementation* impl;
};
// That's mandatory to specify that distance_callback
// is a distance
TAPKEE_CALLBACK_IS_DISTANCE(pimpl_distance_callback);

#endif
