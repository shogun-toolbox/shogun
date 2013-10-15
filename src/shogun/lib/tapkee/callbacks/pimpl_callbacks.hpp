/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
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
	inline tapkee::ScalarType kernel(int a, int b) const
	{
		return impl->kernel(a,b);
	}
	Implementation* impl;
};

// Distance function callback that provides
// dissimilarity function values on vectors
// given by their indices. This impl. computes
// euclidean distance between two vectors.
template<class Implementation>
struct pimpl_distance_callback
{
	pimpl_distance_callback(Implementation* i) : impl(i) {};
	inline tapkee::ScalarType distance(int a, int b) const
	{
		return impl->distance(a,b);
	}
	Implementation* impl;
};

#endif
