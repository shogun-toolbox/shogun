/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn, Fernando J. Iglesias García
 *
 */

#ifndef TAPKEE_PRECOMPUTED_CALLBACKS_H_
#define TAPKEE_PRECOMPUTED_CALLBACKS_H_

// Here we provide basic but still full set of callbacks
// based on the Eigen3 template matrix library

// Kernel function callback that computes
// similarity function values on vectors 
// given by their indices. This impl. computes 
// linear kernel i.e. dot product between two vectors.
struct precomputed_kernel_callback
{
	precomputed_kernel_callback(const tapkee::DenseMatrix& matrix) : kernel_matrix(matrix) {};
	inline tapkee::DefaultScalarType operator()(int a, int b) const
	{
		return kernel_matrix(a,b);
	}
	const tapkee::DenseMatrix& kernel_matrix;
};
// That's mandatory to specify that kernel_callback
// is a kernel (and it is good to know that it is linear).
TAPKEE_CALLBACK_IS_KERNEL(precomputed_kernel_callback);

// Distance function callback that provides
// dissimilarity function values on vectors
// given by their indices. This impl. computes
// euclidean distance between two vectors.
struct precomputed_distance_callback
{
	precomputed_distance_callback(const tapkee::DenseMatrix& matrix) : distance_matrix(matrix) {};
	inline tapkee::DefaultScalarType operator()(int a, int b) const
	{
		return distance_matrix(a,b);
	}
	const tapkee::DenseMatrix& distance_matrix;
};
// That's mandatory to specify that distance_callback
// is a distance
TAPKEE_CALLBACK_IS_DISTANCE(precomputed_distance_callback);
#endif

