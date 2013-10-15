/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_PRECOMPUTED_CALLBACKS_H_
#define TAPKEE_PRECOMPUTED_CALLBACKS_H_

namespace tapkee
{
// Here we provide basic but still full set of callbacks
// based on the Eigen3 template matrix library

// Kernel function callback that computes
// similarity function values on vectors
// given by their indices. This impl. computes
// linear kernel i.e. dot product between two vectors.
struct precomputed_kernel_callback
{
	precomputed_kernel_callback(const tapkee::DenseMatrix& matrix) : kernel_matrix(matrix) {};
	inline tapkee::ScalarType kernel(int a, int b) const
	{
		return kernel_matrix(a,b);
	}
	const tapkee::DenseMatrix& kernel_matrix;
};

// Distance function callback that provides
// dissimilarity function values on vectors
// given by their indices. This impl. computes
// euclidean distance between two vectors.
struct precomputed_distance_callback
{
	precomputed_distance_callback(const tapkee::DenseMatrix& matrix) : distance_matrix(matrix) {};
	inline tapkee::ScalarType distance(int a, int b) const
	{
		return distance_matrix(a,b);
	}
	const tapkee::DenseMatrix& distance_matrix;
};

}
#endif

