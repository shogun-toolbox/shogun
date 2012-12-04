/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn, Fernando J. Iglesias García
 *
 */

#ifndef TAPKEE_EIGEN_CALLBACKS_H_
#define TAPKEE_EIGEN_CALLBACKS_H_

// Here we provide basic but still full set of callbacks
// based on the Eigen3 template matrix library

// Feature vector access callback that provides operation that 
// puts contents of the specified feature 
// vector to given DenseVector instance.
struct feature_vector_callback
{
	feature_vector_callback(const tapkee::DenseMatrix& matrix) : feature_matrix(matrix) {};
	inline void operator()(int i, tapkee::DenseVector& vector) const
	{
		vector = feature_matrix.col(i);
	}
	const tapkee::DenseMatrix& feature_matrix;
};

// Kernel function callback that computes
// similarity function values on vectors 
// given by their indices. This impl. computes 
// linear kernel i.e. dot product between two vectors.
struct kernel_callback
{
	kernel_callback(const tapkee::DenseMatrix& matrix) : feature_matrix(matrix) {};
	inline tapkee::DefaultScalarType operator()(int a, int b) const
	{
		return feature_matrix.col(a).dot(feature_matrix.col(b));
	}
	const tapkee::DenseMatrix& feature_matrix;
};
// That's mandatory to specify that kernel_callback
// is a kernel (and it is good to know that it is linear).
TAPKEE_CALLBACK_IS_LINEAR_KERNEL(kernel_callback);

// Distance function callback that provides
// dissimilarity function values on vectors
// given by their indices. This impl. computes
// euclidean distance between two vectors.
struct distance_callback
{
	distance_callback(const tapkee::DenseMatrix& matrix) : feature_matrix(matrix) {};
	inline tapkee::DefaultScalarType operator()(int a, int b) const
	{
		return (feature_matrix.col(a)-feature_matrix.col(b)).norm();
	}
	const tapkee::DenseMatrix& feature_matrix;
};
// That's mandatory to specify that distance_callback
// is a distance
TAPKEE_CALLBACK_IS_DISTANCE(distance_callback);

#endif
