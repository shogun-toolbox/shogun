/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_EIGEN_CALLBACKS_H_
#define TAPKEE_EIGEN_CALLBACKS_H_

namespace tapkee
{
	// Features callback that provides operation that
	// puts contents of the specified feature
	// vector to given DenseVector instance.
	struct eigen_features_callback
	{
		eigen_features_callback(const tapkee::DenseMatrix& matrix) : feature_matrix(matrix) {};
		inline tapkee::IndexType dimension() const
		{
			return feature_matrix.rows();
		}
		inline void vector(tapkee::IndexType i, tapkee::DenseVector& v) const
		{
			v = feature_matrix.col(i);
		}
		const tapkee::DenseMatrix& feature_matrix;
	};

	// Kernel function callback that computes
	// similarity function values on vectors
	// given by their indices. This impl. computes
	// linear kernel i.e. dot product between two vectors.
	struct eigen_kernel_callback
	{
		eigen_kernel_callback(const tapkee::DenseMatrix& matrix) : feature_matrix(matrix) {};
		inline tapkee::ScalarType kernel(tapkee::IndexType a, tapkee::IndexType b) const
		{
			return feature_matrix.col(a).dot(feature_matrix.col(b));
		}
		inline tapkee::ScalarType operator()(tapkee::IndexType a, tapkee::IndexType b) const
		{
			return kernel(a,b);
		}
		const tapkee::DenseMatrix& feature_matrix;
	};

	// Distance function callback that provides
	// dissimilarity function values on vectors
	// given by their indices. This impl. computes
	// euclidean distance between two vectors.
	struct eigen_distance_callback
	{
		eigen_distance_callback(const tapkee::DenseMatrix& matrix) : feature_matrix(matrix) {};
		inline tapkee::ScalarType distance(tapkee::IndexType a, tapkee::IndexType b) const
		{
			return (feature_matrix.col(a)-feature_matrix.col(b)).norm();
		}
		inline tapkee::ScalarType operator()(tapkee::IndexType a, tapkee::IndexType b) const
		{
			return distance(a,b);
		}
		const tapkee::DenseMatrix& feature_matrix;
};

}

#endif
