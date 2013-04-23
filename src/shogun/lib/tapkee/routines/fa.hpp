/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2013 Sergey Lisitsyn, Fernando J. Iglesias Garcia
 */

#ifndef TAPKEE_FA_H_
#define TAPKEE_FA_H_

namespace tapkee
{
namespace tapkee_internal
{

template <class RandomAccessIterator, class FeatureVectorCallback>
DenseMatrix project(RandomAccessIterator begin, RandomAccessIterator end, FeatureVectorCallback callback,
		IndexType dimension, const IndexType max_iter, const ScalarType epsilon,
		const IndexType target_dimension, const DenseVector& mean_vector)
{
	timed_context context("Data projection");

	// The number of data points
	const IndexType n = end-begin;

	// Dense representation of the data points

	DenseVector current_vector(dimension);

	DenseMatrix X = DenseMatrix::Zero(dimension,n);

	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		callback.vector(*iter,current_vector);
		X.col(iter-begin) = current_vector - mean_vector;
	}

	// Initialize FA model

	// Initial variances
	DenseMatrix sig = DenseMatrix::Identity(dimension,dimension);
	// Initial linear mapping
	DenseMatrix A = DenseMatrix::Random(dimension, target_dimension).cwiseAbs();

	// Main loop
	IndexType iter = 0;
	DenseMatrix invC,M,SC;
	ScalarType ll = 0, newll = 0;
	while (iter < max_iter)
	{
		++iter;

		// Perform E-step

		// Compute the inverse of the covariance matrix
		invC = (A*A.transpose() + sig).inverse();
		M = A.transpose()*invC*X;
		SC = n*(DenseMatrix::Identity(target_dimension,target_dimension) - A.transpose()*invC*A) + M*M.transpose();

		// Perform M-step
		A = (X*M.transpose())*SC.inverse();
		sig = DenseMatrix(((X*X.transpose() - A*M*X.transpose()).diagonal()/n).asDiagonal()).array() + epsilon;

		// Compute log-likelihood of FA model
		newll = 0.5*(log(invC.determinant()) - (invC*X).cwiseProduct(X).sum()/n);

		// Check for convergence
		if ((iter > 1) && (fabs(newll - ll) < epsilon))
			break;

		ll = newll;
	}

	return X.transpose()*A;
}

} /* namespace tapkee */
} /* namespace tapkee_internal */

#endif /* TAPKEE_FA_H_ */
