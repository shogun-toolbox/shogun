/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_PCA_H_
#define TAPKEE_PCA_H_

/* Tapkee includes */
#include <lib/tapkee/defines.hpp>
#include <lib/tapkee/utils/time.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

template <class RandomAccessIterator, class FeatureVectorCallback>
DenseMatrix project(const DenseMatrix& projection_matrix, const DenseVector& mean_vector,
                    RandomAccessIterator begin, RandomAccessIterator end,
                    FeatureVectorCallback callback, IndexType dimension)
{
	timed_context context("Data projection");

	DenseVector current_vector(dimension);
	DenseVector current_vector_subtracted_mean(dimension);

	DenseMatrix embedding = DenseMatrix::Zero((end-begin),projection_matrix.cols());

	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		callback.vector(*iter,current_vector);
		current_vector_subtracted_mean = current_vector - mean_vector;
		embedding.row(iter-begin) = projection_matrix.transpose()*current_vector_subtracted_mean;
	}

	return embedding;
}

template <class RandomAccessIterator, class FeatureVectorCallback>
DenseVector compute_mean(RandomAccessIterator begin, RandomAccessIterator end,
                         FeatureVectorCallback callback, IndexType dimension)
{
	DenseVector mean = DenseVector::Zero(dimension);
	DenseVector current_vector(dimension);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		callback.vector(*iter,current_vector);
		mean += current_vector;
	}
	mean.array() /= (end-begin);
	return mean;
}

template <class RandomAccessIterator, class FeatureVectorCallback>
DenseSymmetricMatrix compute_covariance_matrix(RandomAccessIterator begin, RandomAccessIterator end,
                                               const DenseVector& mean, FeatureVectorCallback callback, IndexType dimension)
{
	timed_context context("Constructing PCA covariance matrix");

	DenseSymmetricMatrix covariance_matrix = DenseSymmetricMatrix::Zero(dimension,dimension);

	DenseVector current_vector(dimension);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		callback.vector(*iter,current_vector);
		covariance_matrix.selfadjointView<Eigen::Upper>().rankUpdate(current_vector,1.0);
	}
	covariance_matrix.selfadjointView<Eigen::Upper>().rankUpdate(mean,-1.0);

	return covariance_matrix;
}

template <class RandomAccessIterator, class KernelCallback>
DenseSymmetricMatrix compute_centered_kernel_matrix(RandomAccessIterator begin, RandomAccessIterator end,
                                                    KernelCallback callback)
{
	timed_context context("Constructing kPCA centered kernel matrix");

	DenseSymmetricMatrix kernel_matrix(end-begin,end-begin);

	for (RandomAccessIterator i_iter=begin; i_iter!=end; ++i_iter)
	{
		for (RandomAccessIterator j_iter=i_iter; j_iter!=end; ++j_iter)
		{
			ScalarType k = callback.kernel(*i_iter,*j_iter);
			kernel_matrix(i_iter-begin,j_iter-begin) = k;
			kernel_matrix(j_iter-begin,i_iter-begin) = k;
		}
	}

	centerMatrix(kernel_matrix);

	return kernel_matrix;
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
