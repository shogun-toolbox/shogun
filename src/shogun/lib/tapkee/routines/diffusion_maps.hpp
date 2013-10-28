/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DiffusionMapS_H_
#define TAPKEE_DiffusionMapS_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/defines.hpp>
#include <shogun/lib/tapkee/utils/time.hpp>
/* End of Tapke includes */

namespace tapkee
{
namespace tapkee_internal
{

//! Computes diffusion process matrix. Uses the following algorithm:
//!
//! <ol>
//! <li> Compute matrix \f$ K \f$ such as \f$ K_{i,j} = \exp\left(-\frac{d(x_i,x_j)^2}{w}\right) \f$.
//! <li> Compute sum vector \f$ p = \sum_i K_{i,j}\f$.
//! <li> Modify \f$ K \f$ with \f$ K_{i,j} = K_{i,j} / (p_i  p_j)^t \f$.
//! <li> Compute sum vector \f$ p = \sum_i K_{i,j}\f$ again.
//! <li> Normalize \f$ K \f$ with \f$ K_{i,j} = K_{i,j} / (p_i p_j) \f$.
//! </ol>
//!
//! @param begin begin data iterator
//! @param end end data iterator
//! @param callback distance callback
//! @param timesteps number of timesteps \f$ t \f$ of diffusion process
//! @param width width \f$ w \f$ of the gaussian kernel
//!
template <class RandomAccessIterator, class DistanceCallback>
DenseSymmetricMatrix compute_diffusion_matrix(RandomAccessIterator begin, RandomAccessIterator end, DistanceCallback callback,
                                              const IndexType timesteps, const ScalarType width)
{
	timed_context context("Diffusion map matrix computation");

	const IndexType n_vectors = end-begin;
	DenseSymmetricMatrix diffusion_matrix(n_vectors,n_vectors);
	DenseVector p = DenseVector::Zero(n_vectors);

	RESTRICT_ALLOC;

	// compute gaussian kernel matrix
#pragma omp parallel shared(diffusion_matrix,begin,callback) default(none)
	{
		IndexType i_index_iter, j_index_iter;
#pragma omp for nowait
		for (i_index_iter=0; i_index_iter<n_vectors; ++i_index_iter)
		{
			for (j_index_iter=i_index_iter; j_index_iter<n_vectors; ++j_index_iter)
			{
				ScalarType k = callback.distance(begin[i_index_iter],begin[j_index_iter]);
				ScalarType gk = exp(-(k*k)/width);
				diffusion_matrix(i_index_iter,j_index_iter) = gk;
				diffusion_matrix(j_index_iter,i_index_iter) = gk;
			}
		}
	}
	// compute column sum vector
	p = diffusion_matrix.colwise().sum();

	// compute full matrix as we need to compute sum later
	for (IndexType i=0; i<n_vectors; i++)
		for (IndexType j=0; j<n_vectors; j++)
			diffusion_matrix(i,j) /= pow(p(i)*p(j),timesteps);

	// compute sqrt of column sum vector
	p = diffusion_matrix.colwise().sum().cwiseSqrt();

	for (IndexType i=0; i<n_vectors; i++)
		for (IndexType j=i; j<n_vectors; j++)
			diffusion_matrix(i,j) /= p(i)*p(j);

	UNRESTRICT_ALLOC;

	return diffusion_matrix;
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
