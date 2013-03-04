/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DIFFUSION_MAPS_H_
#define TAPKEE_DIFFUSION_MAPS_H_

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
                                              IndexType timesteps, ScalarType width)
{
	timed_context context("Diffusion map matrix computation");

	DenseSymmetricMatrix diffusion_matrix(end-begin,end-begin);	
	DenseVector p = DenseVector::Zero(end-begin);

	RESTRICT_ALLOC;

	// compute gaussian kernel matrix
	for (RandomAccessIterator i_iter=begin; i_iter!=end; ++i_iter)
	{
		for (RandomAccessIterator j_iter=i_iter; j_iter!=end; ++j_iter)
		{
			ScalarType k = callback(*i_iter,*j_iter);
			ScalarType gk = exp(-(k*k)/width);
			diffusion_matrix(i_iter-begin,j_iter-begin) = gk;
			diffusion_matrix(j_iter-begin,i_iter-begin) = gk;
		}
	}
	// compute column sum vector
	p = diffusion_matrix.colwise().sum();

	// compute full matrix as we need to compute sum later
	for (IndexType i=0; i<(end-begin); i++)
		for (IndexType j=0; j<(end-begin); j++)
			diffusion_matrix(i,j) /= pow(p(i)*p(j),timesteps);

	// compute sqrt of column sum vector
	p = diffusion_matrix.colwise().sum().cwiseSqrt();
	
	for (IndexType i=0; i<(end-begin); i++)
		for (IndexType j=i; j<(end-begin); j++)
			diffusion_matrix(i,j) /= p(i)*p(j);

	UNRESTRICT_ALLOC;

	return diffusion_matrix;
};

}
}

#endif
