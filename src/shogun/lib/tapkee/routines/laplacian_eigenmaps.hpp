/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_LAPLACIAN_EIGENMAPS_H_
#define TAPKEE_LAPLACIAN_EIGENMAPS_H_
	
//! Computes laplacian of neighborhood graph.
//!
//! Follows the algorithm described below:
//! <ul>
//! <li> For each vector compute gaussian exp of distances to its neighbor vectors and 
//!      put it to sparse matrix \f$ L_{i,N_i(j)} = \exp\left( - \frac{d(x_i,x_{N_i(j)})^2}{w} \right) \f$.
//! <li> Symmetrize matrix \f$ L \f$ with \f$ L_{i,j} = \max (L_{i,j}, L_{j,i}) \f$ to
//!      make neighborhood relationship symmetric.
//! <li> Compute sum vector \f$ D = \sum_{i} L_{i,j} \f$.
//! <li> Modify \f$ L = D - L \f$.
//! <li> Output matrix sparse matrix \f$ L \f$ and diagonal matrix built of vector \f$ D \f$.
//! </ul>
//!
//! @param begin begin data iterator
//! @param end end data iterator
//! @param neighbors neighbors of each vector
//! @param callback distance callback
//! @param width width \f$ w \f$ of the gaussian kernel
//!
template<class RandomAccessIterator, class DistanceCallback>
Laplacian compute_laplacian(RandomAccessIterator begin, 
			RandomAccessIterator end,const Neighbors& neighbors, 
			DistanceCallback callback, DefaultScalarType width)
{
	SparseTriplets sparse_triplets;

	timed_context context("Laplacian computation");
	const unsigned int k = neighbors[0].size();
	sparse_triplets.reserve(k*(end-begin));

	DenseVector D = DenseVector::Zero(end-begin);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];

		for (unsigned int i=0; i<k; ++i)
		{
			DefaultScalarType distance = callback(*iter,begin[current_neighbors[i]]);
			DefaultScalarType heat = exp(-distance*distance/width);
			D(iter-begin) += heat;
			//sparse_triplets.push_back(SparseTriplet(begin[current_neighbors[i]],(iter-begin),-heat));
			sparse_triplets.push_back(SparseTriplet((iter-begin),current_neighbors[i],-heat));
		}
	}
	for (unsigned int i=0; i<(end-begin); ++i)
		sparse_triplets.push_back(SparseTriplet(i,i,D(i)));

	SparseWeightMatrix weight_matrix(end-begin,end-begin);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());
	weight_matrix.cwiseMax(SparseWeightMatrix(weight_matrix.transpose()));

	return Laplacian(weight_matrix,DenseDiagonalMatrix(D));
}

template<class RandomAccessIterator, class FeatureVectorCallback>
DenseSymmetricMatrixPair construct_locality_preserving_eigenproblem(SparseWeightMatrix L,
		DenseDiagonalMatrix D, RandomAccessIterator begin, RandomAccessIterator end, FeatureVectorCallback feature_vector_callback,
		unsigned int dimension)
{
	timed_context context("Constructing LPP eigenproblem");

	DenseSymmetricMatrix lhs = DenseSymmetricMatrix::Zero(dimension,dimension);
	DenseSymmetricMatrix rhs = DenseSymmetricMatrix::Zero(dimension,dimension);

	DenseVector rank_update_vector_i(dimension);
	DenseVector rank_update_vector_j(dimension);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		feature_vector_callback(*iter,rank_update_vector_i);
		rhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i,D.diagonal()(iter-begin));
	}

	for (int i=0; i<L.outerSize(); ++i)
	{
		for (SparseWeightMatrix::InnerIterator it(L,i); it; ++it)
		{
			feature_vector_callback(begin[it.row()],rank_update_vector_i);
			feature_vector_callback(begin[it.col()],rank_update_vector_j);
			lhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i, rank_update_vector_j, it.value());
		}
	}

	return DenseSymmetricMatrixPair(lhs,rhs);
}

#endif
