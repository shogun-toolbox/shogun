/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_LaplacianEigenmaps_H_
#define TAPKEE_LaplacianEigenmaps_H_

/* Tapkee includes */
#include <lib/tapkee/defines.hpp>
#include <lib/tapkee/utils/time.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

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
			DistanceCallback callback, ScalarType width)
{
	SparseTriplets sparse_triplets;

	timed_context context("Laplacian computation");
	const IndexType k = neighbors[0].size();
	sparse_triplets.reserve((k+1)*(end-begin));

	DenseVector D = DenseVector::Zero(end-begin);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];

		for (IndexType i=0; i<k; ++i)
		{
			ScalarType distance = callback.distance(*iter,begin[current_neighbors[i]]);
			ScalarType heat = exp(-distance*distance/width);
			D(iter-begin) += heat;
			D(current_neighbors[i]) += heat;
			sparse_triplets.push_back(SparseTriplet(current_neighbors[i],(iter-begin),-heat));
			sparse_triplets.push_back(SparseTriplet((iter-begin),current_neighbors[i],-heat));
		}
	}
	for (IndexType i=0; i<static_cast<IndexType>(end-begin); ++i)
		sparse_triplets.push_back(SparseTriplet(i,i,D(i)));

#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	Eigen::DynamicSparseMatrix<ScalarType> dynamic_weight_matrix(end-begin,end-begin);
	dynamic_weight_matrix.reserve(sparse_triplets.size());
	for (SparseTriplets::const_iterator it=sparse_triplets.begin(); it!=sparse_triplets.end(); ++it)
		dynamic_weight_matrix.coeffRef(it->col(),it->row()) += it->value();
	SparseWeightMatrix weight_matrix(dynamic_weight_matrix);
#else
	SparseWeightMatrix weight_matrix(end-begin,end-begin);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());
#endif

	return Laplacian(weight_matrix,DenseDiagonalMatrix(D));
}

template<class RandomAccessIterator, class FeatureVectorCallback>
DenseSymmetricMatrixPair construct_locality_preserving_eigenproblem(SparseWeightMatrix& L,
		DenseDiagonalMatrix& D, RandomAccessIterator begin, RandomAccessIterator end, FeatureVectorCallback feature_vector_callback,
		IndexType dimension)
{
	timed_context context("Constructing LPP eigenproblem");

	DenseSymmetricMatrix lhs = DenseSymmetricMatrix::Zero(dimension,dimension);
	DenseSymmetricMatrix rhs = DenseSymmetricMatrix::Zero(dimension,dimension);

	DenseVector rank_update_vector_i(dimension);
	DenseVector rank_update_vector_j(dimension);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		feature_vector_callback.vector(*iter,rank_update_vector_i);
		rhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i,D.diagonal()(iter-begin));
	}

	for (int i=0; i<L.outerSize(); ++i)
	{
		for (SparseWeightMatrix::InnerIterator it(L,i); it; ++it)
		{
			feature_vector_callback.vector(begin[it.row()],rank_update_vector_i);
			feature_vector_callback.vector(begin[it.col()],rank_update_vector_j);
			lhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i, rank_update_vector_j, it.value());
		}
	}

	return DenseSymmetricMatrixPair(lhs,rhs);
}

}
}

#endif
