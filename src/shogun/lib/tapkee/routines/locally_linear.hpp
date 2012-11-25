/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_LOCALLY_LINEAR_H_
#define TAPKEE_LOCALLY_LINEAR_H_

#include "eigen_embedding.hpp"
#include "../tapkee_defines.hpp"

template <class RandomAccessIterator, class PairwiseCallback>
SparseWeightMatrix kltsa_weight_matrix(RandomAccessIterator begin, RandomAccessIterator end, 
                                       const Neighbors& neighbors, PairwiseCallback callback, 
                                       unsigned int target_dimension, DefaultScalarType shift,
                                       bool partial_eigendecomposer=false)
{
	timed_context context("KLTSA weight matrix computation");
	const unsigned int k = neighbors[0].size();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve((k*k+k+1)*(end-begin));

	RandomAccessIterator iter;
	RandomAccessIterator iter_begin = begin, iter_end = end;
	DenseMatrix gram_matrix = DenseMatrix::Zero(k,k);
	DenseVector rhs = DenseVector::Ones(k);
	DenseMatrix G = DenseMatrix::Zero(k,target_dimension+1);
	G.col(0).setConstant(1/sqrt(DefaultScalarType(k)));
	DefaultDenseSelfAdjointEigenSolver solver;

	RESTRICT_ALLOC;
//#pragma omp parallel for private(iter,gram_matrix,G)
	for (iter=iter_begin; iter<iter_end; ++iter)
	{
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];
	
		for (unsigned int i=0; i<k; ++i)
		{
			for (unsigned int j=i; j<k; ++j)
			{
				DefaultScalarType kij = callback(begin[current_neighbors[i]],begin[current_neighbors[j]]);
				gram_matrix(i,j) = kij;
				gram_matrix(j,i) = kij;
			}
		}
		
		gram_matrix.centerMatrix();

		UNRESTRICT_ALLOC;
		if (partial_eigendecomposer)
		{
			G.rightCols(target_dimension).noalias() = eigen_embedding<DenseMatrix,DenseMatrixOperation>(ARPACK,gram_matrix,target_dimension,0).first;
		}
		else
		{
			solver.compute(gram_matrix);
			G.rightCols(target_dimension).noalias() = solver.eigenvectors().rightCols(target_dimension);
		}
		RESTRICT_ALLOC;
		gram_matrix.noalias() = G * G.transpose();
		
		sparse_triplets.push_back(SparseTriplet(iter-begin,iter-begin,shift));
		for (unsigned int i=0; i<k; ++i)
		{
			sparse_triplets.push_back(SparseTriplet(current_neighbors[i],current_neighbors[i],1.0));
			for (unsigned int j=0; j<k; ++j)
				sparse_triplets.push_back(SparseTriplet(current_neighbors[i],current_neighbors[j],
				                                        -gram_matrix(i,j)));
		}
	}
	UNRESTRICT_ALLOC;

	SparseWeightMatrix weight_matrix(end-begin,end-begin);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());

	return weight_matrix;
}

template <class RandomAccessIterator, class PairwiseCallback>
SparseWeightMatrix klle_weight_matrix(RandomAccessIterator begin, RandomAccessIterator end, 
                                      const Neighbors& neighbors, PairwiseCallback callback, DefaultScalarType shift)
{
	timed_context context("KLLE weight computation");
	const unsigned int k = neighbors[0].size();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve((k*k+2*k+1)*(end-begin));

	RandomAccessIterator iter;
	RandomAccessIterator iter_begin = begin, iter_end = end;
	DenseMatrix gram_matrix = DenseMatrix::Zero(k,k);
	DenseVector dots(k);
	DenseVector rhs = DenseVector::Ones(k);
	DenseVector weights;
	
	RESTRICT_ALLOC;
	for (RandomAccessIterator iter=iter_begin; iter!=iter_end; ++iter)
	{
		DefaultScalarType kernel_value = callback(*iter,*iter);
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];
		
		for (unsigned int i=0; i<k; ++i)
			dots[i] = callback(*iter, begin[current_neighbors[i]]);

		for (unsigned int i=0; i<k; ++i)
		{
			for (unsigned int j=i; j<k; ++j)
				gram_matrix(i,j) = kernel_value - dots(i) - dots(j) + callback(begin[current_neighbors[i]],begin[current_neighbors[j]]);
		}
		
		DefaultScalarType trace = gram_matrix.trace();
		gram_matrix.diagonal().array() += 1e-3*trace;
		weights = gram_matrix.selfadjointView<Eigen::Upper>().ldlt().solve(rhs);
		weights /= weights.sum();

		sparse_triplets.push_back(SparseTriplet(iter-begin,iter-begin,1.0+shift));
		for (unsigned int i=0; i<k; ++i)
		{
			sparse_triplets.push_back(SparseTriplet(current_neighbors[i],iter-begin,
			                                        -weights[i]));
			sparse_triplets.push_back(SparseTriplet(iter-begin,current_neighbors[i],
			                                        -weights[i]));
			for (unsigned int j=0; j<k; ++j)
				sparse_triplets.push_back(SparseTriplet(current_neighbors[i],current_neighbors[j],
				                                        +weights(i)*weights(j)));
		}
	}
	UNRESTRICT_ALLOC;

	SparseWeightMatrix weight_matrix(end-begin,end-begin);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());

	return weight_matrix;
}

	template <class RandomAccessIterator, class PairwiseCallback>
SparseWeightMatrix hlle_weight_matrix(RandomAccessIterator begin, RandomAccessIterator end, 
                                      const Neighbors& neighbors, PairwiseCallback callback, unsigned int target_dimension)
{
	timed_context context("KLTSA weight matrix computation");
	const unsigned int k = neighbors[0].size();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve(k*k*(end-begin));

	RandomAccessIterator iter_begin = begin, iter_end = end;
	DenseMatrix gram_matrix = DenseMatrix::Zero(k,k);
	DenseVector rhs = DenseVector::Ones(k);
	DenseMatrix G = DenseMatrix::Zero(k,target_dimension+1);

	RandomAccessIterator iter;
	for (iter=iter_begin; iter!=iter_end; ++iter)
	{
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];
	
		for (unsigned int i=0; i<k; ++i)
		{
			for (unsigned int j=i; j<k; ++j)
			{
				DefaultScalarType kij = callback(begin[current_neighbors[i]],begin[current_neighbors[j]]);
				gram_matrix(i,j) = kij;
				gram_matrix(j,i) = kij;
			}
		}
		
		DefaultDenseSelfAdjointEigenSolver sae_solver;
		sae_solver.compute(gram_matrix);

		G.col(0).setConstant(1/sqrt(DefaultScalarType(k)));
		G.rightCols(target_dimension).noalias() = sae_solver.eigenvectors().rightCols(target_dimension);
		gram_matrix = G * G.transpose();
		
		sparse_triplets.push_back(SparseTriplet(iter-begin,iter-begin,1e-8));
		for (unsigned int i=0; i<k; ++i)
		{
			sparse_triplets.push_back(SparseTriplet(current_neighbors[i],current_neighbors[i],1.0));
			for (unsigned int j=0; j<k; ++j)
				sparse_triplets.push_back(SparseTriplet(current_neighbors[i],current_neighbors[j],
				                                        -gram_matrix(i,j)));
		}
	}

	SparseWeightMatrix weight_matrix(end-begin,end-begin);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());

	return weight_matrix;
};

template<class RandomAccessIterator, class FeatureVectorCallback>
DenseSymmetricMatrixPair construct_neighborhood_preserving_eigenproblem(SparseWeightMatrix W,
		RandomAccessIterator begin, RandomAccessIterator end, FeatureVectorCallback feature_vector_callback,
		unsigned int dimension)
{
	timed_context context("NPE eigenproblem construction");
	
	DenseSymmetricMatrix lhs = DenseSymmetricMatrix::Zero(dimension,dimension);
	DenseSymmetricMatrix rhs = DenseSymmetricMatrix::Zero(dimension,dimension);

	DenseVector rank_update_vector_i(dimension);
	DenseVector rank_update_vector_j(dimension);

	RESTRICT_ALLOC;
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		feature_vector_callback(*iter,rank_update_vector_i);
		rhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i);
	}

	for (int i=0; i<W.outerSize(); ++i)
	{
		for (SparseWeightMatrix::InnerIterator it(W,i); it; ++it)
		{
			feature_vector_callback(begin[it.row()],rank_update_vector_i);
			feature_vector_callback(begin[it.col()],rank_update_vector_j);
			lhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i, rank_update_vector_j, it.value());
		}
	}
	
	rhs += rhs.transpose();
	rhs /= 2;

	UNRESTRICT_ALLOC;

	return DenseSymmetricMatrixPair(lhs,rhs);
}

template<class RandomAccessIterator, class FeatureVectorCallback>
DenseSymmetricMatrixPair construct_lltsa_eigenproblem(SparseWeightMatrix W,
		RandomAccessIterator begin, RandomAccessIterator end, FeatureVectorCallback feature_vector_callback,
		unsigned int dimension)
{
	timed_context context("LLTSA eigenproblem construction");

	DenseSymmetricMatrix lhs = DenseSymmetricMatrix::Zero(dimension,dimension);
	DenseSymmetricMatrix rhs = DenseSymmetricMatrix::Zero(dimension,dimension);

	DenseVector rank_update_vector_i(dimension);
	DenseVector rank_update_vector_j(dimension);
	DenseVector sum = DenseVector::Zero(dimension);
	
	RESTRICT_ALLOC;
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		feature_vector_callback(*iter,rank_update_vector_i);
		sum += rank_update_vector_i;
		rhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i);
	}
	rhs.selfadjointView<Eigen::Upper>().rankUpdate(sum,-1./(end-begin));

	for (int i=0; i<W.outerSize(); ++i)
	{
		for (SparseWeightMatrix::InnerIterator it(W,i); it; ++it)
		{
			feature_vector_callback(begin[it.row()],rank_update_vector_i);
			feature_vector_callback(begin[it.col()],rank_update_vector_j);
			lhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i, rank_update_vector_j, it.value());
		}
	}
	lhs.selfadjointView<Eigen::Upper>().rankUpdate(sum,-1./(end-begin));

	rhs += rhs.transpose();
	rhs /= 2;

	UNRESTRICT_ALLOC;

	return DenseSymmetricMatrixPair(lhs,rhs);
}

#endif
