/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_LOCALLY_LINEAR_H_
#define TAPKEE_LOCALLY_LINEAR_H_

/* Tapkee includes */
#include <lib/tapkee/routines/eigendecomposition.hpp>
#include <lib/tapkee/defines.hpp>
#include <lib/tapkee/utils/matrix.hpp>
#include <lib/tapkee/utils/time.hpp>
#include <lib/tapkee/utils/sparse.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{



template <class RandomAccessIterator, class PairwiseCallback>
SparseWeightMatrix tangent_weight_matrix(RandomAccessIterator begin, RandomAccessIterator end,
                                         const Neighbors& neighbors, PairwiseCallback callback,
                                         const IndexType target_dimension, const ScalarType shift,
                                         const bool partial_eigendecomposer=false)
{
	timed_context context("KLTSA weight matrix computation");
	const IndexType k = neighbors[0].size();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve((k*k+2*k+1)*(end-begin));

#pragma omp parallel shared(begin,end,neighbors,callback,sparse_triplets) default(none)
	{
		IndexType index_iter;
		DenseMatrix gram_matrix = DenseMatrix::Zero(k,k);
		DenseVector rhs = DenseVector::Ones(k);
		DenseMatrix G = DenseMatrix::Zero(k,target_dimension+1);
		G.col(0).setConstant(1/sqrt(static_cast<ScalarType>(k)));
		DenseSelfAdjointEigenSolver solver;
		SparseTriplets local_triplets;
		local_triplets.reserve(k*k+2*k+1);

#pragma omp for nowait
		for (index_iter=0; index_iter<static_cast<IndexType>(end-begin); index_iter++)
		{
			const LocalNeighbors& current_neighbors = neighbors[index_iter];

			for (IndexType i=0; i<k; ++i)
			{
				for (IndexType j=i; j<k; ++j)
				{
					ScalarType kij = callback.kernel(begin[current_neighbors[i]],begin[current_neighbors[j]]);
					gram_matrix(i,j) = kij;
					gram_matrix(j,i) = kij;
				}
			}

			centerMatrix(gram_matrix);

			//UNRESTRICT_ALLOC;
#ifdef TAPKEE_WITH_ARPACK
			if (partial_eigendecomposer)
			{
				G.rightCols(target_dimension).noalias() =
					eigendecomposition<DenseMatrix,DenseMatrixOperation>(Arpack,gram_matrix,target_dimension,0).first;
			}
			else
#endif
			{
				solver.compute(gram_matrix);
				G.rightCols(target_dimension).noalias() = solver.eigenvectors().rightCols(target_dimension);
			}
			//RESTRICT_ALLOC;
			gram_matrix.noalias() = G * G.transpose();

			SparseTriplet diagonal_triplet(index_iter,index_iter,shift);
			local_triplets.push_back(diagonal_triplet);
			for (IndexType i=0; i<k; ++i)
			{
				SparseTriplet neighborhood_diagonal_triplet(current_neighbors[i],current_neighbors[i],1.0);
				local_triplets.push_back(neighborhood_diagonal_triplet);

				for (IndexType j=0; j<k; ++j)
				{
					SparseTriplet tangent_triplet(current_neighbors[i],current_neighbors[j],-gram_matrix(i,j));
					local_triplets.push_back(tangent_triplet);
				}
			}
#pragma omp critical
			{
				copy(local_triplets.begin(),local_triplets.end(),back_inserter(sparse_triplets));
			}

			local_triplets.clear();
		}
	}

	return sparse_matrix_from_triplets(sparse_triplets, end-begin, end-begin);
}

template <class RandomAccessIterator, class PairwiseCallback>
SparseWeightMatrix linear_weight_matrix(const RandomAccessIterator& begin, const RandomAccessIterator& end,
                                        const Neighbors& neighbors, PairwiseCallback callback,
                                        const ScalarType shift, const ScalarType trace_shift)
{
	timed_context context("KLLE weight computation");
	const IndexType k = neighbors[0].size();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve((k*k+2*k+1)*(end-begin));

#pragma omp parallel shared(begin,end,neighbors,callback,sparse_triplets) default(none)
	{
		IndexType index_iter;
		DenseMatrix gram_matrix = DenseMatrix::Zero(k,k);
		DenseVector dots(k);
		DenseVector rhs = DenseVector::Ones(k);
		DenseVector weights;
		SparseTriplets local_triplets;
		local_triplets.reserve(k*k+2*k+1);

		//RESTRICT_ALLOC;
#pragma omp for nowait
		for (index_iter=0; index_iter<static_cast<IndexType>(end-begin); index_iter++)
		{
			ScalarType kernel_value = callback.kernel(begin[index_iter],begin[index_iter]);
			const LocalNeighbors& current_neighbors = neighbors[index_iter];

			for (IndexType i=0; i<k; ++i)
				dots[i] = callback.kernel(begin[index_iter], begin[current_neighbors[i]]);

			for (IndexType i=0; i<k; ++i)
			{
				for (IndexType j=i; j<k; ++j)
					gram_matrix(i,j) = kernel_value - dots(i) - dots(j) +
					                   callback.kernel(begin[current_neighbors[i]],begin[current_neighbors[j]]);
			}

			ScalarType trace = gram_matrix.trace();
			gram_matrix.diagonal().array() += trace_shift*trace;
			weights = gram_matrix.selfadjointView<Eigen::Upper>().ldlt().solve(rhs);
			weights /= weights.sum();

			SparseTriplet diagonal_triplet(index_iter,index_iter,1.0+shift);
			local_triplets.push_back(diagonal_triplet);
			for (IndexType i=0; i<k; ++i)
			{
				SparseTriplet row_side_triplet(current_neighbors[i],index_iter,-weights[i]);
				SparseTriplet col_side_triplet(index_iter,current_neighbors[i],-weights[i]);
				local_triplets.push_back(row_side_triplet);
				local_triplets.push_back(col_side_triplet);
				for (IndexType j=0; j<k; ++j)
				{
					SparseTriplet cross_triplet(current_neighbors[i],current_neighbors[j],weights(i)*weights(j));
					local_triplets.push_back(cross_triplet);
				}
			}

#pragma omp critical
			{
				copy(local_triplets.begin(),local_triplets.end(),back_inserter(sparse_triplets));
			}

			local_triplets.clear();
		}
		//UNRESTRICT_ALLOC;
	}

	return sparse_matrix_from_triplets(sparse_triplets, end-begin, end-begin);
}

template <class RandomAccessIterator, class PairwiseCallback>
SparseWeightMatrix hessian_weight_matrix(RandomAccessIterator begin, RandomAccessIterator end,
                                         const Neighbors& neighbors, PairwiseCallback callback,
                                         const IndexType target_dimension)
{
	timed_context context("Hessian weight matrix computation");
	const IndexType k = neighbors[0].size();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve(k*k*(end-begin));

	const IndexType dp = target_dimension*(target_dimension+1)/2;

#pragma omp parallel shared(begin,end,neighbors,callback,sparse_triplets) default(none)
	{
		IndexType index_iter;
		DenseMatrix gram_matrix = DenseMatrix::Zero(k,k);
		DenseMatrix Yi(k,1+target_dimension+dp);

		SparseTriplets local_triplets;
		local_triplets.reserve(k*k+2*k+1);

#pragma omp for nowait
		for (index_iter=0; index_iter<static_cast<IndexType>(end-begin); index_iter++)
		{
			const LocalNeighbors& current_neighbors = neighbors[index_iter];

			for (IndexType i=0; i<k; ++i)
			{
				for (IndexType j=i; j<k; ++j)
				{
					ScalarType kij = callback.kernel(begin[current_neighbors[i]],begin[current_neighbors[j]]);
					gram_matrix(i,j) = kij;
					gram_matrix(j,i) = kij;
				}
			}

			centerMatrix(gram_matrix);

			DenseSelfAdjointEigenSolver sae_solver;
			sae_solver.compute(gram_matrix);

			Yi.col(0).setConstant(1.0);
			Yi.block(0,1,k,target_dimension).noalias() = sae_solver.eigenvectors().rightCols(target_dimension);

			IndexType ct = 0;
			for (IndexType j=0; j<target_dimension; ++j)
			{
				for (IndexType p=0; p<target_dimension-j; ++p)
				{
					Yi.col(ct+p+1+target_dimension).noalias() = Yi.col(j+1).cwiseProduct(Yi.col(j+p+1));
				}
				ct += ct + target_dimension - j;
			}

			for (IndexType i=0; i<static_cast<IndexType>(Yi.cols()); i++)
			{
				for (IndexType j=0; j<i; j++)
				{
					ScalarType r = Yi.col(i).dot(Yi.col(j));
					Yi.col(i) -= r*Yi.col(j);
				}
				ScalarType norm = Yi.col(i).norm();
				Yi.col(i) *= (1.f / norm);
			}
			for (IndexType i=0; i<dp; i++)
			{
				ScalarType colsum = Yi.col(1+target_dimension+i).sum();
				if (colsum > 1e-4)
					Yi.col(1+target_dimension+i).array() /= colsum;
			}

			// reuse gram matrix storage m'kay?
			gram_matrix.noalias() = Yi.rightCols(dp)*(Yi.rightCols(dp).transpose());

			for (IndexType i=0; i<k; ++i)
			{
				for (IndexType j=0; j<k; ++j)
				{
					SparseTriplet hessian_triplet(current_neighbors[i],current_neighbors[j],gram_matrix(i,j));
					local_triplets.push_back(hessian_triplet);
				}
			}

			#pragma omp critical
			{
				copy(local_triplets.begin(),local_triplets.end(),back_inserter(sparse_triplets));
			}

			local_triplets.clear();
		}
	}

	return sparse_matrix_from_triplets(sparse_triplets, end-begin, end-begin);
}

template<class RandomAccessIterator, class FeatureVectorCallback>
DenseSymmetricMatrixPair construct_neighborhood_preserving_eigenproblem(SparseWeightMatrix W,
		RandomAccessIterator begin, RandomAccessIterator end, FeatureVectorCallback feature_vector_callback,
		IndexType dimension)
{
	timed_context context("NPE eigenproblem construction");

	DenseSymmetricMatrix lhs = DenseSymmetricMatrix::Zero(dimension,dimension);
	DenseSymmetricMatrix rhs = DenseSymmetricMatrix::Zero(dimension,dimension);

	DenseVector rank_update_vector_i(dimension);
	DenseVector rank_update_vector_j(dimension);

	//RESTRICT_ALLOC;
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		feature_vector_callback.vector(*iter,rank_update_vector_i);
		rhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i);
	}

	for (int i=0; i<W.outerSize(); ++i)
	{
		for (SparseWeightMatrix::InnerIterator it(W,i); it; ++it)
		{
			feature_vector_callback.vector(begin[it.row()],rank_update_vector_i);
			feature_vector_callback.vector(begin[it.col()],rank_update_vector_j);
			lhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i, rank_update_vector_j, it.value());
		}
	}

	rhs += rhs.transpose().eval();
	rhs /= 2;

	//UNRESTRICT_ALLOC;

	return DenseSymmetricMatrixPair(lhs,rhs);
}

template<class RandomAccessIterator, class FeatureVectorCallback>
DenseSymmetricMatrixPair construct_lltsa_eigenproblem(SparseWeightMatrix W,
		RandomAccessIterator begin, RandomAccessIterator end, FeatureVectorCallback feature_vector_callback,
		IndexType dimension)
{
	timed_context context("LLTSA eigenproblem construction");

	DenseSymmetricMatrix lhs = DenseSymmetricMatrix::Zero(dimension,dimension);
	DenseSymmetricMatrix rhs = DenseSymmetricMatrix::Zero(dimension,dimension);

	DenseVector rank_update_vector_i(dimension);
	DenseVector rank_update_vector_j(dimension);
	DenseVector sum = DenseVector::Zero(dimension);

	//RESTRICT_ALLOC;
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		feature_vector_callback.vector(*iter,rank_update_vector_i);
		sum += rank_update_vector_i;
		rhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i);
	}
	rhs.selfadjointView<Eigen::Upper>().rankUpdate(sum,-1./(end-begin));

	for (int i=0; i<W.outerSize(); ++i)
	{
		for (SparseWeightMatrix::InnerIterator it(W,i); it; ++it)
		{
			feature_vector_callback.vector(begin[it.row()],rank_update_vector_i);
			feature_vector_callback.vector(begin[it.col()],rank_update_vector_j);
			lhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i, rank_update_vector_j, it.value());
		}
	}
	lhs.selfadjointView<Eigen::Upper>().rankUpdate(sum,-1./(end-begin));

	rhs += rhs.transpose().eval();
	rhs /= 2;

	//UNRESTRICT_ALLOC;

	return DenseSymmetricMatrixPair(lhs,rhs);
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
