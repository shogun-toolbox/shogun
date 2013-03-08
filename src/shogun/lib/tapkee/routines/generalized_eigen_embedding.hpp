/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_GENERALIZED_EIGEN_EMBEDDING_H_
#define TAPKEE_GENERALIZED_EIGEN_EMBEDDING_H_

#ifndef TAPKEE_NO_ARPACK
	#include <shogun/lib/tapkee/utils/arpack_wrapper.hpp>
#endif
#include <shogun/lib/tapkee/routines/matrix_operations.hpp>

namespace tapkee
{
namespace tapkee_internal
{

//! Templated implementation of eigendecomposition-based embedding. 
template <class LMatrixType, class RMatrixType, class MatrixOperationType, int IMPLEMENTATION> 
struct generalized_eigen_embedding_impl
{
	/** Construct embedding
	 * @param wm weight matrix to eigendecompose
	 * @param target_dimension target dimension of embedding (number of eigenvectors to find)
	 * @param skip number of eigenvectors to skip
	 */
	virtual EmbeddingResult embed(const LMatrixType& lhs, const RMatrixType& rhs, IndexType target_dimension, unsigned int skip);
};

//! ARPACK implementation of eigendecomposition-based embedding
template <class LMatrixType, class RMatrixType, class MatrixOperationType> 
struct generalized_eigen_embedding_impl<LMatrixType, RMatrixType, MatrixOperationType, ARPACK>
{
	EmbeddingResult embed(const LMatrixType& lhs, const RMatrixType& rhs, IndexType target_dimension, unsigned int skip)
	{
		timed_context context("ARPACK DSXUPD generalized eigendecomposition");

#ifndef TAPKEE_NO_ARPACK
		ArpackGeneralizedSelfAdjointEigenSolver<LMatrixType, RMatrixType, MatrixOperationType> 
			arpack(lhs,rhs,target_dimension+skip,"SM");
		
		if (arpack.info() == Eigen::Success)
		{
			stringstream ss;
			ss << "Took " << arpack.getNbrIterations() << " iterations.";
			LoggingSingleton::instance().message_info(ss.str());
			DenseMatrix embedding_feature_matrix = (arpack.eigenvectors()).rightCols(target_dimension);
			return EmbeddingResult(embedding_feature_matrix,arpack.eigenvalues().tail(target_dimension));
		}
		else
		{
			throw eigendecomposition_error("eigendecomposition failed");
		}
		return EmbeddingResult();
#else
		throw new unsupported_method_error("ARPACK is not available");
		return EmbeddingResult();
#endif
	}
};

//! Eigen library dense implementation of eigendecomposition
template <class LMatrixType, class RMatrixType, class MatrixOperationType> 
struct generalized_eigen_embedding_impl<LMatrixType, RMatrixType, MatrixOperationType, EIGEN_DENSE_SELFADJOINT_SOLVER>
{
	EmbeddingResult embed(const LMatrixType& lhs, const RMatrixType& rhs, IndexType target_dimension, unsigned int skip)
	{
		timed_context context("Eigen dense generalized eigendecomposition");

		DenseMatrix dense_lhs = lhs;
		DenseMatrix dense_rhs = rhs;
		Eigen::GeneralizedSelfAdjointEigenSolver<DenseMatrix> solver(dense_lhs, dense_rhs);
		if (solver.info() == Eigen::Success)
		{
			if (MatrixOperationType::largest)
			{
				assert(skip==0);
				DenseMatrix embedding_feature_matrix = solver.eigenvectors().rightCols(target_dimension);
				return EmbeddingResult(embedding_feature_matrix,solver.eigenvalues().tail(target_dimension));
			} 
			else
			{
				DenseMatrix embedding_feature_matrix = solver.eigenvectors().leftCols(target_dimension+skip).rightCols(target_dimension);
				return EmbeddingResult(embedding_feature_matrix,solver.eigenvalues().segment(skip,skip+target_dimension));
			}
		}
		else
		{
			throw eigendecomposition_error("eigendecomposition failed");
		}

		return EmbeddingResult();
	}
};

//! Adapter method for various generalized eigendecomposition methods. Currently
//! supports two methods:
//! <ul>
//! <li> ARPACK_XSXUPD
//! <li> EIGEN_DENSE_SELFADJOINT_SOLVER
//! </ul>
template <class LMatrixType, class RMatrixType, class MatrixOperationType>
EmbeddingResult generalized_eigen_embedding(TAPKEE_EIGEN_EMBEDDING_METHOD method, const LMatrixType& lhs,
                                            const RMatrixType& rhs,
                                            IndexType target_dimension, unsigned int skip)
{
	switch (method)
	{
#ifndef TAPKEE_NO_ARPACK
		case ARPACK: 
			return generalized_eigen_embedding_impl<LMatrixType, RMatrixType, MatrixOperationType, 
				ARPACK>().embed(lhs, rhs, target_dimension, skip);
#endif
		case EIGEN_DENSE_SELFADJOINT_SOLVER:
			return generalized_eigen_embedding_impl<LMatrixType, RMatrixType, MatrixOperationType,
				EIGEN_DENSE_SELFADJOINT_SOLVER>().embed(lhs, rhs, target_dimension, skip);
		case RANDOMIZED:
			throw unsupported_method_error("Randomized method is not supported for generalized eigenproblems");
			return EmbeddingResult();
		default: break;
	}
	return EmbeddingResult();
};

}
}

#endif
