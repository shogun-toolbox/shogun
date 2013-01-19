/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
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
template <class LMatrixType, class RMatrixType, class MatrixTypeOperation, int IMPLEMENTATION> 
struct generalized_eigen_embedding_impl
{
	/** Construct embedding
	 * @param wm weight matrix to eigendecompose
	 * @param target_dimension target dimension of embedding (number of eigenvectors to find)
	 * @param skip number of eigenvectors to skip
	 */
	virtual EmbeddingResult embed(const LMatrixType& lhs, const RMatrixType& rhs, unsigned int target_dimension, unsigned int skip);
};

//! ARPACK implementation of eigendecomposition-based embedding
template <class LMatrixType, class RMatrixType, class MatrixTypeOperation> 
struct generalized_eigen_embedding_impl<LMatrixType, RMatrixType, MatrixTypeOperation, ARPACK>
{
	EmbeddingResult embed(const LMatrixType& lhs, const RMatrixType& rhs, unsigned int target_dimension, unsigned int skip)
	{
		timed_context context("ARPACK DSXUPD generalized eigendecomposition");

#ifndef TAPKEE_NO_ARPACK
		ArpackGeneralizedSelfAdjointEigenSolver<LMatrixType, RMatrixType, MatrixTypeOperation> arpack(lhs,rhs,target_dimension+skip,"SM");

		DenseMatrix embedding_feature_matrix = (arpack.eigenvectors()).block(0,skip,lhs.cols(),target_dimension);

		return EmbeddingResult(embedding_feature_matrix,arpack.eigenvalues().tail(target_dimension));
#else
		return EmbeddingResult();
#endif
	}
};

//! Eigen library dense implementation of eigendecomposition
template <class LMatrixType, class RMatrixType, class MatrixTypeOperation> 
struct generalized_eigen_embedding_impl<LMatrixType, RMatrixType, MatrixTypeOperation, EIGEN_DENSE_SELFADJOINT_SOLVER>
{
	EmbeddingResult embed(const LMatrixType& lhs, const RMatrixType& rhs, unsigned int target_dimension, unsigned int skip)
	{
		timed_context context("Eigen dense generalized eigendecomposition");

		DenseMatrix dense_lhs = lhs;
		DenseMatrix dense_rhs = rhs;
		Eigen::GeneralizedSelfAdjointEigenSolver<DenseMatrix> solver(dense_lhs, dense_rhs);

		DenseMatrix embedding_feature_matrix = (solver.eigenvectors()).block(0,skip,lhs.cols(),target_dimension);

		return EmbeddingResult(embedding_feature_matrix,solver.eigenvalues().tail(target_dimension));
	}
};

//! Adapter method for various generalized eigendecomposition methods. Currently
//! supports two methods:
//! <ul>
//! <li> ARPACK_XSXUPD
//! <li> EIGEN_DENSE_SELFADJOINT_SOLVER
//! </ul>
template <class LMatrixType, class RMatrixType, class MatrixTypeOperation>
EmbeddingResult generalized_eigen_embedding(TAPKEE_EIGEN_EMBEDDING_METHOD method, const LMatrixType& lhs,
                                            const RMatrixType& rhs,
                                            unsigned int target_dimension, unsigned int skip)
{
	switch (method)
	{
#ifndef TAPKEE_NO_ARPACK
		case ARPACK: 
			return generalized_eigen_embedding_impl<LMatrixType, RMatrixType, MatrixTypeOperation, 
				ARPACK>().embed(lhs, rhs, target_dimension, skip);
#endif
		case EIGEN_DENSE_SELFADJOINT_SOLVER:
			return generalized_eigen_embedding_impl<LMatrixType, RMatrixType, MatrixTypeOperation,
				EIGEN_DENSE_SELFADJOINT_SOLVER>().embed(lhs, rhs, target_dimension, skip);
		case RANDOMIZED:
			// TODO fail here
			return EmbeddingResult();
		default: break;
	}
	return EmbeddingResult();
};

}
}

#endif
