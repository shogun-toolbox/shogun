/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 *
 * Randomized eigendecomposition code is inspired by the redsvd library
 * code which is also distributed under BSD 3-clause license.
 *
 * Copyright (c) 2010-2013 Daisuke Okanohara
 *
 */

#ifndef TAPKEE_EIGENDECOMPOSITION_H_
#define TAPKEE_EIGENDECOMPOSITION_H_

/* Tapkee includes */
#ifdef TAPKEE_WITH_ARPACK
	#include <lib/tapkee/utils/arpack_wrapper.hpp>
#endif
#include <lib/tapkee/routines/matrix_operations.hpp>
#include <lib/tapkee/defines.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

#ifdef TAPKEE_WITH_ARPACK
//! ARPACK implementation of eigendecomposition-based embedding
template <class MatrixType, class MatrixOperationType>
EigendecompositionResult eigendecomposition_impl_arpack(const MatrixType& wm, IndexType target_dimension, unsigned int skip)
{
	timed_context context("ARPACK eigendecomposition");

	ArpackGeneralizedSelfAdjointEigenSolver<MatrixType, MatrixType, MatrixOperationType>
		arpack(wm,target_dimension+skip,MatrixOperationType::ARPACK_CODE);

	if (arpack.info() == Eigen::Success)
	{
		std::stringstream ss;
		ss << "Took " << arpack.getNbrIterations() << " iterations.";
		LoggingSingleton::instance().message_info(ss.str());
		DenseMatrix selected_eigenvectors = arpack.eigenvectors().rightCols(target_dimension);
		return EigendecompositionResult(selected_eigenvectors,arpack.eigenvalues().tail(target_dimension));
	}
	else
	{
		throw eigendecomposition_error("eigendecomposition failed");
	}
	return EigendecompositionResult();
}
#endif

//! Eigen library dense implementation of eigendecomposition-based embedding
template <class MatrixType, class MatrixOperationType>
EigendecompositionResult eigendecomposition_impl_dense(const MatrixType& wm, IndexType target_dimension, unsigned int skip)
{
	timed_context context("Eigen library dense eigendecomposition");

	DenseMatrix dense_wm = wm;
	DenseSelfAdjointEigenSolver solver(dense_wm);

	if (solver.info() == Eigen::Success)
	{
		if (MatrixOperationType::largest)
		{
			assert(skip==0);
			DenseMatrix selected_eigenvectors = solver.eigenvectors().rightCols(target_dimension);
			return EigendecompositionResult(selected_eigenvectors,solver.eigenvalues().tail(target_dimension));
		}
		else
		{
			DenseMatrix selected_eigenvectors = solver.eigenvectors().leftCols(target_dimension+skip).rightCols(target_dimension);
			return EigendecompositionResult(selected_eigenvectors,solver.eigenvalues().segment(skip,skip+target_dimension));
		}
	}
	else
	{
		throw eigendecomposition_error("eigendecomposition failed");
	}
	return EigendecompositionResult();
}

//! Randomized redsvd-like implementation of eigendecomposition-based embedding
template <class MatrixType, class MatrixOperationType>
EigendecompositionResult eigendecomposition_impl_randomized(const MatrixType& wm, IndexType target_dimension, unsigned int skip)
{
	timed_context context("Randomized eigendecomposition");

	DenseMatrix O(wm.rows(), target_dimension+skip);
	for (IndexType i=0; i<O.rows(); ++i)
	{
		for (IndexType j=0; j<O.cols(); j++)
		{
			O(i,j) = tapkee::gaussian_random();
		}
	}
	MatrixOperationType operation(wm);

	DenseMatrix Y = operation(O);
	for (IndexType i=0; i<Y.cols(); i++)
	{
		for (IndexType j=0; j<i; j++)
		{
			ScalarType r = Y.col(i).dot(Y.col(j));
			Y.col(i) -= r*Y.col(j);
		}
		ScalarType norm = Y.col(i).norm();
		if (norm < 1e-4)
		{
			for (int k = i; k<Y.cols(); k++)
				Y.col(k).setZero();
		}
		Y.col(i) *= (1.f / norm);
	}

	DenseMatrix B1 = operation(Y);
	DenseMatrix B = Y.householderQr().solve(B1);
	DenseSelfAdjointEigenSolver eigenOfB(B);

	if (eigenOfB.info() == Eigen::Success)
	{
		if (MatrixOperationType::largest)
		{
			assert(skip==0);
			DenseMatrix selected_eigenvectors = (Y*eigenOfB.eigenvectors()).rightCols(target_dimension);
			return EigendecompositionResult(selected_eigenvectors,eigenOfB.eigenvalues());
		}
		else
		{
			DenseMatrix selected_eigenvectors = (Y*eigenOfB.eigenvectors()).leftCols(target_dimension+skip).rightCols(target_dimension);
			return EigendecompositionResult(selected_eigenvectors,eigenOfB.eigenvalues());
		}
	}
	else
	{
		throw eigendecomposition_error("eigendecomposition failed");
	}
	return EigendecompositionResult();
}

//! Multiple implementation handler method for various eigendecomposition methods.
//!
//! Has three template parameters:
//! MatrixType - class of weight matrix to perform eigendecomposition of
//! MatrixOperationType - class of product operation over matrix.
//!
//! In order to compute largest eigenvalues MatrixOperationType should provide
//! implementation of operator()(DenseMatrix) which computes right product
//! of the parameter with the MatrixType.
//!
//! In order to compute smallest eigenvalues MatrixOperationType should provide
//! implementation of operator()(DenseMatrix) which solves linear system with
//! given right-hand side part.
//!
//! Currently supports three methods:
//!
//! <ul>
//! <li> Arpack
//! <li> Randomized
//! <li> Dense
//! </ul>
//!
//! @param method one of supported eigendecomposition methods
//! @param m matrix to be eigendecomposed
//! @param target_dimension target dimension of embedding i.e. number of eigenvectors to be
//!        computed
//! @param skip number of eigenvectors to skip (from either smallest or largest side)
//!
template <class MatrixType, class MatrixOperationType>
EigendecompositionResult eigendecomposition(EigenMethod method, const MatrixType& m,
                                            IndexType target_dimension, unsigned int skip)
{
	LoggingSingleton::instance().message_info("Using the " + get_eigen_method_name(method) + " eigendecomposition method.");
	switch (method)
	{
#ifdef TAPKEE_WITH_ARPACK
		case Arpack:
			return eigendecomposition_impl_arpack<MatrixType, MatrixOperationType>(m, target_dimension, skip);
#endif
		case Randomized:
			return eigendecomposition_impl_randomized<MatrixType, MatrixOperationType>(m, target_dimension, skip);
		case Dense:
			return eigendecomposition_impl_dense<MatrixType, MatrixOperationType>(m, target_dimension, skip);
		default: break;
	}
	return EigendecompositionResult();
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
