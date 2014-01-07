/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_GENERALIZED_EIGENDECOMPOSITION_H_
#define TAPKEE_GENERALIZED_EIGENDECOMPOSITION_H_

/* Tapkee includes */
#ifdef TAPKEE_WITH_ARPACK
	#include <lib/tapkee/utils/arpack_wrapper.hpp>
#endif
#include <lib/tapkee/routines/matrix_operations.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

#ifdef TAPKEE_WITH_ARPACK
//! ARPACK implementation of eigendecomposition-based embedding
template <class LMatrixType, class RMatrixType, class MatrixOperationType>
EigendecompositionResult generalized_eigendecomposition_impl_arpack(const LMatrixType& lhs,
		const RMatrixType& rhs, IndexType target_dimension, unsigned int skip)
{
	timed_context context("ARPACK DSXUPD generalized eigendecomposition");

	ArpackGeneralizedSelfAdjointEigenSolver<LMatrixType, RMatrixType, MatrixOperationType>
		arpack(lhs,rhs,target_dimension+skip,"SM");

	if (arpack.info() == Eigen::Success)
	{
		std::stringstream ss;
		ss << "Took " << arpack.getNbrIterations() << " iterations.";
		LoggingSingleton::instance().message_info(ss.str());
		DenseMatrix selected_eigenvectors = (arpack.eigenvectors()).rightCols(target_dimension);
		return EigendecompositionResult(selected_eigenvectors,arpack.eigenvalues().tail(target_dimension));
	}
	else
	{
		throw eigendecomposition_error("eigendecomposition failed");
	}
	return EigendecompositionResult();
}
#endif

//! Eigen library dense implementation of eigendecomposition
template <class LMatrixType, class RMatrixType, class MatrixOperationType>
EigendecompositionResult generalized_eigendecomposition_impl_dense(const LMatrixType& lhs,
		const RMatrixType& rhs, IndexType target_dimension, unsigned int skip)
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

template <class LMatrixType, class RMatrixType, class MatrixOperationType>
EigendecompositionResult generalized_eigendecomposition(EigenMethod method, const LMatrixType& lhs,
                                                        const RMatrixType& rhs,
                                                        IndexType target_dimension, unsigned int skip)
{
	LoggingSingleton::instance().message_info("Using the " + get_eigen_method_name(method) + " eigendecomposition method.");
	switch (method)
	{
#ifdef TAPKEE_WITH_ARPACK
		case Arpack:
			return generalized_eigendecomposition_impl_arpack<LMatrixType, RMatrixType, MatrixOperationType>(lhs, rhs, target_dimension, skip);
#endif
		case Dense:
			return generalized_eigendecomposition_impl_dense<LMatrixType, RMatrixType, MatrixOperationType>(lhs, rhs, target_dimension, skip);
		case Randomized:
			throw unsupported_method_error("Randomized method is not supported for generalized eigenproblems");
			return EigendecompositionResult();
		default: break;
	}
	return EigendecompositionResult();
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
