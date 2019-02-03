/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_GENERALIZED_EIGENDECOMPOSITION_H_
#define TAPKEE_GENERALIZED_EIGENDECOMPOSITION_H_

/* Tapkee includes */
#ifdef TAPKEE_WITH_ARPACK
	#include <shogun/lib/tapkee/utils/arpack_wrapper.hpp>
#endif
#include <shogun/lib/tapkee/routines/matrix_operations.hpp>
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
		std::string message = formatting::format("Took {} iterations.", arpack.getNbrIterations());
		LoggingSingleton::instance().message_info(message);
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

template <typename LMatrixType, typename RMatrixType>
struct generalized_eigendecomposition_impl
{
#ifdef TAPKEE_WITH_ARPACK
	EigendecompositionResult arpack(const LMatrixType& lhs, const RMatrixType& rhs,
                                    const ComputationStrategy& strategy,
                                    const EigendecompositionStrategy& eigen_strategy,
                                    IndexType target_dimension);
#endif
	EigendecompositionResult dense(const LMatrixType& lhs, const RMatrixType& rhs,
                                   const ComputationStrategy& strategy,
                                   const EigendecompositionStrategy& eigen_strategy,
                                   IndexType target_dimension);
};

template <>
struct generalized_eigendecomposition_impl<SparseWeightMatrix, DenseDiagonalMatrix>
{
#ifdef TAPKEE_WITH_ARPACK
	EigendecompositionResult arpack(const SparseWeightMatrix& lhs, const DenseDiagonalMatrix& rhs,
                                    const ComputationStrategy& strategy,
                                    const EigendecompositionStrategy& eigen_strategy,
                                    IndexType target_dimension)
	{
		if (strategy.is(HomogeneousCPUStrategy))
		{
			if (eigen_strategy.is(SmallestEigenvalues))
				return generalized_eigendecomposition_impl_arpack
					<SparseWeightMatrix,DenseDiagonalMatrix,SparseInverseMatrixOperation>
					(lhs,rhs,target_dimension,eigen_strategy.skip());
			unsupported();
		}
		unsupported();
		return EigendecompositionResult();
	}
#endif
	EigendecompositionResult dense(const SparseWeightMatrix& lhs, const DenseDiagonalMatrix& rhs,
                                   const ComputationStrategy& strategy,
                                   const EigendecompositionStrategy& eigen_strategy,
                                   IndexType target_dimension)
	{
		if (strategy.is(HomogeneousCPUStrategy))
		{
			if (eigen_strategy.is(SmallestEigenvalues))
				return generalized_eigendecomposition_impl_dense
					<SparseWeightMatrix,DenseDiagonalMatrix,SparseInverseMatrixOperation>
					(lhs,rhs,target_dimension,eigen_strategy.skip());
			unsupported();
		}
		unsupported();
		return EigendecompositionResult();
	}
	inline void unsupported() const
	{
		throw unsupported_method_error("Unsupported method");
	}
};

template <>
struct generalized_eigendecomposition_impl<DenseMatrix, DenseMatrix>
{
#ifdef TAPKEE_WITH_ARPACK
	EigendecompositionResult arpack(const DenseMatrix& lhs, const DenseMatrix& rhs,
                                    const ComputationStrategy& strategy,
                                    const EigendecompositionStrategy& eigen_strategy,
                                    IndexType target_dimension)
	{
		if (strategy.is(HomogeneousCPUStrategy))
		{
			if (eigen_strategy.is(SmallestEigenvalues))
				return generalized_eigendecomposition_impl_arpack
					<DenseMatrix,DenseMatrix,DenseInverseMatrixOperation>
					(lhs,rhs,target_dimension,0);
			unsupported();
		}
		unsupported();
		return EigendecompositionResult();
	}
#endif
	EigendecompositionResult dense(const DenseMatrix& lhs, const DenseMatrix& rhs,
                                   const ComputationStrategy& strategy,
                                   const EigendecompositionStrategy& eigen_strategy,
                                   IndexType target_dimension)
	{
		if (strategy.is(HomogeneousCPUStrategy))
		{
			if (eigen_strategy.is(SmallestEigenvalues))
				return generalized_eigendecomposition_impl_dense
					<DenseMatrix,DenseMatrix,DenseInverseMatrixOperation>
					(lhs,rhs,target_dimension,0);
			unsupported();
		}
		unsupported();
		return EigendecompositionResult();
	}
	inline void unsupported() const
	{
		throw unsupported_method_error("Unsupported method");
	}
};

template <class LMatrixType, class RMatrixType>
EigendecompositionResult generalized_eigendecomposition(const EigenMethod& method, const ComputationStrategy& strategy,
                                                        const EigendecompositionStrategy& eigen_strategy,
                                                        const LMatrixType& lhs, const RMatrixType& rhs, IndexType target_dimension)
{
	LoggingSingleton::instance().message_info(formatting::format("Using the {} eigendecomposition method.",
		get_eigen_method_name(method)));
#ifdef TAPKEE_WITH_ARPACK
	if (method.is(Arpack))
		return generalized_eigendecomposition_impl<LMatrixType, RMatrixType>()
			.arpack(lhs, rhs, strategy, eigen_strategy, target_dimension);
#endif
	if (method.is(Dense))
		return generalized_eigendecomposition_impl<LMatrixType, RMatrixType>()
			.dense(lhs, rhs, strategy, eigen_strategy, target_dimension);
	if (method.is(Randomized))
		throw unsupported_method_error("Randomized method is not supported for generalized eigenproblems");
	return EigendecompositionResult();
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
