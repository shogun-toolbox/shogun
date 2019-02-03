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
	#include <shogun/lib/tapkee/utils/arpack_wrapper.hpp>
#endif
#include <shogun/lib/tapkee/routines/matrix_operations.hpp>
#include <shogun/lib/tapkee/defines.hpp>
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
		arpack(wm,target_dimension+skip,MatrixOperationType::ARPACK_CODE());

	if (arpack.info() == Eigen::Success)
	{
		std::string message = formatting::format("Took {} iterations.", arpack.getNbrIterations());
		LoggingSingleton::instance().message_info(message);
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

	DenseSymmetricMatrix dense_wm = wm;
	dense_wm += dense_wm.transpose().eval();
	dense_wm /= 2.0;
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

template <typename MatrixType>
struct eigendecomposition_impl
{
#ifdef TAPKEE_WITH_ARPACK
	EigendecompositionResult arpack(const MatrixType& m, const ComputationStrategy& strategy,
                                    const EigendecompositionStrategy& eigen_strategy,
                                    IndexType target_dimension);
#endif
	EigendecompositionResult dense(const MatrixType& m, const ComputationStrategy& strategy,
                                   const EigendecompositionStrategy& eigen_strategy,
                                   IndexType target_dimension);
	EigendecompositionResult randomized(const MatrixType& m, const ComputationStrategy& strategy,
                                        const EigendecompositionStrategy& eigen_strategy,
                                        IndexType target_dimension);
};

template <>
struct eigendecomposition_impl<DenseMatrix>
{
#ifdef TAPKEE_WITH_ARPACK
	EigendecompositionResult arpack(const DenseMatrix& m, const ComputationStrategy& strategy,
                                    const EigendecompositionStrategy& eigen_strategy,
                                    IndexType target_dimension)
	{
		if (strategy.is(HomogeneousCPUStrategy))
		{
			if (eigen_strategy.is(LargestEigenvalues))
				return eigendecomposition_impl_arpack<DenseMatrix,DenseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			if (eigen_strategy.is(SquaredLargestEigenvalues))
				return eigendecomposition_impl_arpack<DenseMatrix,DenseImplicitSquareMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			if (eigen_strategy.is(SmallestEigenvalues))
				return eigendecomposition_impl_arpack<DenseMatrix,DenseInverseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			unsupported();
		}
#ifdef TAPKEE_WITH_VIENNACL
		if (strategy.is(HeterogeneousOpenCLStrategy))
		{
			if (eigen_strategy.is(LargestEigenvalues))
				return eigendecomposition_impl_arpack<DenseMatrix,GPUDenseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			if (eigen_strategy.is(SquaredLargestEigenvalues))
				return eigendecomposition_impl_arpack<DenseMatrix,GPUDenseImplicitSquareMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			unsupported();
		}
#endif
		unsupported();
		return EigendecompositionResult();
	}
#endif
	EigendecompositionResult dense(const DenseMatrix& m, const ComputationStrategy& strategy,
                                   const EigendecompositionStrategy& eigen_strategy,
                                   IndexType target_dimension)
	{
		if(strategy.is(HomogeneousCPUStrategy))
		{
			if (eigen_strategy.is(LargestEigenvalues))
				return eigendecomposition_impl_dense<DenseMatrix,DenseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			if (eigen_strategy.is(SquaredLargestEigenvalues))
				return eigendecomposition_impl_dense<DenseMatrix,DenseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			if (eigen_strategy.is(SmallestEigenvalues))
				return eigendecomposition_impl_dense<DenseMatrix,DenseInverseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			unsupported();
		}
		unsupported();
		return EigendecompositionResult();
	}
	EigendecompositionResult randomized(const DenseMatrix& m, const ComputationStrategy& strategy,
                                        const EigendecompositionStrategy& eigen_strategy,
                                        IndexType target_dimension)
	{
		if (strategy.is(HomogeneousCPUStrategy))
		{
			if (eigen_strategy.is(LargestEigenvalues))
				return eigendecomposition_impl_randomized<DenseMatrix,DenseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			if (eigen_strategy.is(SquaredLargestEigenvalues))
				return eigendecomposition_impl_randomized<DenseMatrix,DenseImplicitSquareMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			if (eigen_strategy.is(SmallestEigenvalues))
				return eigendecomposition_impl_randomized<DenseMatrix,DenseInverseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			unsupported();
		}
#ifdef TAPKEE_WITH_VIENNACL
		if (strategy.is(HeterogeneousOpenCLStrategy))
		{
			if (eigen_strategy.is(LargestEigenvalues))
				return eigendecomposition_impl_randomized<DenseMatrix,GPUDenseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			if (eigen_strategy.is(SquaredLargestEigenvalues))
				return eigendecomposition_impl_randomized<DenseMatrix,GPUDenseImplicitSquareMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			unsupported();
		}
#endif
		unsupported();
		return EigendecompositionResult();
	}
	inline void unsupported() const
	{
		throw unsupported_method_error("Unsupported method");
	}
};

template <>
struct eigendecomposition_impl<SparseWeightMatrix>
{
#ifdef TAPKEE_WITH_ARPACK
	EigendecompositionResult arpack(const SparseWeightMatrix& m, const ComputationStrategy& strategy,
                                    const EigendecompositionStrategy& eigen_strategy,
                                    IndexType target_dimension)
	{
		if (strategy.is(HomogeneousCPUStrategy))
		{
			if (eigen_strategy.is(SmallestEigenvalues))
				return eigendecomposition_impl_arpack<SparseWeightMatrix,SparseInverseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			unsupported();
		}
		unsupported();
		return EigendecompositionResult();
	}
#endif
	EigendecompositionResult dense(const SparseWeightMatrix& m, const ComputationStrategy& strategy,
                                   const EigendecompositionStrategy& eigen_strategy,
                                   IndexType target_dimension)
	{
		if (strategy.is(HomogeneousCPUStrategy))
		{
			if (eigen_strategy.is(SmallestEigenvalues))
				return eigendecomposition_impl_dense<SparseWeightMatrix,SparseInverseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
			unsupported();
		}
		unsupported();
		return EigendecompositionResult();
	}
	EigendecompositionResult randomized(const SparseWeightMatrix& m, const ComputationStrategy& strategy,
                                        const EigendecompositionStrategy& eigen_strategy,
                                        IndexType target_dimension)
	{
		if (strategy.is(HomogeneousCPUStrategy))
		{
			if (eigen_strategy.is(SmallestEigenvalues))
				return eigendecomposition_impl_randomized<SparseWeightMatrix,SparseInverseMatrixOperation>
					(m,target_dimension,eigen_strategy.skip());
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
template <class MatrixType>
EigendecompositionResult eigendecomposition(const EigenMethod& method, const ComputationStrategy& strategy,
                                            const EigendecompositionStrategy& eigen_strategy,
                                            const MatrixType& m, IndexType target_dimension)
{
	LoggingSingleton::instance().message_info(formatting::format("Using the {} eigendecomposition method.",
		get_eigen_method_name(method)));
#ifdef TAPKEE_WITH_ARPACK
	if (method.is(Arpack))
		return eigendecomposition_impl<MatrixType>().arpack(m,strategy,eigen_strategy,target_dimension);
#endif
	if (method.is(Randomized))
		return eigendecomposition_impl<MatrixType>().randomized(m,strategy,eigen_strategy,target_dimension);
	if (method.is(Dense))
		return eigendecomposition_impl<MatrixType>().dense(m,strategy,eigen_strategy,target_dimension);
	return EigendecompositionResult();
}


} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
