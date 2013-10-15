/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DEFINES_TYPES_H_
#define TAPKEE_DEFINES_TYPES_H_

namespace tapkee
{
#ifdef TAPKEE_CUSTOM_INTERNAL_NUMTYPE
	typedef TAPKEE_CUSTOM_INTERNAL_NUMTYPE ScalarType;
#else
	//! default scalar value (can be overrided with TAPKEE_CUSTOM_INTERNAL_NUMTYPE define)
	typedef double ScalarType;
#endif
	//! indexing type (non-overridable)
	//! set to int for compatibility with OpenMP 2.0
	typedef int IndexType;
	//! dense vector type (non-overridable)
	typedef Eigen::Matrix<tapkee::ScalarType,Eigen::Dynamic,1> DenseVector;
	//! dense matrix type (non-overridable)
	typedef Eigen::Matrix<tapkee::ScalarType,Eigen::Dynamic,Eigen::Dynamic> DenseMatrix;
	//! dense symmetric matrix (non-overridable, currently just dense matrix, can be improved later)
	typedef tapkee::DenseMatrix DenseSymmetricMatrix;
	//! dense diagonal matrix
	typedef Eigen::DiagonalMatrix<tapkee::ScalarType,Eigen::Dynamic> DenseDiagonalMatrix;
	//! sparse weight matrix type (non-overridable)
	typedef Eigen::SparseMatrix<tapkee::ScalarType> SparseWeightMatrix;
	//! sparse matrix type (non-overridable)
	typedef Eigen::SparseMatrix<tapkee::ScalarType> SparseMatrix;
	//! selfadjoint solver (non-overridable)
	typedef Eigen::SelfAdjointEigenSolver<tapkee::DenseMatrix> DenseSelfAdjointEigenSolver;
	//! dense solver (non-overridable)
	typedef Eigen::LDLT<tapkee::DenseMatrix> DenseSolver;
#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	//! sparse solver (it is Eigen::SimplicialCholesky in case of eigen version <3.1.0,
	//! in case of TAPKEE_USE_SUPERLU being defined it is Eigen::SuperLU, by default
	//! it is Eigen::SimplicialLDLT)
	typedef Eigen::SimplicialCholesky<tapkee::SparseWeightMatrix> SparseSolver;
#else
	#if defined(TAPKEE_SUPERLU_AVAILABLE) && defined(TAPKEE_USE_SUPERLU)
	typedef Eigen::SuperLU<tapkee::SparseWeightMatrix> SparseSolver;
	#else
	typedef Eigen::SimplicialLDLT<tapkee::SparseWeightMatrix> SparseSolver;
	#endif
#endif
}

#endif
