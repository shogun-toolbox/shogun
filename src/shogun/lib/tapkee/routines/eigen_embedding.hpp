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

#ifndef TAPKEE_EIGEN_EMBEDDING_H_
#define TAPKEE_EIGEN_EMBEDDING_H_

/* Tapkee includes */
#ifdef TAPKEE_WITH_ARPACK
	#include <shogun/lib/tapkee/utils/arpack_wrapper.hpp>
#endif
#include <shogun/lib/tapkee/routines/matrix_operations.hpp>
#include <shogun/lib/tapkee/tapkee_defines.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

std::string get_eigen_embedding_name(TAPKEE_EIGEN_EMBEDDING_METHOD m)
{
	switch (m)
	{
#ifdef TAPKEE_WITH_ARPACK
		case ARPACK: return "ARPACK library";
#endif
		case RANDOMIZED: return "randomized (redsvd)";
		case EIGEN_DENSE_SELFADJOINT_SOLVER: return "dense (Eigen3)";
		default: return "Unknown eigendecomposition method (yes it is a bug)";
	}
}

//! Templated implementation of eigendecomposition-based embedding. 
template <class MatrixType, class MatrixOperationType, int IMPLEMENTATION> 
struct eigen_embedding_impl
{
	//! Construct embedding
	//! @param wm weight matrix to eigendecompose
	//! @param target_dimension target dimension of embedding (number of eigenvectors to find)
	//! @param skip number of eigenvectors to skip
	//!
	EmbeddingResult embed(const MatrixType& wm, IndexType target_dimension, unsigned int skip);
};

#ifdef TAPKEE_WITH_ARPACK
//! ARPACK implementation of eigendecomposition-based embedding
template <class MatrixType, class MatrixOperationType> 
struct eigen_embedding_impl<MatrixType, MatrixOperationType, ARPACK>
{
	EmbeddingResult embed(const MatrixType& wm, IndexType target_dimension, unsigned int skip)
	{
		timed_context context("ARPACK eigendecomposition");

		ArpackGeneralizedSelfAdjointEigenSolver<MatrixType, MatrixType, MatrixOperationType> 
			arpack(wm,target_dimension+skip,MatrixOperationType::ARPACK_CODE);

		if (arpack.info() == Eigen::Success)
		{
			stringstream ss;
			ss << "Took " << arpack.getNbrIterations() << " iterations.";
			LoggingSingleton::instance().message_info(ss.str());
			DenseMatrix embedding_feature_matrix = arpack.eigenvectors().rightCols(target_dimension);
			return EmbeddingResult(embedding_feature_matrix,arpack.eigenvalues().tail(target_dimension));
		}
		else
		{
			throw eigendecomposition_error("eigendecomposition failed");
		}
		return EmbeddingResult();
	}
};
#endif

//! Eigen library dense implementation of eigendecomposition-based embedding
template <class MatrixType, class MatrixOperationType> 
struct eigen_embedding_impl<MatrixType, MatrixOperationType, EIGEN_DENSE_SELFADJOINT_SOLVER>
{
	EmbeddingResult embed(const MatrixType& wm, IndexType target_dimension, unsigned int skip)
	{
		timed_context context("Eigen library dense eigendecomposition");

		DenseMatrix dense_wm = wm;
		DenseSelfAdjointEigenSolver solver(dense_wm);

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

//! Randomized redsvd-like implementation of eigendecomposition-based embedding
template <class MatrixType, class MatrixOperationType> 
struct eigen_embedding_impl<MatrixType, MatrixOperationType, RANDOMIZED>
{
	EmbeddingResult embed(const MatrixType& wm, IndexType target_dimension, unsigned int skip)
	{
		timed_context context("Randomized eigendecomposition");
		
		DenseMatrix O(wm.rows(), target_dimension+skip);
		for (IndexType i=0; i<O.rows(); ++i)
		{
			IndexType j=0;
			for ( ; j+1 < O.cols(); j+= 2)
			{
				ScalarType v1 = (ScalarType)(rand()+1.f)/((float)RAND_MAX+2.f);
				ScalarType v2 = (ScalarType)(rand()+1.f)/((float)RAND_MAX+2.f);
				ScalarType len = sqrt(-2.f*log(v1));
				O(i,j) = len*cos(2.f*M_PI*v2);
				O(i,j+1) = len*sin(2.f*M_PI*v2);
			}
			for ( ; j < O.cols(); j++)
			{
				ScalarType v1 = (ScalarType)(rand()+1.f)/((float)RAND_MAX+2.f);
				ScalarType v2 = (ScalarType)(rand()+1.f)/((float)RAND_MAX+2.f);
				ScalarType len = sqrt(-2.f*log(v1));
				O(i,j) = len*cos(2.f*M_PI*v2);
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
				DenseMatrix embedding_feature_matrix = (Y*eigenOfB.eigenvectors()).rightCols(target_dimension);
				return EmbeddingResult(embedding_feature_matrix,eigenOfB.eigenvalues());
			} 
			else
			{
				DenseMatrix embedding_feature_matrix = (Y*eigenOfB.eigenvectors()).leftCols(target_dimension+skip).rightCols(target_dimension);
				return EmbeddingResult(embedding_feature_matrix,eigenOfB.eigenvalues());
			}
		}
		else
		{
			throw eigendecomposition_error("eigendecomposition failed");
		}
		return EmbeddingResult();
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
//! <li> ARPACK
//! <li> RANDOMIZED
//! <li> EIGEN_DENSE_SELFADJOINT_SOLVER
//! </ul>
//!
//! @param method one of supported eigendecomposition methods
//! @param m matrix to be eigendecomposed 
//! @param target_dimension target dimension of embedding i.e. number of eigenvectors to be
//!        computed
//! @param skip number of eigenvectors to skip (from either smallest or largest side)
//!
template <class MatrixType, class MatrixOperationType>
EmbeddingResult eigen_embedding(TAPKEE_EIGEN_EMBEDDING_METHOD method, const MatrixType& m, 
                                IndexType target_dimension, unsigned int skip)
{
	LoggingSingleton::instance().message_info("Using " + get_eigen_embedding_name(method) +
			" eigendecomposition.");
	switch (method)
	{
#ifdef TAPKEE_WITH_ARPACK
		case ARPACK: 
			return eigen_embedding_impl<MatrixType, MatrixOperationType, 
				ARPACK>().embed(m, target_dimension, skip);
#endif
		case RANDOMIZED: 
			return eigen_embedding_impl<MatrixType, MatrixOperationType,
				RANDOMIZED>().embed(m, target_dimension, skip);
		case EIGEN_DENSE_SELFADJOINT_SOLVER:
			return eigen_embedding_impl<MatrixType, MatrixOperationType, 
				EIGEN_DENSE_SELFADJOINT_SOLVER>().embed(m, target_dimension, skip);
		default: break;
	}
	return EmbeddingResult();
};

}
}

#endif
