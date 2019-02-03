/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 *
 */

#ifndef TAPKEE_MATRIX_OPS_H_
#define TAPKEE_MATRIX_OPS_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/defines.hpp>
/* End of Tapkee includes */

#ifdef TAPKEE_WITH_VIENNACL
	#define VIENNACL_HAVE_EIGEN
	#include <viennacl/matrix.hpp>
	#include <viennacl/vector.hpp>
	#include <viennacl/linalg/prod.hpp>
#endif

namespace tapkee
{
namespace tapkee_internal
{

//! Matrix-matrix operation used to
//! compute smallest eigenvalues and
//! associated eigenvectors of a sparse matrix
//! Essentially solves linear system
//! with provided right-hand side part.
//!
struct SparseInverseMatrixOperation
{
	SparseInverseMatrixOperation(const SparseWeightMatrix& matrix) : solver()
	{
		solver.compute(matrix);
	}
	/** Solves linear system with provided right-hand size
	 */
	DenseMatrix operator()(const DenseMatrix& operatee)
	{
		return solver.solve(operatee);
	}
	SparseSolver solver;
	static const std::string& ARPACK_CODE()
	{
		static std::string foo("SM");
		return foo;
	}
	static const bool largest = false;
};

//! Matrix-matrix operation used to
//! compute smallest eigenvalues and
//! associated eigenvectors of a dense matrix
//! Essentially solves linear system
//! with provided right-hand side part.
//!
struct DenseInverseMatrixOperation
{
	DenseInverseMatrixOperation(const DenseMatrix& matrix) : solver()
	{
		solver.compute(matrix);
	}
	/** Solves linear system with provided right-hand size
	 */
	DenseMatrix operator()(const DenseMatrix& operatee)
	{
		return solver.solve(operatee);
	}
	DenseSolver solver;
	static const std::string& ARPACK_CODE()
	{
		static std::string foo("SM");
		return foo;
	}
	static const bool largest = false;
};

//! Matrix-matrix operation used to
//! compute largest eigenvalues and
//! associated eigenvectors. Essentially
//! computes matrix product with
//! provided right-hand side part.
//!
struct DenseMatrixOperation
{
	DenseMatrixOperation(const DenseMatrix& matrix) : _matrix(matrix)
	{
	}
	//! Computes matrix product of the matrix and provided right-hand
	//! side matrix
	//!
	//! @param rhs right-hand size matrix
	//!
	DenseMatrix operator()(const DenseMatrix& rhs)
	{
		return _matrix.selfadjointView<Eigen::Upper>()*rhs;
	}
	const DenseMatrix& _matrix;
	static const std::string& ARPACK_CODE()
	{
		static std::string foo("LA");
		return foo;
	}
	static const bool largest = true;
};

//! Matrix-matrix operation used to
//! compute largest eigenvalues and
//! associated eigenvectors of X*X^T like
//! matrix implicitly. Essentially
//! computes matrix product with provided
//! right-hand side part *twice*.
//!
struct DenseImplicitSquareSymmetricMatrixOperation
{
	DenseImplicitSquareSymmetricMatrixOperation(const DenseMatrix& matrix) : _matrix(matrix)
	{
	}
	//! Computes matrix product of the matrix and provided right-hand
	//! side matrix twice
	//!
	//! @param rhs right-hand side matrix
	//!
	DenseMatrix operator()(const DenseMatrix& rhs)
	{
		return _matrix.selfadjointView<Eigen::Upper>()*(_matrix.selfadjointView<Eigen::Upper>()*rhs);
	}
	const DenseMatrix& _matrix;
	static const std::string& ARPACK_CODE()
	{
		static std::string foo("LA");
		return foo;
	}
	static const bool largest = true;
};

//! Matrix-matrix operation used to
//! compute largest eigenvalues and
//! associated eigenvectors of X*X^T like
//! matrix implicitly. Essentially
//! computes matrix product with provided
//! right-hand side part *twice*.
//!
struct DenseImplicitSquareMatrixOperation
{
	DenseImplicitSquareMatrixOperation(const DenseMatrix& matrix) : _matrix(matrix)
	{
	}
	//! Computes matrix product of the matrix and provided right-hand
	//! side matrix twice
	//!
	//! @param rhs right-hand side matrix
	//!
	DenseMatrix operator()(const DenseMatrix& rhs)
	{
		return _matrix*(_matrix.transpose()*rhs);
	}
	const DenseMatrix& _matrix;
	static const std::string& ARPACK_CODE()
	{
		static std::string foo("LA");
		return foo;
	}
	static constexpr bool largest = true;
};

#ifdef TAPKEE_WITH_VIENNACL
struct GPUDenseImplicitSquareMatrixOperation
{
	GPUDenseImplicitSquareMatrixOperation(const DenseMatrix& matrix)
	{
		mat = viennacl::matrix<ScalarType>(matrix.cols(),matrix.rows());
		vec = viennacl::vector<ScalarType>(matrix.cols());
		res = viennacl::vector<ScalarType>(matrix.cols());
		viennacl::copy(matrix,mat);
	}
	//! Computes matrix product of the matrix and provided right-hand
	//! side matrix twice
	//!
	//! @param rhs right-hand side matrix
	//!
	DenseVector operator()(const DenseVector& rhs)
	{
		viennacl::copy(rhs,vec);
		res = viennacl::linalg::prod(mat, vec);
		vec = res;
		res = viennacl::linalg::prod(mat, vec);
		DenseVector result(rhs);
		viennacl::copy(res,result);
		return result;
	}
	viennacl::matrix<ScalarType> mat;
	viennacl::vector<ScalarType> vec;
	viennacl::vector<ScalarType> res;
	static const std::string& ARPACK_CODE()
	{
		static std::string foo("LA");
		return foo;
	}
	static const bool largest = true;
};

struct GPUDenseMatrixOperation
{
	GPUDenseMatrixOperation(const DenseMatrix& matrix)
	{
		mat = viennacl::matrix<ScalarType>(matrix.cols(),matrix.rows());
		vec = viennacl::vector<ScalarType>(matrix.cols());
		res = viennacl::vector<ScalarType>(matrix.cols());
		viennacl::copy(matrix,mat);
	}
	//! Computes matrix product of the matrix and provided right-hand
	//! side matrix twice
	//!
	//! @param rhs right-hand side matrix
	//!
	DenseVector operator()(const DenseVector& rhs)
	{
		viennacl::copy(rhs,vec);
		res = viennacl::linalg::prod(mat, vec);
		DenseVector result(rhs);
		viennacl::copy(res,result);
		return result;
	}
	viennacl::matrix<ScalarType> mat;
	viennacl::vector<ScalarType> vec;
	viennacl::vector<ScalarType> res;
	static const std::string& ARPACK_CODE()
	{
		static std::string foo("LA");
		return foo;
	}
	static const bool largest = true;
};
#endif

}
}

#endif
