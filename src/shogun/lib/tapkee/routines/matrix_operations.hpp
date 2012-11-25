/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 *
 */

#ifndef TAPKEE_MATRIX_OPS_H_
#define TAPKEE_MATRIX_OPS_H_

//! Matrix-matrix operation used to 
//! compute smallest eigenvalues and 
//! associated eigenvectors. Essentially
//! solves linear system with provided
//! right-hand side part.
//!
struct InverseSparseMatrixOperation
{
	InverseSparseMatrixOperation(const SparseWeightMatrix& matrix) : solver()
	{
		solver.compute(matrix);
	}
	/** Solves linear system with provided right-hand size
	 */
	inline DenseMatrix operator()(const DenseMatrix& operatee)
	{
		return solver.solve(operatee);
	}
	DefaultSparseSolver solver;
	static const char* ARPACK_CODE;
};
const char* InverseSparseMatrixOperation::ARPACK_CODE = "SM";

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
	inline DenseMatrix operator()(const DenseMatrix& rhs)
	{
		return _matrix.selfadjointView<Eigen::Upper>()*rhs;
	}
	const DenseMatrix& _matrix;
	static const char* ARPACK_CODE;
};
const char* DenseMatrixOperation::ARPACK_CODE = "LM";

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
	inline DenseMatrix operator()(const DenseMatrix& rhs)
	{
		return _matrix.selfadjointView<Eigen::Upper>()*(_matrix.selfadjointView<Eigen::Upper>()*rhs);
	}
	const DenseMatrix& _matrix;
	static const char* ARPACK_CODE;
};
const char* DenseImplicitSquareMatrixOperation::ARPACK_CODE = "LM";

#ifdef TAPKEE_GPU
struct GPUDenseImplicitSquareMatrixOperation
{
	GPUDenseImplicitSquareMatrixOperation(const DenseMatrix& matrix)
	{
		timed_context c("Storing matrices");
		mat = viennacl::matrix<DefaultScalarType>(matrix.cols(),matrix.rows());
		vec = viennacl::matrix<DefaultScalarType>(matrix.cols(),1);
		res = viennacl::matrix<DefaultScalarType>(matrix.cols(),1);
		viennacl::copy(matrix,mat);
	}
	//! Computes matrix product of the matrix and provided right-hand 
	//! side matrix twice
	//! 
	//! @param rhs right-hand side matrix
	//!
	inline DenseMatrix operator()(const DenseMatrix& rhs)
	{
		timed_context c("Computing product");
		viennacl::copy(rhs,vec);
		res = viennacl::linalg::prod(mat, vec);
		vec = res;
		res = viennacl::linalg::prod(mat, vec);
		DenseMatrix result(rhs);
		viennacl::copy(res,result);
		return result;
	}
	viennacl::matrix<DefaultScalarType> mat;
	viennacl::matrix<DefaultScalarType> vec;
	viennacl::matrix<DefaultScalarType> res;
	static const char* ARPACK_CODE;
};
const char* GPUDenseImplicitSquareMatrixOperation::ARPACK_CODE = "LM";

struct GPUDenseMatrixOperation
{
	GPUDenseMatrixOperation(const DenseMatrix& matrix)
	{
		mat = viennacl::matrix<DefaultScalarType>(matrix.cols(),matrix.rows());
		vec = viennacl::matrix<DefaultScalarType>(matrix.cols(),1);
		res = viennacl::matrix<DefaultScalarType>(matrix.cols(),1);
		viennacl::copy(matrix,mat);
	}
	//! Computes matrix product of the matrix and provided right-hand 
	//! side matrix twice
	//! 
	//! @param rhs right-hand side matrix
	//!
	inline DenseMatrix operator()(const DenseMatrix& rhs)
	{
		viennacl::copy(rhs,vec);
		res = viennacl::linalg::prod(mat, vec);
		DenseMatrix result(rhs);
		viennacl::copy(res,result);
		return result;
	}
	viennacl::matrix<DefaultScalarType> mat;
	viennacl::matrix<DefaultScalarType> vec;
	viennacl::matrix<DefaultScalarType> res;
	static const char* ARPACK_CODE;
};
const char* GPUDenseMatrixOperation::ARPACK_CODE = "LM";
#endif

#endif
