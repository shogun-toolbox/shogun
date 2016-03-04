/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Khaled Nasr
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef MATRIX_PRODUCT_IMPL_H_
#define MATRIX_PRODUCT_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>

#include <shogun/mathematics/eigen3.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#include <viennacl/linalg/prod.hpp>
#include <viennacl/matrix.hpp>
#endif // HAVE_VIENNACL

namespace shogun
{

namespace linalg
{

namespace implementation
{

/** Generic class which is specialized for different backends to compute matrix
 * products
 */
template <enum Backend, class Matrix>
struct matrix_product
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Performs matrix multiplication
	 *
	 * @param A First matrix
	 * @param B Second matrix
	 * @param C Result of the operation
	 * @param transpose_A Whether to the transpose of A should be used instead of A
	 * @param transpose_B Whether to the transpose of B should be used instead of B
	 * @param overwrite If true, the values in C are overwritten with the result,
	 * otherwise, the result is added to the existing values
	 */
	static void compute(Matrix A, Matrix B, Matrix C,
		bool transpose_A, bool transpose_B, bool overwrite);
};


/** Specialization of matrix_product for the Eigen3 backend */
template <class Matrix>
struct matrix_product<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Return type */
	typedef SGMatrix<T> ReturnType;

	/** Eigen3 matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/** Performs matrix multiplication
	 *
	 * @param A First matrix
	 * @param B Second matrix
	 * @param transpose_A Whether to the transpose of A should be used instead of A
	 * @param transpose_B Whether to the transpose of B should be used instead of B
	 * @return Result of the operation
	 */
	static ReturnType compute(SGMatrix<T> A, SGMatrix<T> B,
		bool transpose_A, bool transpose_B)
	{
		REQUIRE(A.matrix, "Matrix A is not initialized!\n");
		REQUIRE(B.matrix, "Matrix B is not initialized!\n");
		REQUIRE(A.num_cols == B.num_rows, "Number of columns for A (%d) and "
				"number of rows for B (%d) should be equal!\n", A.num_cols, B.num_rows);

		ReturnType retMatrix(A.num_rows, B.num_cols);
		compute(A, B, retMatrix, transpose_A, transpose_B, true);

		return retMatrix;
	}

	/** Performs matrix multiplication
	 *
	 * @param A First matrix
	 * @param B Second matrix
	 * @param C Result of the operation
	 * @param transpose_A Whether to the transpose of A should be used instead of A
	 * @param transpose_B Whether to the transpose of B should be used instead of B
	 * @param overwrite If true, the values in C are overwritten with the result,
	 * otherwise, the result is added to the existing values
	 */
	static void compute(SGMatrix<T> A, SGMatrix<T> B, SGMatrix<T> C,
		bool transpose_A, bool transpose_B, bool overwrite)
	{
		Eigen::Map<MatrixXt> A_eig = A;
		Eigen::Map<MatrixXt> B_eig = B;
		Eigen::Map<MatrixXt> C_eig = C;

		if (overwrite)
		{
			if (transpose_A && transpose_B)
				C_eig = A_eig.transpose() * B_eig.transpose();

			else if (transpose_A)
				C_eig = A_eig.transpose() * B_eig;

			else if (transpose_B)
				C_eig = A_eig * B_eig.transpose();

			else
				C_eig = A_eig * B_eig;
		}
		else
		{
			if (transpose_A && transpose_B)
				C_eig += A_eig.transpose() * B_eig.transpose();

			else if (transpose_A)
				C_eig += A_eig.transpose() * B_eig;

			else if (transpose_B)
				C_eig += A_eig * B_eig.transpose();

			else
				C_eig += A_eig * B_eig;
		}
	}
};

#ifdef HAVE_VIENNACL

/** Specialization of matrix_product for the ViennaCL backend */
template <class Matrix>
struct matrix_product<Backend::VIENNACL, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Return type */
	typedef CGPUMatrix<T> ReturnType;

	/** Performs matrix multiplication
	 *
	 * @param A First matrix
	 * @param B Second matrix
	 * @param transpose_A Whether to the transpose of A should be used instead of A
	 * @param transpose_B Whether to the transpose of B should be used instead of B
	 * @return Result of the operation
	 */
	static ReturnType compute(CGPUMatrix<T> A, CGPUMatrix<T> B,
		bool transpose_A, bool transpose_B)
	{
		REQUIRE(A.matrix, "Matrix A is not initialized!\n");
		REQUIRE(B.matrix, "Matrix B is not initialized!\n");
		REQUIRE(A.num_cols == B.num_rows, "Number of columns for A (%d) and "
				"number of rows for B (%d) should be equal!\n", A.num_cols, B.num_rows);

		ReturnType retMatrix(A.num_rows, B.num_cols);
		compute(A, B, retMatrix, transpose_A, transpose_B, true);

		return retMatrix;
	}

	/** Performs matrix multiplication
	 *
	 * @param A First matrix
	 * @param B Second matrix
	 * @param C Result of the operation
	 * @param transpose_A Whether to the transpose of A should be used instead of A
	 * @param transpose_B Whether to the transpose of B should be used instead of B
	 * @param overwrite If true, the values in C are overwritten with the result,
	 * otherwise, the result is added to the existing values
	 */
	static void compute(CGPUMatrix<T> A, CGPUMatrix<T> B, CGPUMatrix<T> C,
		bool transpose_A, bool transpose_B, bool overwrite)
	{
		if (overwrite)
		{
			if (transpose_A && transpose_B)
				C.vcl_matrix() = viennacl::linalg::prod(
					viennacl::trans(A.vcl_matrix()), viennacl::trans(B.vcl_matrix()));

			else if (transpose_A)
				C.vcl_matrix() = viennacl::linalg::prod(
					viennacl::trans(A.vcl_matrix()), B.vcl_matrix());

			else if (transpose_B)
				C.vcl_matrix() = viennacl::linalg::prod(
					A.vcl_matrix(), viennacl::trans(B.vcl_matrix()));

			else
				C.vcl_matrix() = viennacl::linalg::prod(A.vcl_matrix(), B.vcl_matrix());
		}
		else
		{
			if (transpose_A && transpose_B)
				C.vcl_matrix() += viennacl::linalg::prod(
					viennacl::trans(A.vcl_matrix()), viennacl::trans(B.vcl_matrix()));

			else if (transpose_A)
				C.vcl_matrix() += viennacl::linalg::prod(
					viennacl::trans(A.vcl_matrix()), B.vcl_matrix());

			else if (transpose_B)
				C.vcl_matrix() += viennacl::linalg::prod(
					A.vcl_matrix(), viennacl::trans(B.vcl_matrix()));

			else
				C.vcl_matrix() += viennacl::linalg::prod(A.vcl_matrix(), B.vcl_matrix());
		}
	}
};

#endif // HAVE_VIENNACL

}

}

}
#endif // MATRIX_PRODUCT_IMPL_H_
