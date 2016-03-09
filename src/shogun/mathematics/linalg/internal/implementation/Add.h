/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Khaled Nasr
 * Written (w) 2015 Sanuj Sharma
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

#ifndef ADD_IMPL_H_
#define ADD_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <shogun/mathematics/eigen3.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#include <shogun/lib/GPUVector.h>
#include <viennacl/linalg/matrix_operations.hpp>
#include <viennacl/linalg/vector_operations.hpp>
#endif // HAVE_VIENNACL

namespace shogun
{

namespace linalg
{

namespace implementation
{

/**
 * @brief Generic class which is specialized for different backends to perform addition
 */
template <enum Backend, class Matrix>
struct add
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Performs the operation C = alpha*A + beta*B. Works for both matrices and vectors
	 * @param A first matrix
	 * @param B second matrix
	 * @param C matrix to store the result
	 * @param alpha constant to be multiplied by the first matrix
	 * @param beta constant to be multiplied by the second matrix
	 */
	static void compute(Matrix A, Matrix B, Matrix C, T alpha, T beta);
};

/**
 * @brief Partial specialization of add for the Native backend
 */
template <class Matrix>
struct add<Backend::NATIVE, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Performs the operation C = alpha*A + beta*B.
	 * @param A first matrix
	 * @param B second matrix
	 * @param alpha constant to be multiplied by the first matrix
	 * @param beta constant to be multiplied by the second matrix
	 * @return The return matrix
	 */
	static SGMatrix<T> compute(SGMatrix<T> A, SGMatrix<T> B, T alpha=1, T beta=1)
	{
		REQUIRE(A.matrix, "Matrix A is not initialized!\n");
		REQUIRE(B.matrix, "Matrix B is not initialized!\n");

		REQUIRE(A.num_rows == B.num_rows && A.num_cols == B.num_cols,
				"Dimension mismatch! A(%d x %d) vs B(%d x %d)\n",
				A.num_rows, A.num_cols, B.num_rows, B.num_cols);

		SGMatrix<T> C(A.num_rows, A.num_cols);
		compute(A.matrix, B.matrix, C.matrix, alpha, beta, A.num_rows*A.num_cols);

		return C;
	}

	/**
	 * Performs the operation C = alpha*A + beta*B.
	 * @param A first matrix
	 * @param B second matrix
	 * @param C matrix to store the result
	 * @param alpha constant to be multiplied by the first matrix
	 * @param beta constant to be multiplied by the second matrix
	 */
	static void compute(SGMatrix<T> A, SGMatrix<T> B, SGMatrix<T> C,
		T alpha=1, T beta=1)
	{
		REQUIRE(A.matrix, "Matrix A is not initialized!\n");
		REQUIRE(B.matrix, "Matrix B is not initialized!\n");
		REQUIRE(C.matrix, "Matrix C is not initialized!\n");

		REQUIRE(A.num_rows == B.num_rows && A.num_cols == B.num_cols,
				"Dimension mismatch! A(%d x %d) vs B(%d x %d)\n",
				A.num_rows, A.num_cols, B.num_rows, B.num_cols);
		REQUIRE(A.num_rows == C.num_rows && A.num_cols == C.num_cols,
				"Dimension mismatch! A(%d x %d) vs C(%d x %d)\n",
				A.num_rows, A.num_cols, C.num_rows, C.num_cols);
		compute(A.matrix, B.matrix, C.matrix, alpha, beta, A.num_rows*A.num_cols);
	}

	/**
	 * Performs the operation C = alpha*A + beta*B.
	 * @param A first vector
	 * @param B second vector
	 * @param alpha constant to be multiplied by the first vector
	 * @param beta constant to be multiplied by the second vector
	 * @return The result vector
	 */
	static SGVector<T> compute(SGVector<T> A, SGVector<T> B, T alpha=1, T beta=1)
	{
		REQUIRE(A.vlen == B.vlen, "Vectors should have same length! "
				"A(%d) vs B(%d)\n", A.vlen, B.vlen);

		SGVector<T> C(A.vlen);
		compute(A.vector, B.vector, C.vector, alpha, beta, A.vlen);

		return C;
	}

	/**
	 * Performs the operation C = alpha*A + beta*B.
	 * @param A first vector
	 * @param B second vector
	 * @param C vector to store the result
	 * @param alpha constant to be multiplied by the first vector
	 * @param beta constant to be multiplied by the second vector
	 */
	static void compute(SGVector<T> A, SGVector<T> B, SGVector<T> C,
		T alpha=1, T beta=1)
	{
		REQUIRE(A.vlen == B.vlen, "Vectors should have same length! "
				"A(%d) vs B(%d)\n", A.vlen, B.vlen);
		REQUIRE(A.vlen == C.vlen, "Vectors should have same length! "
				"A(%d) vs C(%d)\n", A.vlen, C.vlen);

		compute(A.vector, B.vector, C.vector, alpha, beta, A.vlen);
	}

	/**
	 * Performs the operation C = alpha*A + beta*B. Vectors or Matrices passed as pointers
	 * @param A first vector
	 * @param B second vector
	 * @param C vector to store the result
	 * @param alpha constant to be multiplied by the first vector
	 * @param beta constant to be multiplied by the second vector
	 * @param len length of the vectors/matrices
	 */
	static void compute(T* A, T* B, T* C, T alpha, T beta, index_t len)
	{
		for (int32_t i=0; i<len; i++)
			C[i]=alpha*A[i]+beta*B[i];
	}
};


/**
 * @brief Partial specialization of add for the Eigen3 backend
 */
template <class Matrix>
struct add<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Eigen3 matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/** Eigen3 vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/**
	 * Performs the operation C = alpha*A + beta*B.
	 * @param A first matrix
	 * @param B second matrix
	 * @param alpha constant to be multiplied by the first matrix
	 * @param beta constant to be multiplied by the second matrix
	 * @return The return matrix
	 */
	static SGMatrix<T> compute(SGMatrix<T> A, SGMatrix<T> B, T alpha=1, T beta=1)
	{
		REQUIRE(A.matrix, "Matrix A is not initialized!\n");
		REQUIRE(B.matrix, "Matrix B is not initialized!\n");

		REQUIRE(A.num_rows == B.num_rows && A.num_cols == B.num_cols,
				"Dimension mismatch! A(%d x %d) vs B(%d x %d)\n",
				A.num_rows, A.num_cols, B.num_rows, B.num_cols);

		SGMatrix<T> C(A.num_rows, A.num_cols);
		compute(A, B, C, alpha, beta);

		return C;
	}

	/**
	 * Performs the operation C = alpha*A + beta*B using Eigen3
	 * @param A first matrix
	 * @param B second matrix
	 * @param C matrix to store the result
	 * @param alpha constant to be multiplied by the first matrix
	 * @param beta constant to be multiplied by the second matrix
	 */
	static void compute(SGMatrix<T> A, SGMatrix<T> B, SGMatrix<T> C,
		T alpha, T beta)
	{
		Eigen::Map<MatrixXt> A_eig=A;
		Eigen::Map<MatrixXt> B_eig=B;
		Eigen::Map<MatrixXt> C_eig=C;

		C_eig=alpha*A_eig+beta*B_eig;
	}

	/**
	 * Performs the operation C = alpha*A + beta*B.
	 * @param A first vector
	 * @param B second vector
	 * @param alpha constant to be multiplied by the first vector
	 * @param beta constant to be multiplied by the second vector
	 * @return The result vector
	 */
	static SGVector<T> compute(SGVector<T> A, SGVector<T> B, T alpha=1, T beta=1)
	{
		REQUIRE(A.vlen == B.vlen, "Vectors should have same length! "
				"A(%d) vs B(%d)\n", A.vlen, B.vlen);

		SGVector<T> C(A.vlen);
		compute(A, B, C, alpha, beta);

		return C;
	}

	/**
	 * Performs the operation C = alpha*A + beta*B using Eigen3
	 * @param A first vector
	 * @param B second vector
	 * @param C vector to store the result
	 * @param alpha constant to be multiplied by the first vector
	 * @param beta constant to be multiplied by the second vector
	 */
	static void compute(SGVector<T> A, SGVector<T> B, SGVector<T> C,
		T alpha, T beta)
	{
		Eigen::Map<VectorXt> A_eig=A;
		Eigen::Map<VectorXt> B_eig=B;
		Eigen::Map<VectorXt> C_eig=C;

		C_eig=alpha*A_eig+beta*B_eig;
	}
};

#ifdef HAVE_VIENNACL

/**
 * @brief Partial specialization of add for the ViennaCL backend
 */
template <class Matrix>
struct add<Backend::VIENNACL, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Performs the operation C = alpha*A + beta*B.
	 * @param A first matrix
	 * @param B second matrix
	 * @param alpha constant to be multiplied by the first matrix
	 * @param beta constant to be multiplied by the second matrix
	 * @return The return matrix
	 */
	static CGPUMatrix<T> compute(CGPUMatrix<T> A, CGPUMatrix<T> B, T alpha=1, T beta=1)
	{
		REQUIRE(A.matrix, "Matrix A is not initialized!\n");
		REQUIRE(B.matrix, "Matrix B is not initialized!\n");

		REQUIRE(A.num_rows == B.num_rows && A.num_cols == B.num_cols,
				"Dimension mismatch! A(%d x %d) vs B(%d x %d)\n",
				A.num_rows, A.num_cols, B.num_rows, B.num_cols);

		CGPUMatrix<T> C(A.num_rows, A.num_cols);
		compute(A, B, C, alpha, beta);

		return C;
	}

	/**
	 * Performs the operation C = alpha*A + beta*B using Viennacl
	 * @param A first matrix
	 * @param B second matrix
	 * @param C matrix to store the result
	 * @param alpha constant to be multiplied by the first matrix
	 * @param beta constant to be multiplied by the second matrix
	 */
	static void compute(CGPUMatrix<T> A, CGPUMatrix<T> B, CGPUMatrix<T> C,
		T alpha, T beta)
	{
		C.vcl_matrix()=alpha*A.vcl_matrix()+beta*B.vcl_matrix();
	}

	/**
	 * Performs the operation C = alpha*A + beta*B.
	 * @param A first vector
	 * @param B second vector
	 * @param alpha constant to be multiplied by the first vector
	 * @param beta constant to be multiplied by the second vector
	 * @return The result vector
	 */
	static CGPUVector<T> compute(CGPUVector<T> A, CGPUVector<T> B, T alpha=1, T beta=1)
	{
		REQUIRE(A.vlen == B.vlen, "Vectors should have same length! "
				"A(%d) vs B(%d)\n", A.vlen, B.vlen);

		CGPUVector<T> C(A.vlen);
		compute(A, B, C, alpha, beta);

		return C;
	}

	/**
	 * Performs the operation C = alpha*A + beta*B using Viennacl
	 * @param A first vector
	 * @param B second vector
	 * @param C vector to store the result
	 * @param alpha constant to be multiplied by the first vector
	 * @param beta constant to be multiplied by the second vector
	 */
	static void compute(CGPUVector<T> A, CGPUVector<T> B, CGPUVector<T> C,
		T alpha, T beta)
	{
		C.vcl_vector()=alpha*A.vcl_vector()+beta*B.vcl_vector();
	}
};

#endif // HAVE_VIENNACL

}

}

}
#endif // ADD_IMPL_H_
