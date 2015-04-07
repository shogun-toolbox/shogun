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

#ifndef ELEMENTWISE_PRODUCT_IMPL_H_
#define ELEMENTWISE_PRODUCT_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#include <viennacl/linalg/matrix_operations.hpp>
#endif // HAVE_VIENNACL

namespace shogun
{

namespace linalg
{

namespace implementation
{

/** Generic class which is specialized for different backends to perform
 * elementwise multiplication
 */
template <enum Backend, class Matrix>
struct elementwise_product
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Performs the operation C = A .* B where ".*" denotes elementwise multiplication */
	static void compute(Matrix A, Matrix B, Matrix C);
};

#ifdef HAVE_EIGEN3

/** Specialization of elementwise_product for the Eigen3 backend */
template <class Matrix>
struct elementwise_product<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Return type */
	typedef SGMatrix<T> ReturnType;

	/** Eigen3 matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/** Performs the operation C = A .* B where ".*" denotes elementwise multiplication.
	 *
	 * This version returns the result in a newly created matrix. If elementwise-product
	 * is desired that will work irrespective of the backend and the matrix type used,
	 * then this method should be used.
	 *
	 * @param A First matrix
	 * @param B Second matrix
	 * @return The result of the operation
	 */
	static ReturnType compute(SGMatrix<T> A, SGMatrix<T> B)
	{
		REQUIRE(A.matrix, "Matrix A is not initialized!\n");
		REQUIRE(B.matrix, "Matrix A is not initialized!\n");

		REQUIRE(A.num_rows == B.num_rows && A.num_cols == B.num_cols,
				"Dimension mismatch! A(%d x %d) vs B(%d x %d)\n",
				A.num_rows, A.num_cols, B.num_rows, B.num_cols);

		ReturnType retMatrix(A.num_rows, A.num_cols);
		compute(A, B, retMatrix);

		return retMatrix;
	}

	/** Performs the operation C = A .* B where ".*" denotes elementwise multiplication.
	 *
	 * This version should be used for backend specific code requirements. For example,
	 * use this with CGPUMatrix and explicitly set ViennaCL backend, or SGMatrix and
	 * explicitly set Eigen3 backend. If matrix-type/backend-type independent code is
	 * desired, use the version that does not support preallocated result matrix but
	 * returns the result in a newly created matrix instead.
	 *
	 * @param A First matrix
	 * @param B Second matrix
	 * @param C Result of the operation
	 */
	static void compute(SGMatrix<T> A, SGMatrix<T> B, SGMatrix<T> C)
	{
		Eigen::Map<MatrixXt> A_eig = A;
		Eigen::Map<MatrixXt> B_eig = B;
		Eigen::Map<MatrixXt> C_eig = C;

		C_eig = A_eig.array() * B_eig.array();
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/** Specialization of elementwise_product for the ViennaCL backend */
template <class Matrix>
struct elementwise_product<Backend::VIENNACL, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Return type */
	typedef CGPUMatrix<T> ReturnType;

	/** Performs the operation C = A .* B where ".*" denotes elementwise multiplication.
	 *
	 * This version returns the result in a newly created matrix. If elementwise-product
	 * is desired that will work irrespective of the backend and the matrix type used,
	 * then this method should be used.
	 *
	 * @param A First matrix
	 * @param B Second matrix
	 * @return The result of the operation
	 */
	static ReturnType compute(CGPUMatrix<T> A, CGPUMatrix<T> B)
	{
		REQUIRE(A.matrix, "Matrix A is not initialized!\n");
		REQUIRE(B.matrix, "Matrix A is not initialized!\n");

		REQUIRE(A.num_rows == B.num_rows && A.num_cols == B.num_cols,
				"Dimension mismatch! A(%d x %d) vs B(%d x %d)\n",
				A.num_rows, A.num_cols, B.num_rows, B.num_cols);

		ReturnType retMatrix(A.num_rows, A.num_cols);
		compute(A, B, retMatrix);

		return retMatrix;
	}

	/** Performs the operation C = A .* B where ".*" denotes elementwise multiplication.
	 *
	 * This version should be used for backend specific code requirements. For example,
	 * use this with CGPUMatrix and explicitly set ViennaCL backend, or SGMatrix and
	 * explicitly set Eigen3 backend. If matrix-type/backend-type independent code is
	 * desired, use the version that does not support preallocated result matrix but
	 * returns the result in a newly created matrix instead.
	 *
	 * @param A First matrix
	 * @param B Second matrix
	 * @param C Result of the operation
	 */
	static void compute(CGPUMatrix<T> A, CGPUMatrix<T> B, CGPUMatrix<T> C)
	{
		C.vcl_matrix() = viennacl::linalg::element_prod(A.vcl_matrix(), B.vcl_matrix());
	}
};

#endif // HAVE_VIENNACL

}

}

}
#endif // ELEMENTWISE_PRODUCT_IMPL_H_
