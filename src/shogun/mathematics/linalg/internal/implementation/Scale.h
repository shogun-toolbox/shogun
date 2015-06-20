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

#ifndef SCALE_IMPL_H_
#define SCALE_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

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

/** Generic class which is specialized for different backends to perform scaling */
template <enum Backend, class Matrix>
struct scale
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Performs the operation B = alpha*A */
	static void compute(Matrix A, Matrix B, Matrix C, T alpha, T beta);
};

/** Specialization of scale for the Native backend */
template<class Matrix>
struct scale<Backend::NATIVE, Matrix>
{
	typedef typename Matrix::Scalar T;

	/** Performs the operation B = alpha*A */
	static void compute(SGMatrix<T> A, SGMatrix<T> B, T alpha)
	{
		REQUIRE((A.num_rows==B.num_rows)&&(A.num_cols==B.num_cols),
			"Dimensions of A(%dx%d) and B(%dx%d) don't match.",
			A.num_rows, A.num_cols, B.num_rows, B.num_cols);

		compute(A.matrix, B.matrix, A.num_rows*A.num_cols, alpha);
	}

	/** Performs the operation B = alpha*A */
	static void compute(SGVector<T> A, SGVector<T> B, T alpha)
	{
		REQUIRE(A.vlen==B.vlen,"Number of elements in A(%d) should be "
			"equal to number of elements in (%d)!", A.vlen, B.vlen);

		compute(A.vector, B.vector, A.vlen, alpha);
	}

	/** Performs the operation B = alpha*A for len elements*/
	static void compute(T* A, T* B, index_t len, T alpha)
	{
		REQUIRE(A!=NULL&&B!=NULL, "Invalid pointers to matrices.");

		for (index_t i=0; i<len; i++)
			B[i]=A[i]*alpha;
	}
};

#ifdef HAVE_EIGEN3

/** Specialization of scale for the Eigen3 backend */
template <class Matrix>
struct scale<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Eigen3 matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/** Eigen3 vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/** Performs the operation B = alpha*A */
	static void compute(SGMatrix<T> A, SGMatrix<T> B, T alpha)
	{
		Eigen::Map<MatrixXt> A_eig = A;
		Eigen::Map<MatrixXt> B_eig = B;

		B_eig = alpha * A_eig;
	}

	/** Performs the operation B = alpha*A */
	static void compute(SGVector<T> A, SGVector<T> B, T alpha)
	{
		Eigen::Map<VectorXt> A_eig = A;
		Eigen::Map<VectorXt> B_eig = B;

		B_eig = alpha * A_eig;
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/** Specialization of scale for the ViennaCL backend */
template <class Matrix>
struct scale<Backend::VIENNACL, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Performs the operation B = alpha*A */
	static void compute(CGPUMatrix<T> A, CGPUMatrix<T> B, T alpha)
	{
		B.vcl_matrix() = alpha*A.vcl_matrix();
	}

	/** Performs the operation B = alpha*A */
	static void compute(CGPUVector<T> A, CGPUVector<T> B, T alpha)
	{
		B.vcl_vector() = alpha*A.vcl_vector();
	}
};

#endif // HAVE_VIENNACL

}

}

}
#endif // SCALE_IMPL_H_
