/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Soumyajit De
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

#ifndef APPLY_IMPL_H_
#define APPLY_IMPL_H_

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

#include <type_traits>

namespace shogun
{

namespace linalg
{

namespace implementation
{

/**
 * @brief Generic class which is specialized for different backends to perform apply
 */
template <enum Backend, class Matrix, class Vector>
struct apply
{
};

/**
 * @brief Partial specialization of apply for the Eigen3 backend
 */
template <class Matrix, class Vector>
struct apply<Backend::EIGEN3, Matrix, Vector>
{
	static_assert(std::is_same<typename Matrix::Scalar,typename Vector::Scalar>::value,
			"Different numeric scalar types for matrix and vector not allowed!\n");

	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Eigen matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/** Eigen Vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/** Performs the operation of matrix applied to a vector \f$x = Ab\f$.
	 *
	 * @param A The matrix
	 * @param b The vector
	 * @param transpose Whether to transpose A before applying to b
	 * @return x Result vector
	 */
	static SGVector<T> compute(SGMatrix<T> A, SGVector<T> b, bool transpose)
	{
		SGVector<T> x(A.num_rows);
		compute(A, b, x, transpose);
		return x;
	}

	/** Performs the operation of matrix applied to a vector \f$x = Ab\f$.
	 *
	 * @param A The matrix
	 * @param b The vector
	 * @param x Result vector
	 * @param transpose Whether to transpose A before applying to b
	 */
	static void compute(SGMatrix<T> A, SGVector<T> b, SGVector<T> x, bool transpose)
	{
		Eigen::Map<MatrixXt> A_eig=A;
		Eigen::Map<VectorXt> b_eig=b;
		Eigen::Map<VectorXt> x_eig=x;

		if (transpose)
			x_eig=A_eig.transpose()*b_eig;
		else
			x_eig=A_eig*b_eig;
	}
};

#ifdef HAVE_VIENNACL
/**
 * @brief Partial specialization of apply for the ViennaCL backend
 */
template <class Matrix, class Vector>
struct apply<Backend::VIENNACL, Matrix, Vector>
{
	static_assert(std::is_same<typename Matrix::Scalar,typename Vector::Scalar>::value,
			"Different numeric scalar types for matrix and vector not allowed!\n");

	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Performs the operation of matrix applied to a vector \f$x = Ab\f$.
	 *
	 * @param A The matrix
	 * @param b The vector
	 * @param transpose Whether to transpose A before applying to b
	 * @return x Result vector
	 */
	static CGPUVector<T> compute(CGPUMatrix<T> A, CGPUVector<T> b, bool transpose)
	{
		CGPUVector<T> x(A.num_rows);
		compute(A, b, x, transpose);
		return x;
	}

	/** Performs the operation of matrix applied to a vector \f$x = Ab\f$.
	 *
	 * @param A The matrix
	 * @param b The vector
	 * @param x Result vector
	 * @param transpose Whether to transpose A before applying to b
	 */
	static void compute(CGPUMatrix<T> A, CGPUVector<T> b, CGPUVector<T> x, bool transpose)
	{
		if (transpose)
			x.vcl_vector()=viennacl::linalg::prod(viennacl::trans(A.vcl_matrix()), b.vcl_vector());
		else
			x.vcl_vector()=viennacl::linalg::prod(A.vcl_matrix(), b.vcl_vector());
	}
};
#endif // HAVE_VIENNACL

}

}

}
#endif // APPLY_IMPL_H_
