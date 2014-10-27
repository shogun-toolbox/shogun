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
template <> template <class Matrix>
struct elementwise_product<Backend::EIGEN3, Matrix>
{
	typedef typename Matrix::Scalar T;
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;
	
	/** Performs the operation C = A .* B where ".*" denotes elementwise multiplication */
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
template <> template <class Matrix>
struct elementwise_product<Backend::VIENNACL, Matrix>
{
	typedef typename Matrix::Scalar T;
	
	/** Performs the operation C = A .* B where ".*" denotes elementwise multiplication */
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
