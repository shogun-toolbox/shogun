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

#ifndef SPECIAL_PURPOSE_IMPL_H_
#define SPECIAL_PURPOSE_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#include <shogun/mathematics/linalg/internal/opencl_util.h>
#endif // HAVE_VIENNACL

namespace shogun
{

namespace linalg
{
	
namespace implementation
{

namespace special_purpose
{
	
/** Generic class which is specialized for different backends to perform 
 * the logistic operation
 */
template <enum Backend, class Matrix>
struct logistic
{
	typedef typename Matrix::Scalar T;
	
	/** Applies the elementwise logistic function f(x) = 1/(1+exp(-x)) to a matrix */
	static void compute(Matrix A, Matrix result);
};

#ifdef HAVE_EIGEN3

/** Specialization of logistic for the Eigen3 backend */
template <> template <class Matrix>
struct logistic<Backend::EIGEN3, Matrix>
{
	typedef typename Matrix::Scalar T;
	
	/** Applies the elementwise logistic function f(x) = 1/(1+exp(-x)) to a matrix */
	static void compute(SGMatrix<T> A, SGMatrix<T> result)
	{
		int32_t len = A.num_rows*A.num_cols;
		for (int32_t i=0; i<len; i++)
			result[i] = 1.0/(1+CMath::exp(-1*A[i]));
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/** Specialization of logistic for the ViennaCL backend */
template <> template <class Matrix>
struct logistic<Backend::VIENNACL, Matrix>
{
	typedef typename Matrix::Scalar T;
	
	/** Sets each row of a matrix to some constant value. That is, perfoms the 
	 * operation A[i,j] = v[i], for all i and j
	 */
	static void compute(CGPUMatrix<T> A, CGPUMatrix<T> result)
	{
		const std::string operation = "return 1.0/(1+exp(-1*element));";
		
		std::string kernel_name = "logistic_" + ocl::get_type_string<T>();
		viennacl::ocl::kernel& kernel = 
			ocl::generate_single_arg_elementwise_kernel<T>(kernel_name, operation);
		
		kernel.global_work_size(0, ocl::align_to_multiple_1d(A.num_rows*A.num_cols));
		
		viennacl::ocl::enqueue(kernel(A.vcl_matrix(), 
			cl_int(A.num_rows*A.num_cols), cl_int(A.offset), 
			result.vcl_matrix(), cl_int(result.offset)));
	}
};

#endif // HAVE_VIENNACL

/** Generic class which is specialized for different backends to perform 
 * the multiply_by_logistic_derivative operation
 */
template <enum Backend, class Matrix>
struct multiply_by_logistic_derivative
{
	typedef typename Matrix::Scalar T;
	
	/** Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j) for all i and j*/ 
	static void compute(Matrix A, Matrix C);
};

#ifdef HAVE_EIGEN3

/** Specialization of multiply_by_logistic_derivative for the Eigen3 backend */
template <> template <class Matrix>
struct multiply_by_logistic_derivative<Backend::EIGEN3, Matrix>
{
	typedef typename Matrix::Scalar T;
	
	/** Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j) for all i and j*/ 
	static void compute(SGMatrix<T> A, SGMatrix<T> C)
	{
		int32_t len = A.num_rows*A.num_cols;
		for (int32_t i=0; i<len; i++)
			C[i] *= A[i] * (1.0-A[i]);
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/** Specialization of multiply_by_logistic_derivative for the ViennaCL backend */
template <> template <class Matrix>
struct multiply_by_logistic_derivative<Backend::VIENNACL, Matrix>
{
	typedef typename Matrix::Scalar T;
	
	/** Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j) for all i and j*/ 
	static void compute(CGPUMatrix<T> A, CGPUMatrix<T> C)
	{
		const std::string operation = "return element2 * element1*(1.0-element1);";
		
		std::string kernel_name = "multiply_by_logistic_derivative_" + ocl::get_type_string<T>();
		viennacl::ocl::kernel& kernel = 
			ocl::generate_two_arg_elementwise_kernel<T>(kernel_name, operation);
		
		kernel.global_work_size(0, ocl::align_to_multiple_1d(A.num_rows*A.num_cols));
		
		viennacl::ocl::enqueue(kernel(
			A.vcl_matrix(), cl_int(A.num_rows*A.num_cols), cl_int(A.offset),
			C.vcl_matrix(), cl_int(C.offset),
			C.vcl_matrix(), cl_int(C.offset)));
	}
};

#endif // HAVE_VIENNACL

}

}

}

}
#endif // SPECIAL_PURPOSE_IMPL_H_
