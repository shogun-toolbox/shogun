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

#ifndef SET_ROWS_CONST_IMPL_H_
#define SET_ROWS_CONST_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <shogun/mathematics/eigen3.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#include <shogun/lib/GPUVector.h>
#include <shogun/mathematics/linalg/internal/opencl_util.h>
#endif // HAVE_VIENNACL

namespace shogun
{

namespace linalg
{

namespace implementation
{

/** Generic class which is specialized for different backends to perform
 * the set_rows_const operation
 */
template <enum Backend, class Matrix, class Vector>
struct set_rows_const
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Sets each row of a matrix to some constant value. That is, perfoms the
	 * operation A[i,j] = v[i], for all i and j
	 */
	static void compute(Matrix A, Vector v);
};


/** Specialization of set_rows_const for the Eigen3 backend */
template <class Matrix, class Vector>
struct set_rows_const<Backend::EIGEN3, Matrix, Vector>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Eigen3 matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/** Eigen3 vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/** Sets each row of a matrix to some constant value. That is, perfoms the
	 * operation A[i,j] = v[i], for all i and j
	 */
	static void compute(SGMatrix<T> A, SGVector<T> v)
	{
		Eigen::Map<MatrixXt> A_eig = A;
		Eigen::Map<VectorXt> v_eig = v;

		A_eig.colwise() = v_eig;
	}
};

#ifdef HAVE_VIENNACL

/** Specialization of set_rows_const for the ViennaCL backend */
template <class Matrix, class Vector>
struct set_rows_const<Backend::VIENNACL, Matrix, Vector>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Generates the computation kernel */
	template <class T>
	static viennacl::ocl::kernel& generate_kernel()
	{
		std::string kernel_name = "set_rows_const_" + ocl::get_type_string<T>();

		if (ocl::kernel_exists(kernel_name))
			return ocl::get_kernel(kernel_name);

		std::string source = ocl::generate_kernel_preamble<T>(kernel_name);

		source.append(
			R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* mat, int nrows, int ncols, int offset,
					__global DATATYPE* vec, int vec_offset)
				{
					int i = get_global_id(0);
					int j = get_global_id(1);

					if (i>=nrows || j>=ncols)
						return;

					mat[offset + i+j*nrows] = vec[i+offset];
				}
			)"
		);

		viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_2D);
		kernel.local_work_size(1, OCL_WORK_GROUP_SIZE_2D);

		return kernel;
	}

	/** Sets each row of a matrix to some constant value. That is, perfoms the
	 * operation A[i,j] = v[i], for all i and j
	 */
	static void compute(CGPUMatrix<T> A, CGPUVector<T> v)
	{
		viennacl::ocl::kernel& kernel = generate_kernel<T>();
		kernel.global_work_size(0, ocl::align_to_multiple_2d(A.num_rows));
		kernel.global_work_size(1, ocl::align_to_multiple_2d(A.num_cols));

		viennacl::ocl::enqueue(kernel(A.vcl_matrix(),
			cl_int(A.num_rows), cl_int(A.num_cols), cl_int(A.offset),
			v.vcl_vector(), cl_int(v.offset)));
	}
};

#endif // HAVE_VIENNACL

}

}

}
#endif // SET_ROWS_CONST_IMPL_H_
