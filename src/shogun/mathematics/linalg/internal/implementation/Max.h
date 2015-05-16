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

#ifndef MAX_IMPL_H_
#define MAX_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
#include <shogun/mathematics/linalg/internal/opencl_util.h>
#include <shogun/lib/GPUMatrix.h>
#include <shogun/lib/GPUVector.h>
#endif

#include <string>

namespace shogun
{

namespace linalg
{

namespace implementation
{

/**
 * @brief Generic class which is specialized for different backends to perform
 * the max operation
 */
template <enum Backend,class Matrix>
struct max
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Returns the largest element in a matrix or a vector.
	 * @param m input matrix or vector
	 * @return largest value in the input matrix or vector
	 */
	static T compute(Matrix m);
};

/**
 * @brief Specialization of add for the Native backend
 */
template <class Matrix>
struct max<Backend::NATIVE, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Returns the largest element in a matrix.
	 * @param mat input matrix
	 * @return largest value in the input matrix
	 */
	static T compute(SGMatrix<T> mat)
	{
		REQUIRE(mat.num_cols*mat.num_rows > 0, "Matrix can not be empty!\n");
		return compute(mat.matrix, mat.num_cols*mat.num_rows);
	}

	/**
	 * Returns the largest element in a vector.
	 * @param vec input vector
	 * @return largest value in the input vector
	 */
	static T compute(SGVector<T> vec)
	{
		REQUIRE(vec.vlen > 0, "Vector can not be empty!\n");
		return compute(vec.vector, vec.vlen);
	}

	/**
	 * Returns the largest element in a vector or matrix passed as a pointer.
	 * @param vec input vector or matrix
	 * @param len length of the vector or matrix
	 * @return largest value in the input vector or matrix
	 */
	static T compute(T* vec, index_t len)
	{
		return *std::max_element(vec, vec+len);
	}
};

#ifdef HAVE_EIGEN3

/**
 * @brief Specialization of max for the Eigen3 backend
 */
template <class Matrix>
struct max<Backend::EIGEN3,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Eigen3 matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/** Eigen3 vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/**
	 * Returns the largest element in a matrix
	 * @param mat input matrix
	 * @return largest value in the matrix
	 */
	static T compute(SGMatrix<T> mat)
	{
		Eigen::Map<MatrixXt> m = mat;

		return m.maxCoeff();
	}

	/**
	 * Returns the largest element in a vector
	 * @param vec input vector
	 * @return largest value in the matrix
	 */
	static T compute(SGVector<T> vec)
	{
		Eigen::Map<VectorXt> v = vec;

		return v.maxCoeff();
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/**
 * @brief Specialization of max for the ViennaCL backend
 */
template <class Matrix>
struct max<Backend::VIENNACL,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Generates the computation kernel */
	template <class T>
	static viennacl::ocl::kernel& generate_kernel()
	{
		std::string kernel_name = "max_" + ocl::get_type_string<T>();

		if (ocl::kernel_exists(kernel_name))
			return ocl::get_kernel(kernel_name);

		std::string source = ocl::generate_kernel_preamble<T>(kernel_name);

		source.append(
			R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* vec, int size, int offset,
					__global DATATYPE* result)
				{
					__local DATATYPE buffer[WORK_GROUP_SIZE_1D];

					int local_id = get_local_id(0);

					DATATYPE thread_max = -INFINITY;
					for (int i=local_id; i<size; i+=WORK_GROUP_SIZE_1D)
					{
						DATATYPE v = vec[i+offset];
						thread_max = max(v, thread_max);
					}

					buffer[local_id] = thread_max;

					for (int j = WORK_GROUP_SIZE_1D/2; j > 0; j = j>>1)
					{
						barrier(CLK_LOCAL_MEM_FENCE);
						if (local_id < j)
							buffer[local_id] = max(buffer[local_id], buffer[local_id + j]);
					}

					barrier(CLK_LOCAL_MEM_FENCE);

					if (get_global_id(0)==0)
						*result = buffer[0];
				}
			)"
		);

		viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);
		kernel.global_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/**
	 * Returns the largest element in a matrix
	 * @param mat input matrix
	 * @return largest value in the matrix
	 */
	static T compute(CGPUMatrix<T> mat)
	{
		viennacl::ocl::kernel& kernel = generate_kernel<T>();

		CGPUVector<T> result(1);

		viennacl::ocl::enqueue(kernel(mat.vcl_matrix(),
			cl_int(mat.num_rows*mat.num_cols), cl_int(mat.offset),
			result.vcl_vector()));

		return result[0];
	}

	/**
	 * Returns the largest element in a vector
	 * @param vec input vector
	 * @return largest value in the vector
	 */
	static T compute(CGPUVector<T> vec)
	{
		viennacl::ocl::kernel& kernel = generate_kernel<T>();

		CGPUVector<T> result(1);

		viennacl::ocl::enqueue(kernel(vec.vcl_vector(),
			cl_int(vec.vlen), cl_int(vec.offset),
			result.vcl_vector()));

		return result[0];
	}
};
#endif // HAVE_VIENNACL

}

}

}
#endif // MAX_IMPL_H_
