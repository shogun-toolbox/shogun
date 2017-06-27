/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 2016 Pan Deng, Soumyajit De, Heiko Strathmann, Viktor Gal
 */

#ifndef LINALG_BACKEND_VIENNACL_KERNELS_H__
#define LINALG_BACKEND_VIENNACL_KERNELS_H__

#include <shogun/lib/common.h>

#ifdef HAVE_VIENNACL
#include <shogun/mathematics/linalg/internal/opencl_util.h>
#include <memory>

namespace shogun
{
	/** Generates the cross entropy computation kernel
	 * The OpenCL kernel that helps to calculate the cross entropy SGMatrices
	 */
	template <typename T>
	static viennacl::ocl::kernel& generate_cross_entropy_kernel()
	{
		std::string kernel_name =
		    "cross_entropy_" +
		    linalg::implementation::ocl::get_type_string<T>();

		if (linalg::implementation::ocl::kernel_exists(kernel_name))
			return linalg::implementation::ocl::get_kernel(kernel_name);

		std::string source =
		    linalg::implementation::ocl::generate_kernel_preamble<T>(
		        kernel_name);

		source.append(
		    R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* p, int size, int p_offset,
					__global DATATYPE* q, int q_offset,
					__global DATATYPE* result)
				{
					__local DATATYPE buffer[WORK_GROUP_SIZE_1D];

					int local_id = get_local_id(0);

					DATATYPE thread_sum = 0;
					for (int i=local_id; i<size; i+=WORK_GROUP_SIZE_1D)
						thread_sum += p[i+p_offset]*log(q[i+q_offset]+1e-30);

					buffer[local_id] = thread_sum;

					for (int j = WORK_GROUP_SIZE_1D/2; j > 0; j = j>>1)
					{
						barrier(CLK_LOCAL_MEM_FENCE);
						if (local_id < j)
							buffer[local_id] += buffer[local_id + j];
					}

					barrier(CLK_LOCAL_MEM_FENCE);

					if (get_global_id(0)==0)
						*result = -1*buffer[0];
				}
			)");

		viennacl::ocl::kernel& kernel =
		    linalg::implementation::ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);
		kernel.global_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/** Generates the max computation kernel
	 * The OpenCL kernel that helps to calculate the max of SGVector or SGMatrix
	 */
	template <typename T>
	static viennacl::ocl::kernel& generate_max_kernel()
	{
		std::string kernel_name = "max_" + linalg::implementation::ocl::get_type_string<T>();

		if (linalg::implementation::ocl::kernel_exists(kernel_name))
			return linalg::implementation::ocl::get_kernel(kernel_name);

		std::string source = linalg::implementation::ocl::generate_kernel_preamble<T>(kernel_name);

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

		viennacl::ocl::kernel& kernel = linalg::implementation::ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);
		kernel.global_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/** Generates the softmax computation kernel
	 * The OpenCL kernel that helps to calculate the softmax of SGMatrix
	 */
	template <class T>
	static viennacl::ocl::kernel& generate_softmax_kernel()
	{
		std::string kernel_name =
		    "softmax_" + linalg::implementation::ocl::get_type_string<T>();

		if (linalg::implementation::ocl::kernel_exists(kernel_name))
			return linalg::implementation::ocl::get_kernel(kernel_name);

		std::string source =
		    linalg::implementation::ocl::generate_kernel_preamble<T>(
		        kernel_name);

		source.append(
		    R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* A, int nrows, int ncols, int offset)
				{
					int j = get_global_id(0);

					if (j>=ncols)
						return;

					DATATYPE col_max = -INFINITY;
					for (int i=0; i<nrows; i++)
						col_max = max(col_max, A[offset + i+j*nrows]);

					DATATYPE col_sum = 0;
					for (int i=0; i<nrows; i++)
						col_sum += exp(A[offset + i+j*nrows]-col_max);

					DATATYPE normalizer = log(col_sum);
					for (int i=0; i<nrows; i++)
					{
						int index = offset + i+j*nrows;
						A[index] = exp(A[index]-col_max-normalizer);
					}
				}
			)");

		viennacl::ocl::kernel& kernel =
		    linalg::implementation::ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/** Generates the squared error computation kernel
	 * The OpenCL kernel that helps to calculate the squared error of SGMatrices
	 */
	template <class T>
	static viennacl::ocl::kernel& generate_squared_error_kernel()
	{
		std::string kernel_name =
		    "squared_error_" +
		    linalg::implementation::ocl::get_type_string<T>();

		if (linalg::implementation::ocl::kernel_exists(kernel_name))
			return linalg::implementation::ocl::get_kernel(kernel_name);

		std::string source =
		    linalg::implementation::ocl::generate_kernel_preamble<T>(
		        kernel_name);

		source.append(
		    R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* p, int size, int p_offset,
					__global DATATYPE* q, int q_offset,
					__global DATATYPE* result)
				{
					__local DATATYPE buffer[WORK_GROUP_SIZE_1D];

					int local_id = get_local_id(0);

					DATATYPE thread_sum = 0;
					for (int i=local_id; i<size; i+=WORK_GROUP_SIZE_1D)
						thread_sum += pown(p[i+p_offset]-q[i+q_offset], 2);

					buffer[local_id] = thread_sum;

					for (int j = WORK_GROUP_SIZE_1D/2; j > 0; j = j>>1)
					{
						barrier(CLK_LOCAL_MEM_FENCE);
						if (local_id < j)
							buffer[local_id] += buffer[local_id + j];
					}

					barrier(CLK_LOCAL_MEM_FENCE);

					if (get_global_id(0)==0)
						*result = 0.5*buffer[0];
				}
			)");

		viennacl::ocl::kernel& kernel =
		    linalg::implementation::ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);
		kernel.global_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/** Generates the sum computation kernel
	 * The OpenCL kernel that helps to calculate the sum of SGMatrix
	 *
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 */
	template <class T>
	static viennacl::ocl::kernel& generate_sum_kernel(bool no_diag)
	{
		std::string kernel_name = "sum_" + linalg::implementation::ocl::get_type_string<T>();
		if (no_diag) kernel_name.append("_no_diag");

		if (linalg::implementation::ocl::kernel_exists(kernel_name))
			return linalg::implementation::ocl::get_kernel(kernel_name);

		std::string source = linalg::implementation::ocl::generate_kernel_preamble<T>(kernel_name);
		if (no_diag) source.append("#define NO_DIAG\n");

		source.append(
			R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* mat, int nrows, int ncols, int offset,
					__global DATATYPE* result)
				{
					__local DATATYPE buffer[WORK_GROUP_SIZE_1D];
					int size = nrows*ncols;

					int local_id = get_local_id(0);

					DATATYPE thread_sum = 0;
					for (int i=local_id; i<size; i+=WORK_GROUP_SIZE_1D)
					{
					#ifdef NO_DIAG
						if (!(i/nrows == i%nrows))
					#endif
						thread_sum += mat[i+offset];
					}

					buffer[local_id] = thread_sum;

					for (int j = WORK_GROUP_SIZE_1D/2; j > 0; j = j>>1)
					{
						barrier(CLK_LOCAL_MEM_FENCE);
						if (local_id < j)
							buffer[local_id] += buffer[local_id + j];
					}

					barrier(CLK_LOCAL_MEM_FENCE);

					if (get_global_id(0)==0)
						*result = buffer[0];
				}
			)"
		);

		viennacl::ocl::kernel& kernel =
			linalg::implementation::ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);
		kernel.global_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/** Generates the colwise sum computation kernel
	 * The OpenCL kernel that helps to calculate colwise sum of SGMatrix
	 *
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 */
	template <class T>
	static viennacl::ocl::kernel& generate_colwise_sum_kernel(bool no_diag)
	{
		std::string kernel_name = "colwise_sum_" + linalg::implementation::ocl::get_type_string<T>();
		if (no_diag) kernel_name.append("_no_diag");

		if (linalg::implementation::ocl::kernel_exists(kernel_name))
			return linalg::implementation::ocl::get_kernel(kernel_name);

		std::string source = linalg::implementation::ocl::generate_kernel_preamble<T>(kernel_name);
		if (no_diag) source.append("#define NO_DIAG\n");

		source.append(
			R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* mat, int nrows, int ncols, int offset,
					__global DATATYPE* result, int result_offset)
				{
					int j = get_global_id(0);

					if (j>=ncols)
						return;

					DATATYPE sum = 0;
					for (int i=0; i<nrows; i++)
					{
					#ifdef NO_DIAG
						if (i!=j)
					#endif
						sum += mat[offset+i+j*nrows];
					}

					result[j+result_offset] = sum;
				}
			)"
		);

		viennacl::ocl::kernel& kernel =
			linalg::implementation::ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/** Generates the rowwise sum computation kernel
	 * The OpenCL kernel that helps to calculate rowwise sum of SGMatrix
	 *
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 */
	template <class T>
	static viennacl::ocl::kernel& generate_rowwise_sum_kernel(bool no_diag)
	{
		std::string kernel_name = "rowwise_sum_" + linalg::implementation::ocl::get_type_string<T>();
		if (no_diag) kernel_name.append("_no_diag");

		if (linalg::implementation::ocl::kernel_exists(kernel_name))
			return linalg::implementation::ocl::get_kernel(kernel_name);

		std::string source = linalg::implementation::ocl::generate_kernel_preamble<T>(kernel_name);
		if (no_diag) source.append("#define NO_DIAG\n");

		source.append(
			R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* mat, int nrows, int ncols, int offset,
					__global DATATYPE* result, int result_offset)
				{
					int i = get_global_id(0);

					if (i>=nrows)
						return;

					DATATYPE sum = 0;
					for (int j=0; j<ncols; j++)
					{
					#ifdef NO_DIAG
						if (i!=j)
					#endif
						sum += mat[offset+i+j*nrows];
					}

					result[i+result_offset] = sum;
				}
			)"
		);

		viennacl::ocl::kernel& kernel = linalg::implementation::ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

}
#endif // HAVE_VIENNACL

#endif // LINALG_BACKEND_VIENNACL_KERNELS_H__
