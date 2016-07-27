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
	/** Generates the colwise sum computation kernel
	 * The OpenCL kernel that helps to calculate colwise sum of SGMatrix
	 */
	template <class T>
	static viennacl::ocl::kernel& generate_colwise_sum_kernel()
	{
		std::string kernel_name = "colwise_sum_" + linalg::implementation::ocl::get_type_string<T>();

		if (linalg::implementation::ocl::kernel_exists(kernel_name))
			return linalg::implementation::ocl::get_kernel(kernel_name);

		std::string source = linalg::implementation::ocl::generate_kernel_preamble<T>(kernel_name);

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
	 */
	template <class T>
	static viennacl::ocl::kernel& generate_rowwise_sum_kernel()
	{
		std::string kernel_name = "rowwise_sum_" + linalg::implementation::ocl::get_type_string<T>();

		if (linalg::implementation::ocl::kernel_exists(kernel_name))
			return linalg::implementation::ocl::get_kernel(kernel_name);

		std::string source = linalg::implementation::ocl::generate_kernel_preamble<T>(kernel_name);

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
