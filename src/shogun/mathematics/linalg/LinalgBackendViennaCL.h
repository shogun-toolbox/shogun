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

#ifndef LINALG_BACKEND_VIENNACL_H__
#define LINALG_BACKEND_VIENNACL_H__

#include <shogun/mathematics/linalg/LinalgBackendGPUBase.h>
#include <shogun/mathematics/linalg/LinalgBackendViennaclKernels.h>

#ifdef HAVE_VIENNACL

#include <shogun/mathematics/linalg/GPUMemoryViennaCL.h>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/prod.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>

#if VIENNACL_VERSION >= 10700
#include <viennacl/linalg/sum.hpp>
#endif

namespace shogun
{

	/** @brief linalg methods with ViennaCL backend
	 * implementation of @see LinalgBackendGPUBase
	 */
	class LinalgBackendViennaCL : public LinalgBackendGPUBase
	{
		template <typename T>
		friend struct GPUMemoryViennaCL;

	public:
		/** Constructor */
		LinalgBackendViennaCL()
		{
			set_derived(this);
		}

		/** Static cast @see GPUMemoryBase class to @see GPUMemoryViennaCL */
		template <typename T, template <typename> class Container>
		GPUMemoryViennaCL<T>* cast_to_viennacl(const Container<T>& a) const
		{
			return static_cast<GPUMemoryViennaCL<T>*>(a.gpu_ptr.get());
		}

		/** ViennaCL vector result = alpha * A + beta * B method */
		template <typename T>
		void add_impl(
		    const SGVector<T>& a, const SGVector<T>& b, T alpha, T beta,
		    SGVector<T>& result) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);
			GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

			result_gpu->data_vector(a.size()) =
			    alpha * a_gpu->data_vector(a.size()) +
			    beta * b_gpu->data_vector(b.size());
		}

		/** ViennaCL matrix result = alpha * A + beta * B method */
		template <typename T>
		void add_impl(
		    const SGMatrix<T>& a, const SGMatrix<T>& b, T alpha, T beta,
		    SGMatrix<T>& result) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);
			GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

			result_gpu->data_matrix(a.num_rows, a.num_cols) =
			    alpha * a_gpu->data_matrix(a.num_rows, a.num_cols) +
			    beta * b_gpu->data_matrix(b.num_rows, b.num_cols);
		}

		/** ViennaCL cross_entropy method
		 * The cross entropy is defined as \f$ H(P,Q) = - \sum_{ij}
		 * P[i,j]log(Q[i,j]) \f$
		 */
		template <typename T>
		T cross_entropy_impl(const SGMatrix<T>& p, const SGMatrix<T>& q) const
		{
			typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type
			    aligned_t;

			GPUMemoryViennaCL<T>* p_gpu = cast_to_viennacl(p);
			GPUMemoryViennaCL<T>* q_gpu = cast_to_viennacl(q);
			GPUMemoryViennaCL<T>* result_gpu = new GPUMemoryViennaCL<T>(1);

			viennacl::ocl::kernel& kernel = generate_cross_entropy_kernel<T>();
			viennacl::ocl::enqueue(
			    kernel(
			        p_gpu->data_matrix(p.num_rows, p.num_cols),
			        cl_int(p.num_rows * p.num_cols), cl_int(p_gpu->m_offset),
			        q_gpu->data_matrix(q.num_rows, q.num_cols),
			        cl_int(q_gpu->m_offset), result_gpu->data_vector(1)));

			T* result = reinterpret_cast<T*>(SG_MALLOC(aligned_t, 1));
			viennacl::backend::memory_read(
			    *(result_gpu->m_data), result_gpu->m_offset * sizeof(T),
			    sizeof(T), result);

			return result[0];
		}

		/** ViennaCL vector dot-product method. */
		template <typename T>
		T dot_impl(const SGVector<T>& a, const SGVector<T>& b) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);

			return viennacl::linalg::inner_prod(
			    a_gpu->data_vector(a.size()), b_gpu->data_vector(b.size()));
		}

		/** ViennaCL matrix in-place elementwise product method */
		template <typename T>
		void element_prod_impl(
		    const SGMatrix<T>& a, const SGMatrix<T>& b, SGMatrix<T>& result,
		    bool transpose_A, bool transpose_B) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);
			GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

			if (transpose_A && transpose_B)
				result_gpu->data_matrix(result.num_rows, result.num_cols) =
				    viennacl::linalg::element_prod(
				        viennacl::trans(
				            a_gpu->data_matrix(a.num_rows, a.num_cols)),
				        viennacl::trans(
				            b_gpu->data_matrix(b.num_rows, b.num_cols)));

			else if (transpose_A)
				result_gpu->data_matrix(result.num_rows, result.num_cols) =
				    viennacl::linalg::element_prod(
				        viennacl::trans(
				            a_gpu->data_matrix(a.num_rows, a.num_cols)),
				        b_gpu->data_matrix(b.num_rows, b.num_cols));

			else if (transpose_B)
				result_gpu->data_matrix(result.num_rows, result.num_cols) =
				    viennacl::linalg::element_prod(
				        a_gpu->data_matrix(a.num_rows, a.num_cols),
				        viennacl::trans(
				            b_gpu->data_matrix(b.num_rows, b.num_cols)));

			else
				result_gpu->data_matrix(result.num_rows, result.num_cols) =
				    viennacl::linalg::element_prod(
				        a_gpu->data_matrix(a.num_rows, a.num_cols),
				        b_gpu->data_matrix(b.num_rows, b.num_cols));
		}

		/** ViennaCL logistic method. Calculates f(x) = 1/(1+exp(-x)) */
		template <typename T>
		void logistic_impl(const SGMatrix<T>& a, SGMatrix<T>& result) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

			const std::string operation = "return 1.0/(1+exp(-1*element));";

			std::string kernel_name =
			    "logistic_" + linalg::implementation::ocl::get_type_string<T>();
			viennacl::ocl::kernel& kernel = linalg::implementation::ocl::
			    generate_single_arg_elementwise_kernel<T>(
			        kernel_name, operation);

			kernel.global_work_size(
			    0, linalg::implementation::ocl::align_to_multiple_1d(
			           a.num_rows * a.num_cols));

			viennacl::ocl::enqueue(
			    kernel(
			        a_gpu->data_matrix(a.num_rows, a.num_cols),
			        cl_int(a.num_rows * a.num_cols), cl_int(a_gpu->m_offset),
			        result_gpu->data_matrix(a.num_rows, a.num_cols),
			        cl_int(result_gpu->m_offset)));

			result.gpu_ptr = std::shared_ptr<GPUMemoryBase<T>>(
			    result_gpu->clone_vector(result_gpu, a.num_rows * a.num_cols));
		}

		/** ViennaCL matrix * vector in-place product method */
		template <typename T>
		void matrix_prod_impl(
		    const SGMatrix<T>& a, const SGVector<T>& b, SGVector<T>& result,
		    bool transpose, bool transpose_B = false) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);
			GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

			if (transpose)
				result_gpu->data_vector(result.vlen) = viennacl::linalg::prod(
				    viennacl::trans(a_gpu->data_matrix(a.num_rows, a.num_cols)),
				    b_gpu->data_vector(b.vlen));
			else
				result_gpu->data_vector(result.vlen) = viennacl::linalg::prod(
				    a_gpu->data_matrix(a.num_rows, a.num_cols),
				    b_gpu->data_vector(b.vlen));
		}

		/** ViennaCL matrices in-place product method */
		template <typename T>
		void matrix_prod_impl(
		    const SGMatrix<T>& a, const SGMatrix<T>& b, SGMatrix<T>& result,
		    bool transpose_A, bool transpose_B) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);
			GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

			if (transpose_A && transpose_B)
				result_gpu->data_matrix(result.num_rows, result.num_cols) =
				    viennacl::linalg::prod(
				        viennacl::trans(
				            a_gpu->data_matrix(a.num_rows, a.num_cols)),
				        viennacl::trans(
				            b_gpu->data_matrix(b.num_rows, b.num_cols)));

			else if (transpose_A)
				result_gpu->data_matrix(result.num_rows, result.num_cols) =
				    viennacl::linalg::prod(
				        viennacl::trans(
				            a_gpu->data_matrix(a.num_rows, a.num_cols)),
				        b_gpu->data_matrix(b.num_rows, b.num_cols));

			else if (transpose_B)
				result_gpu->data_matrix(result.num_rows, result.num_cols) =
				    viennacl::linalg::prod(
				        a_gpu->data_matrix(a.num_rows, a.num_cols),
				        viennacl::trans(
				            b_gpu->data_matrix(b.num_rows, b.num_cols)));

			else
				result_gpu->data_matrix(result.num_rows, result.num_cols) =
				    viennacl::linalg::prod(
				        a_gpu->data_matrix(a.num_rows, a.num_cols),
				        b_gpu->data_matrix(b.num_rows, b.num_cols));
		}

		/** ViennaCL max method */
		template <typename T, template <typename> class Container>
		T max_impl(const Container<T>& a) const
		{
			typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type
			    aligned_t;

			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* result_gpu = new GPUMemoryViennaCL<T>(1);

			viennacl::ocl::kernel& kernel = generate_max_kernel<T>();
			viennacl::ocl::enqueue(
			    kernel(
			        a_gpu->data_vector(a.size()), cl_int(a.size()),
			        cl_int(a_gpu->m_offset), result_gpu->data_vector(1)));

			T* result = reinterpret_cast<T*>(SG_MALLOC(aligned_t, 1));
			viennacl::backend::memory_read(
			    *(result_gpu->m_data), result_gpu->m_offset * sizeof(T),
			    sizeof(T), result);

			return result[0];
		}

		/** ViennaCL vectors or matrices mean method */
		template <typename T, template <typename> class Container>
		float64_t mean_impl(const Container<T>& a) const
		{
			return sum_impl(a) / float64_t(a.size());
		}

		/** ViennaCL multiply_by_logistic_derivative method
		 * Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j) for all
		 * i
		 * and j
		 */
		template <typename T>
		void multiply_by_logistic_derivative_impl(
		    const SGMatrix<T>& a, SGMatrix<T>& result) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

			const std::string operation =
			    "return element2 * element1*(1.0-element1);";

			std::string kernel_name =
			    "multiply_by_logistic_derivative_" +
			    linalg::implementation::ocl::get_type_string<T>();
			viennacl::ocl::kernel& kernel = linalg::implementation::ocl::
			    generate_two_arg_elementwise_kernel<T>(kernel_name, operation);

			kernel.global_work_size(
			    0, linalg::implementation::ocl::align_to_multiple_1d(
			           a.num_rows * a.num_cols));

			viennacl::ocl::enqueue(
			    kernel(
			        a_gpu->data_matrix(a.num_rows, a.num_cols),
			        cl_int(a.num_rows * a.num_cols), cl_int(a_gpu->m_offset),
			        result_gpu->data_matrix(result.num_rows, result.num_cols),
			        cl_int(result_gpu->m_offset),
			        result_gpu->data_matrix(result.num_rows, result.num_cols),
			        cl_int(result_gpu->m_offset)));

			result.gpu_ptr = std::shared_ptr<GPUMemoryBase<T>>(
			    result_gpu->clone_vector(result_gpu, a.num_rows * a.num_cols));
		}

		/** ViennaCL multiply_by_rectified_linear_derivative method
		 * Performs the operation C(i,j) = C(i,j) * (A(i,j)!=0) for all i and j
		 */
		template <typename T>
		void multiply_by_rectified_linear_derivative_impl(
		    const SGMatrix<T>& a, SGMatrix<T>& result) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

			const std::string operation = "return element1==0 ? 0 : element2;";

			std::string kernel_name =
			    "multiply_by_rectified_linear_derivative_" +
			    linalg::implementation::ocl::get_type_string<T>();
			viennacl::ocl::kernel& kernel = linalg::implementation::ocl::
			    generate_two_arg_elementwise_kernel<T>(kernel_name, operation);

			kernel.global_work_size(
			    0, linalg::implementation::ocl::align_to_multiple_1d(
			           a.num_rows * a.num_cols));

			viennacl::ocl::enqueue(
			    kernel(
			        a_gpu->data_matrix(a.num_rows, a.num_cols),
			        cl_int(a.num_rows * a.num_cols), cl_int(a_gpu->m_offset),
			        result_gpu->data_matrix(result.num_rows, result.num_cols),
			        cl_int(result_gpu->m_offset),
			        result_gpu->data_matrix(result.num_rows, result.num_cols),
			        cl_int(result_gpu->m_offset)));

			result.gpu_ptr = std::shared_ptr<GPUMemoryBase<T>>(
			    result_gpu->clone_vector(result_gpu, a.num_rows * a.num_cols));
		}

		/** Applies the elementwise rectified linear function f(x) = max(0,x) */
		template <typename T>
		void
		rectified_linear_impl(const SGMatrix<T>& a, SGMatrix<T>& result) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

			const std::string operation = "return max((DATATYPE)0,element);";

			std::string kernel_name =
			    "rectified_linear_" +
			    linalg::implementation::ocl::get_type_string<T>();
			viennacl::ocl::kernel& kernel = linalg::implementation::ocl::
			    generate_single_arg_elementwise_kernel<T>(
			        kernel_name, operation);

			kernel.global_work_size(
			    0, linalg::implementation::ocl::align_to_multiple_1d(
			           a.num_rows * a.num_cols));

			viennacl::ocl::enqueue(
			    kernel(
			        a_gpu->data_matrix(a.num_rows, a.num_cols),
			        cl_int(a.num_rows * a.num_cols), cl_int(a_gpu->m_offset),
			        result_gpu->data_matrix(result.num_rows, result.num_cols),
			        cl_int(result_gpu->m_offset)));

			result.gpu_ptr = std::shared_ptr<GPUMemoryBase<T>>(
			    result_gpu->clone_vector(result_gpu, a.num_rows * a.num_cols));
		}

		/** ViennaCL vector inplace scale method: result = alpha * A */
		template <typename T>
		void
		scale_impl(const SGVector<T>& a, SGVector<T>& result, T alpha) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

			result_gpu->data_vector(a.size()) =
			    alpha * a_gpu->data_vector(a.size());
		}

		/** ViennaCL vector inplace scale method: result = alpha * A */
		template <typename T>
		void
		scale_impl(const SGMatrix<T>& a, SGMatrix<T>& result, T alpha) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

			result_gpu->data_matrix(a.num_rows, a.num_cols) =
			    alpha * a_gpu->data_matrix(a.num_rows, a.num_cols);
		}

		/** Set const to vector or matrix with ViennaCL. */
		template <typename T, template <typename> class Container>
		void set_const_impl(Container<T>& a, T value) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
			typename GPUMemoryViennaCL<T>::VCLVectorBase vcl_vector =
			    a_gpu->data_vector(a.size());
			viennacl::linalg::vector_assign(vcl_vector, value);
		}

		/** ViennaCL softmax method */
		template <typename T, template <typename> class Container>
		void softmax_impl(Container<T>& a) const
		{
			GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);

			viennacl::ocl::kernel& kernel = generate_softmax_kernel<T>();
			kernel.global_work_size(
			    0,
			    linalg::implementation::ocl::align_to_multiple_1d(a.num_cols));

			viennacl::ocl::enqueue(
			    kernel(
			        a_gpu->data_matrix(a.num_rows, a.num_cols),
			        cl_int(a.num_rows), cl_int(a.num_cols),
			        cl_int(a_gpu->m_offset)));

			a.gpu_ptr = std::shared_ptr<GPUMemoryBase<T>>(
			    a_gpu->clone_vector(a_gpu, a.num_rows * a.num_cols));
		}

		/** ViennaCL squared error method
		 * The squared error is defined as \f$ E(P,Q) = \frac{1}{2} \sum_{ij}
		 * (P[i,j]-Q[i,j])^2 \f$
		 */
		template <typename T>
		T squared_error_impl(const SGMatrix<T>& p, const SGMatrix<T>& q) const
		{
			typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type
			    aligned_t;

			GPUMemoryViennaCL<T>* p_gpu = cast_to_viennacl(p);
			GPUMemoryViennaCL<T>* q_gpu = cast_to_viennacl(q);
			GPUMemoryViennaCL<T>* result_gpu = new GPUMemoryViennaCL<T>(1);

			viennacl::ocl::kernel& kernel = generate_squared_error_kernel<T>();
			viennacl::ocl::enqueue(
			    kernel(
			        p_gpu->data_matrix(p.num_rows, p.num_cols),
			        cl_int(p.num_rows * p.num_cols), cl_int(p_gpu->m_offset),
			        q_gpu->data_matrix(q.num_rows, q.num_cols),
			        cl_int(q_gpu->m_offset), result_gpu->data_vector(1)));

			T* result = reinterpret_cast<T*>(SG_MALLOC(aligned_t, 1));
			viennacl::backend::memory_read(
			    *(result_gpu->m_data), result_gpu->m_offset * sizeof(T),
			    sizeof(T), result);

			return result[0];
		}

		/** ViennaCL matrix sum method. */
		template <typename T>
		T sum_impl(const SGMatrix<T>& mat, bool no_diag = false) const
		{
			typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type
			    aligned_t;

			GPUMemoryViennaCL<T>* mat_gpu = cast_to_viennacl(mat);
			GPUMemoryViennaCL<T>* result_gpu = new GPUMemoryViennaCL<T>(1);

			viennacl::ocl::kernel& kernel = generate_sum_kernel<T>(no_diag);
			viennacl::ocl::enqueue(
			    kernel(
			        mat_gpu->data_matrix(mat.num_rows, mat.num_cols),
			        cl_int(mat.num_rows), cl_int(mat.num_cols),
			        cl_int(mat_gpu->m_offset), result_gpu->data_vector(1)));

			T* result;
			result = reinterpret_cast<T*>(SG_MALLOC(aligned_t, 1));
			viennacl::backend::memory_read(
			    *(result_gpu->m_data), result_gpu->m_offset * sizeof(T),
			    sizeof(T), result);

			return result[0];
		}

		/** ViennaCL vector sum method. */
		template <typename T>
		T sum_impl(const SGVector<T>& vec, bool no_diag = false) const
		{
#if VIENNACL_VERSION >= 10700
			GPUMemoryViennaCL<T>* vec_gpu = cast_to_viennacl(vec);
			return viennacl::linalg::sum(vec_gpu->data_vector(vec.size()));
#else
			return sum_impl(SGMatrix<T>(vec));
#endif
		}

		/** ViennaCL matrix sum method. */
		template <typename T>
		T sum_symmetric_impl(const SGMatrix<T>& mat, bool no_diag = false) const
		{
			return sum_impl(mat, no_diag);
		}

		/** ViennaCL matrix colwise sum method */
		template <typename T>
		SGVector<T> colwise_sum_impl(const SGMatrix<T>& mat, bool no_diag) const
		{
			GPUMemoryViennaCL<T>* mat_gpu = cast_to_viennacl(mat);
			GPUMemoryViennaCL<T>* result_gpu =
			    new GPUMemoryViennaCL<T>(mat.num_cols);
			viennacl::ocl::kernel& kernel =
			    generate_colwise_sum_kernel<T>(no_diag);
			kernel.global_work_size(
			    0, linalg::implementation::ocl::align_to_multiple_1d(
			           mat.num_cols));

			viennacl::ocl::enqueue(
			    kernel(
			        mat_gpu->data_matrix(mat.num_rows, mat.num_cols),
			        cl_int(mat.num_rows), cl_int(mat.num_cols),
			        cl_int(mat_gpu->m_offset),
			        result_gpu->data_vector(mat.num_cols),
			        cl_int(result_gpu->m_offset)));

			return SGVector<T>(result_gpu, mat.num_cols);
		}

		/** ViennaCL matrix rowwise sum method */
		template <typename T>
		SGVector<T> rowwise_sum_impl(const SGMatrix<T>& mat, bool no_diag) const
		{
			GPUMemoryViennaCL<T>* mat_gpu = cast_to_viennacl(mat);
			GPUMemoryViennaCL<T>* result_gpu =
			    new GPUMemoryViennaCL<T>(mat.num_rows);
			viennacl::ocl::kernel& kernel =
			    generate_rowwise_sum_kernel<T>(no_diag);
			kernel.global_work_size(
			    0, linalg::implementation::ocl::align_to_multiple_1d(
			           mat.num_rows));

			viennacl::ocl::enqueue(
			    kernel(
			        mat_gpu->data_matrix(mat.num_rows, mat.num_cols),
			        cl_int(mat.num_rows), cl_int(mat.num_cols),
			        cl_int(mat_gpu->m_offset),
			        result_gpu->data_vector(mat.num_rows),
			        cl_int(result_gpu->m_offset)));

			return SGVector<T>(result_gpu, mat.num_rows);
		}

		/** Transfer data to GPU with ViennaCL method. */
		template <typename T, template <typename> class Container>
		GPUMemoryBase<T>* to_gpu_impl(const Container<T>& a) const
		{
			GPUMemoryViennaCL<T>* gpu_ptr = new GPUMemoryViennaCL<T>();

			viennacl::backend::memory_create(
			    *(gpu_ptr->m_data), sizeof(T) * a.size(), viennacl::context());
			viennacl::backend::memory_write(
			    *(gpu_ptr->m_data), 0, a.size() * sizeof(T), a.data());

			return gpu_ptr;
		}

		/** Fetch data from GPU with ViennaCL method. */
		template <typename T, template <typename> class Container>
		void from_gpu_impl(const Container<T>& a, T* data) const
		{
			GPUMemoryViennaCL<T>* gpu_ptr = cast_to_viennacl(a);
			viennacl::backend::memory_read(
			    *(gpu_ptr->m_data), gpu_ptr->m_offset * sizeof(T),
			    a.size() * sizeof(T), data);
		}
	};
}

#else
// FIXME
namespace shogun
{
	class LinalgBackendViennaCL : public LinalgBackendBase {};
}
#endif // HAVE_VIENNACL

#endif // LINALG_BACKEND_VIENNACL_H__
