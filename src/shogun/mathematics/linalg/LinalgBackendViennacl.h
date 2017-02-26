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

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/matrix_operations.hpp>
#include <viennacl/linalg/sum.hpp>
#include <shogun/mathematics/linalg/GPUMemoryViennaCL.h>

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
	#define DEFINE_FOR_ALL_PTYPE(METHODNAME, Container) \
	METHODNAME(char, Container); \
	METHODNAME(uint8_t, Container); \
	METHODNAME(int16_t, Container); \
	METHODNAME(uint16_t, Container); \
	METHODNAME(int32_t, Container); \
	METHODNAME(uint32_t, Container); \
	METHODNAME(float32_t, Container); \
	METHODNAME(float64_t, Container); \

	/** Implementation of @see LinalgBackendBase::add */
	#define BACKEND_GENERIC_IN_PLACE_ADD(Type, Container) \
	virtual void add(Container<Type>& a, Container<Type>& b, Type alpha, \
		Type beta, Container<Type>& result) const \
	{  \
		add_impl(a, b, alpha, beta, result); \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ADD, SGVector)
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ADD, SGMatrix)
	#undef BACKEND_GENERIC_ADD

	/** Implementation of @see LinalgBackendBase::dot */
	#define BACKEND_GENERIC_DOT(Type, Container) \
	virtual Type dot(const Container<Type>& a, const Container<Type>& b) const \
	{  \
		return dot_impl(a, b);  \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_DOT, SGVector)
	#undef BACKEND_GENERIC_DOT

	/** Implementation of @see LinalgBackendBase::element_prod */
	#define BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD(Type, Container) \
	virtual void element_prod(Container<Type>& a, Container<Type>& b,\
		Container<Type>& result) const \
	{  \
		element_prod_impl(a, b, result); \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD, SGMatrix)
	#undef BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD

	/** Implementation of @see LinalgBackendBase::matrix_prod */
	#define BACKEND_GENERIC_IN_PLACE_MATRIX_PROD(Type, Container) \
	virtual void matrix_prod(SGMatrix<Type>& a, Container<Type>& b,\
		Container<Type>& result, bool transpose_A, bool transpose_B) const \
	{  \
		matrix_prod_impl(a, b, result, transpose_A, transpose_B); \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_MATRIX_PROD, SGVector)
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_MATRIX_PROD, SGMatrix)
	#undef BACKEND_GENERIC_IN_PLACE_MATRIX_PROD

	/** Implementation of @see LinalgBackendBase::max */
	#define BACKEND_GENERIC_MAX(Type, Container) \
	virtual Type max(const Container<Type>& a) const \
	{  \
		return max_impl(a); \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_MAX, SGVector)
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_MAX, SGMatrix)
	#undef BACKEND_GENERIC_MAX

	/** Implementation of @see LinalgBackendBase::mean */
	#define BACKEND_GENERIC_MEAN(Type, Container) \
	virtual float64_t mean(const Container<Type>& a) const \
	{  \
		return mean_impl(a);  \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_MEAN, SGVector)
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_MEAN, SGMatrix)
	#undef BACKEND_GENERIC_MEAN

	/** Implementation of @see LinalgBackendBase::scale */
	#define BACKEND_GENERIC_IN_PLACE_SCALE(Type, Container) \
	virtual void scale(Container<Type>& a, Type alpha, Container<Type>& result) const \
	{  \
		scale_impl(a, result, alpha); \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_SCALE, SGVector)
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_SCALE, SGMatrix)
	#undef BACKEND_GENERIC_IN_PLACE_SCALE

	/** Implementation of @see LinalgBackendBase::set_const */
	#define BACKEND_GENERIC_SET_CONST(Type, Container) \
	virtual void set_const(Container<Type>& a, const Type value) const \
	{  \
		set_const_impl(a, value); \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SET_CONST, SGVector)
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SET_CONST, SGMatrix)
	#undef BACKEND_GENERIC_SET_CONST

	/** Implementation of @see LinalgBackendBase::sum */
	#define BACKEND_GENERIC_SUM(Type, Container) \
	virtual Type sum(const Container<Type>& a, bool no_diag) const \
	{  \
		return sum_impl(a, no_diag);  \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGVector)
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGMatrix)
	#undef BACKEND_GENERIC_SUM

	/** Implementation of @see LinalgBackendBase::sum_symmetric */
	#define BACKEND_GENERIC_SYMMETRIC_SUM(Type, Container) \
	virtual Type sum_symmetric(const Container<Type>& a, bool no_diag) const \
	{  \
		return sum_symmetric_impl(a, no_diag); \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SYMMETRIC_SUM, SGMatrix)
	#undef BACKEND_GENERIC_SYMMETRIC_SUM

	/** Implementation of @see LinalgBackendBase::colwise_sum */
	#define BACKEND_GENERIC_COLWISE_SUM(Type, Container) \
	virtual SGVector<Type> colwise_sum(const Container<Type>& a, bool no_diag) const \
	{  \
		return colwise_sum_impl(a, no_diag); \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_COLWISE_SUM, SGMatrix)
	#undef BACKEND_GENERIC_COLWISE_SUM

	/** Implementation of @see LinalgBackendBase::rowwise_sum */
	#define BACKEND_GENERIC_ROWWISE_SUM(Type, Container) \
	virtual SGVector<Type> rowwise_sum(const Container<Type>& a, bool no_diag) const \
	{  \
		return rowwise_sum_impl(a, no_diag); \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ROWWISE_SUM, SGMatrix)
	#undef BACKEND_GENERIC_ROWWISE_SUM

	/** Implementation of @see LinalgBackendBase::to_gpu */
	#define BACKEND_GENERIC_TO_GPU(Type, Container) \
	virtual GPUMemoryBase<Type>* to_gpu(const Container<Type>& a) const \
	{  \
		return to_gpu_impl(a);  \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_TO_GPU, SGVector)
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_TO_GPU, SGMatrix)
	#undef BACKEND_GENERIC_TO_GPU

	/** Implementation of @see LinalgBackendGPUBase::from_gpu */
	#define BACKEND_GENERIC_FROM_GPU(Type, Container) \
	virtual void from_gpu(const Container<Type>& a, Type* data) const \
	{  \
		return from_gpu_impl(a, data);  \
	}
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_FROM_GPU, SGVector)
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_FROM_GPU, SGMatrix)
	#undef BACKEND_GENERIC_FROM_GPU

	#undef DEFINE_FOR_ALL_PTYPE

private:
	/** Static cast @see GPUMemoryBase class to @see GPUMemoryViennaCL */
	template <typename T, template<typename> class Container>
	GPUMemoryViennaCL<T>* cast_to_viennacl(const Container<T> &a) const
	{
		return static_cast<GPUMemoryViennaCL<T>*>(a.gpu_ptr.get());
	}

	/** ViennaCL vector result = alpha * A + beta * B method */
	template <typename T>
	void add_impl(SGVector<T>& a, SGVector<T>& b, T alpha, T beta, SGVector<T>& result) const
	{
		GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
		GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);
		GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

		result_gpu->data_vector(a.size()) =
			alpha * a_gpu->data_vector(a.size()) + beta * b_gpu->data_vector(b.size());
	}

	/** ViennaCL matrix result = alpha * A + beta * B method */
	template <typename T>
	void add_impl(SGMatrix<T>& a, SGMatrix<T>& b, T alpha, T beta, SGMatrix<T>& result) const
	{
		GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
		GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);
		GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

		result_gpu->data_matrix(a.num_rows, a.num_cols) =
			alpha * a_gpu->data_matrix(a.num_rows, a.num_cols)
			+ beta * b_gpu->data_matrix(b.num_rows, b.num_cols);
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
	void element_prod_impl(SGMatrix<T>& a, SGMatrix<T>& b, SGMatrix<T>& result) const
	{
		GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
		GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);
		GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

		result_gpu->data_matrix(a.num_rows, a.num_cols) =
			viennacl::linalg::element_prod(a_gpu->data_matrix(a.num_rows,
				a.num_cols), b_gpu->data_matrix(a.num_rows, a.num_cols));
	}

	/** ViennaCL matrix * vector in-place product method */
	template <typename T>
	void matrix_prod_impl(SGMatrix<T>& a, SGVector<T>& b, SGVector<T>& result,
		bool transpose, bool transpose_B=false) const
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
				a_gpu->data_matrix(a.num_rows, a.num_cols), b_gpu->data_vector(b.vlen));
	}

	/** ViennaCL matrices in-place product method */
	template <typename T>
	void matrix_prod_impl(SGMatrix<T>& a, SGMatrix<T>& b, SGMatrix<T>& result,
		bool transpose_A, bool transpose_B) const
	{
		GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
		GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);
		GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

		if (transpose_A && transpose_B)
			result_gpu->data_matrix(result.num_rows, result.num_cols) =
				viennacl::linalg::prod(viennacl::trans(a_gpu->data_matrix(
				a.num_rows, a.num_cols)), viennacl::trans(b_gpu->data_matrix(
				b.num_rows, b.num_cols)));

		else if (transpose_A)
			result_gpu->data_matrix(result.num_rows, result.num_cols) =
				viennacl::linalg::prod(viennacl::trans(a_gpu->data_matrix(
				a.num_rows, a.num_cols)), b_gpu->data_matrix(b.num_rows,
				b.num_cols));

		else if (transpose_B)
			result_gpu->data_matrix(result.num_rows, result.num_cols) =
				viennacl::linalg::prod(a_gpu->data_matrix(a.num_rows, a.num_cols),
				viennacl::trans(b_gpu->data_matrix(b.num_rows, b.num_cols)));

		else
			result_gpu->data_matrix(result.num_rows, result.num_cols) =
				viennacl::linalg::prod(a_gpu->data_matrix(a.num_rows, a.num_cols),
				b_gpu->data_matrix(b.num_rows, b.num_cols));
	}

	/** ViennaCL max method */
	template <typename T, template<typename> class Container>
	T max_impl(const Container<T>& a) const
	{
		typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type aligned_t;

		GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
		GPUMemoryViennaCL<T>* result_gpu = new GPUMemoryViennaCL<T>(1);

		viennacl::ocl::kernel& kernel = generate_max_kernel<T>();
		viennacl::ocl::enqueue(kernel(a_gpu->data_vector(a.size()),
			cl_int(a.size()), cl_int(a_gpu->m_offset),
			result_gpu->data_vector(1)));

		T* result = reinterpret_cast<T*>(SG_MALLOC(aligned_t, 1));
		viennacl::backend::memory_read(*(result_gpu->m_data),
			result_gpu->m_offset*sizeof(T), sizeof(T), result);

		return result[0];
	}

	/** ViennaCL vectors or matrices mean method */
	template <typename T, template <typename> class Container>
	float64_t mean_impl(const Container<T>& a) const
	{
		return sum_impl(a)/float64_t(a.size());
	}

	/** ViennaCL vector inplace scale method: result = alpha * A */
	template <typename T>
	void scale_impl(SGVector<T>& a, SGVector<T>& result, T alpha) const
	{
		GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
		GPUMemoryViennaCL<T>* result_gpu = cast_to_viennacl(result);

		result_gpu->data_vector(a.size()) = alpha * a_gpu->data_vector(a.size());
	}

	/** ViennaCL vector inplace scale method: result = alpha * A */
	template <typename T>
	void scale_impl(SGMatrix<T>& a, SGMatrix<T>& result, T alpha) const
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

	/** ViennaCL vector sum method. */
	template <typename T>
	T sum_impl(const SGVector<T>& vec, bool no_diag=false) const
	{
		GPUMemoryViennaCL<T>* vec_gpu = cast_to_viennacl(vec);
		return viennacl::linalg::sum(vec_gpu->data_vector(vec.size()));
	}

	/** ViennaCL matrix sum method. */
	template <typename T>
	T sum_impl(const SGMatrix<T>& mat, bool no_diag=false) const
	{
		typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type aligned_t;

		GPUMemoryViennaCL<T>* mat_gpu = cast_to_viennacl(mat);
		GPUMemoryViennaCL<T>* result_gpu = new GPUMemoryViennaCL<T>(1);

		viennacl::ocl::kernel& kernel = generate_sum_kernel<T>(no_diag);
		viennacl::ocl::enqueue(kernel(mat_gpu->data_matrix(mat.num_rows, mat.num_cols),
			cl_int(mat.num_rows), cl_int(mat.num_cols), cl_int(mat_gpu->m_offset),
			result_gpu->data_vector(1)));

		T* result;
		result = reinterpret_cast<T*>(SG_MALLOC(aligned_t, 1));
		viennacl::backend::memory_read(*(result_gpu->m_data),
			result_gpu->m_offset*sizeof(T), sizeof(T), result);

		return result[0];
	}

	/** ViennaCL matrix sum method. */
	template <typename T>
	T sum_symmetric_impl(const SGMatrix<T>& mat, bool no_diag=false) const
	{
		return sum_impl(mat, no_diag);
	}

	/** ViennaCL matrix colwise sum method */
	template <typename T>
	SGVector<T> colwise_sum_impl(const SGMatrix<T>& mat, bool no_diag) const
	{
		GPUMemoryViennaCL<T>* mat_gpu = cast_to_viennacl(mat);
		GPUMemoryViennaCL<T>* result_gpu = new GPUMemoryViennaCL<T>(mat.num_cols);
		viennacl::ocl::kernel& kernel = generate_colwise_sum_kernel<T>(no_diag);
		kernel.global_work_size(0, linalg::implementation::ocl::align_to_multiple_1d(mat.num_cols));

		viennacl::ocl::enqueue(kernel(mat_gpu->data_matrix(mat.num_rows, mat.num_cols),
			cl_int(mat.num_rows), cl_int(mat.num_cols), cl_int(mat_gpu->m_offset),
			result_gpu->data_vector(mat.num_cols), cl_int(result_gpu->m_offset)));

		return SGVector<T>(result_gpu, mat.num_cols);
	}

	/** ViennaCL matrix rowwise sum method */
	template <typename T>
	SGVector<T> rowwise_sum_impl(const SGMatrix<T>& mat, bool no_diag) const
	{
		GPUMemoryViennaCL<T>* mat_gpu = cast_to_viennacl(mat);
		GPUMemoryViennaCL<T>* result_gpu = new GPUMemoryViennaCL<T>(mat.num_rows);
		viennacl::ocl::kernel& kernel = generate_rowwise_sum_kernel<T>(no_diag);
		kernel.global_work_size(0, linalg::implementation::ocl::align_to_multiple_1d(mat.num_rows));

		viennacl::ocl::enqueue(kernel(mat_gpu->data_matrix(mat.num_rows, mat.num_cols),
			cl_int(mat.num_rows), cl_int(mat.num_cols), cl_int(mat_gpu->m_offset),
			result_gpu->data_vector(mat.num_rows), cl_int(result_gpu->m_offset)));

		return SGVector<T>(result_gpu, mat.num_rows);
	}

	/** Transfer data to GPU with ViennaCL method. */
	template <typename T, template<typename> class Container>
	GPUMemoryBase<T>* to_gpu_impl(const Container<T>& a) const
	{
		GPUMemoryViennaCL<T>* gpu_ptr = new GPUMemoryViennaCL<T>();

		viennacl::backend::memory_create(*(gpu_ptr->m_data), sizeof(T)*a.size(),
			viennacl::context());
		viennacl::backend::memory_write(*(gpu_ptr->m_data), 0,
			a.size()*sizeof(T), a.data());

		return gpu_ptr;
	}

	/** Fetch data from GPU with ViennaCL method. */
	template <typename T, template<typename> class Container>
	void from_gpu_impl(const Container<T>& a, T* data) const
	{
		GPUMemoryViennaCL<T>* gpu_ptr = cast_to_viennacl(a);
		viennacl::backend::memory_read(*(gpu_ptr->m_data),
			gpu_ptr->m_offset*sizeof(T), a.size()*sizeof(T), data);
	}
};

}

#endif //HAVE_VIENNACL

#endif //LINALG_BACKEND_VIENNACL_H__
