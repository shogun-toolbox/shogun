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

#include <shogun/mathematics/linalg/LinalgBackendGPUBase.h>

#ifndef LINALG_BACKEND_VIENNACL_H__
#define LINALG_BACKEND_VIENNACL_H__

#ifdef HAVE_VIENNACL

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>
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
	/** Implementation of @see LinalgBackendBase::add */
	#define BACKEND_GENERIC_ADD(Type) \
	virtual SGVector<Type> add(const SGVector<Type>& a, const SGVector<Type>& b, Type alpha, Type beta) const \
	{ \
		return add_impl(a, b, alpha, beta); \
	}

	BACKEND_GENERIC_ADD(char);
	BACKEND_GENERIC_ADD(uint8_t);
	BACKEND_GENERIC_ADD(int16_t);
	BACKEND_GENERIC_ADD(uint16_t);
	BACKEND_GENERIC_ADD(int32_t);
	BACKEND_GENERIC_ADD(uint32_t);
	BACKEND_GENERIC_ADD(float32_t);
	BACKEND_GENERIC_ADD(float64_t);
	#undef BACKEND_GENERIC_ADD

	/** Implementation of @see LinalgBackendBase::dot */
	#define BACKEND_GENERIC_DOT(Type) \
	virtual Type dot(const SGVector<Type>& a, const SGVector<Type>& b) const \
	{  \
		return dot_impl(a, b);  \
	}

	BACKEND_GENERIC_DOT(char);
	BACKEND_GENERIC_DOT(uint8_t);
	BACKEND_GENERIC_DOT(int16_t);
	BACKEND_GENERIC_DOT(uint16_t);
	BACKEND_GENERIC_DOT(int32_t);
	BACKEND_GENERIC_DOT(uint32_t);
	BACKEND_GENERIC_DOT(float32_t);
	BACKEND_GENERIC_DOT(float64_t);
	#undef BACKEND_GENERIC_DOT

	/** Implementation of @see LinalgBackendBase::to_gpu */
	#define BACKEND_GENERIC_TO_GPU(Type) \
	virtual GPUMemoryBase<Type>* to_gpu(const SGVector<Type>& vector) const \
	{  \
		return to_gpu_impl(vector);  \
	}

	BACKEND_GENERIC_TO_GPU(char);
	BACKEND_GENERIC_TO_GPU(uint8_t);
	BACKEND_GENERIC_TO_GPU(int16_t);
	BACKEND_GENERIC_TO_GPU(uint16_t);
	BACKEND_GENERIC_TO_GPU(int32_t);
	BACKEND_GENERIC_TO_GPU(uint32_t);
	BACKEND_GENERIC_TO_GPU(float32_t);
	BACKEND_GENERIC_TO_GPU(float64_t);
	#undef BACKEND_GENERIC_TO_GPU

	/** Implementation of @see LinalgBackendGPUBase::from_gpu */
	#define BACKEND_GENERIC_FROM_GPU(Type) \
	virtual void from_gpu(const SGVector<Type>& vector, Type* data) const \
	{  \
		return from_gpu_impl(vector, data);  \
	}

	BACKEND_GENERIC_FROM_GPU(char);
	BACKEND_GENERIC_FROM_GPU(uint8_t);
	BACKEND_GENERIC_FROM_GPU(int16_t);
	BACKEND_GENERIC_FROM_GPU(uint16_t);
	BACKEND_GENERIC_FROM_GPU(int32_t);
	BACKEND_GENERIC_FROM_GPU(uint32_t);
	BACKEND_GENERIC_FROM_GPU(float32_t);
	BACKEND_GENERIC_FROM_GPU(float64_t);
	#undef BACKEND_GENERIC_FROM_GPU

private:
	/** ViennaCL vector C = alpha*A + beta*B method */
	template <typename T>
	SGVector<T> add_impl(const SGVector<T>& a, const SGVector<T>& b, T alpha, T beta) const
	{
		GPUMemoryViennaCL<T>* a_gpu = static_cast<GPUMemoryViennaCL<T>*>(a.gpu_vector.get());
		GPUMemoryViennaCL<T>* b_gpu = static_cast<GPUMemoryViennaCL<T>*>(b.gpu_vector.get());
		GPUMemoryViennaCL<T>* c_gpu = new GPUMemoryViennaCL<T>(a.size());

		c_gpu->data(a.size()) = alpha * a_gpu->data(a.size()) + beta * b_gpu->data(b.size());
		return SGVector<T>(c_gpu, a.size());
	}

	/** ViennaCL vector dot-product method. */
	template <typename T>
	T dot_impl(const SGVector<T>& a, const SGVector<T>& b) const
	{
		GPUMemoryViennaCL<T>* a_gpu = static_cast<GPUMemoryViennaCL<T>*>(a.gpu_vector.get());
		GPUMemoryViennaCL<T>* b_gpu = static_cast<GPUMemoryViennaCL<T>*>(b.gpu_vector.get());

		return viennacl::linalg::inner_prod(a_gpu->data(a.size()), b_gpu->data(b.size()));
	}

	/** Transfers data to GPU with ViennaCL method. */
	template <typename T>
	GPUMemoryBase<T>* to_gpu_impl(const SGVector<T>& vector) const \
	{
		GPUMemoryViennaCL<T>* gpu_vec;
		gpu_vec = new GPUMemoryViennaCL<T>();

		viennacl::backend::memory_create(*(gpu_vec->m_data), sizeof(T)*vector.size(),
				viennacl::context());
		viennacl::backend::memory_write(*(gpu_vec->m_data), 0,
				vector.size()*sizeof(T), vector.data());

		return gpu_vec;
	}

	/** Fetches data from GPU with ViennaCL method. */
	template <typename T>
	void from_gpu_impl(const SGVector<T>& vector, T* data) const \
	{
		GPUMemoryViennaCL<T>* gpu_vec = static_cast<GPUMemoryViennaCL<T>*>(vector.gpu_vector.get());
		viennacl::backend::memory_read(*(gpu_vec->m_data),
			gpu_vec->m_offset*sizeof(T), vector.size()*sizeof(T), data);
	}
};

}

#endif //HAVE_VIENNACL

#endif //LINALG_BACKEND_VIENNACL_H__
