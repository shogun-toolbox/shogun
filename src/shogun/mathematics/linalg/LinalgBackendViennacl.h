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
	#define BACKEND_GENERIC_ADD(Type, Container) \
	virtual Container<Type> add(const Container<Type>& a, const Container<Type>& b, Type alpha, Type beta) const \
	{  \
		return add_impl(a, b, alpha, beta); \
	}

	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ADD, SGVector)
	#undef BACKEND_GENERIC_ADD

	/** Implementation of @see LinalgBackendBase::dot */
	#define BACKEND_GENERIC_DOT(Type, Container) \
	virtual Type dot(const Container<Type>& a, const Container<Type>& b) const \
	{  \
		return dot_impl(a, b);  \
	}

	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_DOT, SGVector)
	#undef BACKEND_GENERIC_DOT

	/** Implementation of @see LinalgBackendBase::sum */
	#define BACKEND_GENERIC_SUM(Type, Container) \
	virtual Type sum(const Container<Type>& a) const \
	{  \
		return sum_impl(a);  \
	}

	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGVector)
	DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGMatrix)
	#undef BACKEND_GENERIC_SUM

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
	/** static cast GPUMemoryBase class to GPUMemoryViennaCL */
	template <typename T, template<typename> class Container>
	GPUMemoryViennaCL<T>* cast_to_viennacl(const Container<T> &a) const
	{
		return static_cast<GPUMemoryViennaCL<T>*>(a.gpu_ptr.get());
	}

	/** ViennaCL vector C = alpha*A + beta*B method */
	template <typename T>
	SGVector<T> add_impl(const SGVector<T>& a, const SGVector<T>& b, T alpha, T beta) const
	{
		GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
		GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);
		GPUMemoryViennaCL<T>* c_gpu = new GPUMemoryViennaCL<T>(a.size());

		c_gpu->data(a.size()) = alpha * a_gpu->data(a.size()) + beta * b_gpu->data(b.size());
		return SGVector<T>(c_gpu, a.size());
	}

	/** ViennaCL vector dot-product method. */
	template <typename T>
	T dot_impl(const SGVector<T>& a, const SGVector<T>& b) const
	{
		GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
		GPUMemoryViennaCL<T>* b_gpu = cast_to_viennacl(b);

		return viennacl::linalg::inner_prod(a_gpu->data(a.size()), b_gpu->data(b.size()));
	}

	/** ViennaCL sum method. */
	template <typename T, template <typename> class Container>
	T sum_impl(const Container<T>& a) const
	{
		GPUMemoryViennaCL<T>* a_gpu = cast_to_viennacl(a);
		return viennacl::linalg::sum(a_gpu->data(a.size()));
	}

	/** Transfers data to GPU with ViennaCL method. */
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

	/** Fetches data from GPU with ViennaCL method. */
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
