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

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/GPUMemoryBase.h>
#include <memory>

#ifndef LINALG_BACKEND_GPU_BASE_H__
#define LINALG_BACKEND_GPU_BASE_H__

namespace shogun
{

/** @brief Base interface of generic GPU linalg methods
 * and generic GPU memory transfer methods.
 */
class LinalgBackendGPUBase : public LinalgBackendBase
{
public:
	/**
	 * Wrapper method of Transferring data to GPU memory.
	 *
	 * @see LinalgBackendBase::to_gpu
	 */
	#define BACKEND_GENERIC_TO_GPU(Type) \
	virtual GPUMemoryBase<Type>* to_gpu(const SGVector<Type>&) const = 0;\

	BACKEND_GENERIC_TO_GPU(char);
	BACKEND_GENERIC_TO_GPU(uint8_t);
	BACKEND_GENERIC_TO_GPU(int16_t);
	BACKEND_GENERIC_TO_GPU(uint16_t);
	BACKEND_GENERIC_TO_GPU(int32_t);
	BACKEND_GENERIC_TO_GPU(uint32_t);
	BACKEND_GENERIC_TO_GPU(float32_t);
	BACKEND_GENERIC_TO_GPU(float64_t);
	#undef BACKEND_GENERIC_TO_GPU

	/**
	 * Wrapper method of fetching data from GPU memory.
	 *
	 * @see LinalgBackendBase::from_gpu
	 */
	#define BACKEND_GENERIC_FROM_GPU(Type) \
	virtual void from_gpu(const SGVector<Type>&, Type* data) const = 0;\

	BACKEND_GENERIC_FROM_GPU(char);
	BACKEND_GENERIC_FROM_GPU(uint8_t);
	BACKEND_GENERIC_FROM_GPU(int16_t);
	BACKEND_GENERIC_FROM_GPU(uint16_t);
	BACKEND_GENERIC_FROM_GPU(int32_t);
	BACKEND_GENERIC_FROM_GPU(uint32_t);
	BACKEND_GENERIC_FROM_GPU(float32_t);
	BACKEND_GENERIC_FROM_GPU(float64_t);
	#undef BACKEND_GENERIC_FROM_GPU

};

}

#endif //LINALG_BACKEND_GPU_BASE_H__
