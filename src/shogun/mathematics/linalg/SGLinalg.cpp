/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
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
 * Authors: 2016 Pan Deng, Soumyajit De, Viktor Gal
 */

#include <shogun/mathematics/linalg/SGLinalg.h>

#ifdef HAVE_CXX11

namespace shogun
{

SGLinalg::SGLinalg():m_cpubackend(std::shared_ptr<CPUBackend>(new CPUBackend)), m_gpubackend(nullptr)
{
}

SGLinalg::SGLinalg(std::shared_ptr<CPUBackend> cpubackend)
:m_cpubackend(cpubackend), m_gpubackend(nullptr)
{
}

SGLinalg::SGLinalg(std::shared_ptr<GPUBackend> gpubackend)
:m_cpubackend(std::shared_ptr<CPUBackend>(new CPUBackend)), m_gpubackend(gpubackend)
 {
 }

SGLinalg::SGLinalg(std::shared_ptr<CPUBackend> cpubackend, std::shared_ptr<GPUBackend> gpubackend)
:m_cpubackend(cpubackend), m_gpubackend(gpubackend)
 {
 }

void SGLinalg::set_cpu_backend(std::shared_ptr<CPUBackend> cpubackend)
{
	m_cpubackend = cpubackend;
}

std::shared_ptr<CPUBackend> SGLinalg::get_cpu_backend()
{
	return m_cpubackend;
}

void SGLinalg::set_gpu_backend(std::shared_ptr<GPUBackend> gpubackend)
{
	m_gpubackend = gpubackend;
}

std::shared_ptr<GPUBackend> SGLinalg::get_gpu_backend()
{
	return m_gpubackend;
}

template <class T>
T SGLinalg::dot(BaseVector<T> *a, BaseVector<T> *b) const
{
	if (a->onGPU() && b->onGPU())
	{
		if (this->hasGPUBackend())
		{
			// do the gpu backend dot product
			// you shouldn't care whether it's viennacl or some other GPU backend.
			return m_gpubackend->dot<T>(static_cast<GPUVector<T>&>(*a),
                        	                    static_cast<GPUVector<T>&>(*b));
		}
		else
		{
			SG_SERROR("User did not register GPU backend. \n");
 			return -1;
		}
	}
	else
	{
		// take care that the matricies are on the same backend
		if (a->onGPU())
		{
			SG_SERROR("User did not register GPU backend. \n");
			return -1;
		}
		else if (b->onGPU())
		{
			SG_SERROR("User did not register GPU backend. \n");
			return -1;
		}
		return m_cpubackend->dot<T>(static_cast<CPUVector<T>&>(*a),
                                        static_cast<CPUVector<T>&>(*b));
	}
}

bool SGLinalg::hasGPUBackend() const
{
	return m_gpubackend != nullptr;
}

template int32_t SGLinalg::dot<int32_t>(BaseVector<int32_t> *a, BaseVector<int32_t> *b) const;
template float32_t SGLinalg::dot<float32_t>(BaseVector<float32_t> *a, BaseVector<float32_t> *b) const;
}

#endif //HAVE_CXX11
