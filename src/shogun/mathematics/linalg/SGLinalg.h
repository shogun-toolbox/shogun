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

#ifndef SG_LINALG_H__
#define SG_LINALG_H__

#include <shogun/lib/config.h>

#include <shogun/lib/memory.h>
#include <shogun/lib/common.h>

#include <shogun/mathematics/linalg/LinalgBackendBase.h>
#include <shogun/mathematics/linalg/LinalgBackendEigen.h>

#include <memory>

namespace shogun
{

/** @brief linalg library backend */
class SGLinalg
{
public:
	/** Default constructor */
	SGLinalg()
	{
		cpu_backend = std::unique_ptr<LinalgBackendBase>(new LinalgBackendEigen());
		gpu_backend = nullptr;
	}

	/** Default destructor */
	~SGLinalg()
	{
	}

	/** Set CPU backend
	 * The default CPU backend is EIGEN3
	 */
	void set_cpu_backend(LinalgBackendBase* backend)
	{
		cpu_backend = std::unique_ptr<LinalgBackendBase>(backend);
	}

	/** Set CPU backend
	 *
	 * @return Pointer of LinalgBackendBase type
	 */
	LinalgBackendBase* const get_cpu_backend() const
	{
		return cpu_backend.get();
	}

	/** Set GPU backend
	 * The default GPU backend is NULL
	 */
	void set_gpu_backend(LinalgBackendBase* backend)
	{
		gpu_backend = std::unique_ptr<LinalgBackendBase>(backend);
	}

	/** Set GPU backend
	 *
	 * @return Pointer of LinalgBackendBase type
	 */
	LinalgBackendBase* const get_gpu_backend() const
	{
		return gpu_backend.get();
	}

private:
	/** Pointer to CPU backend. CPU backend is always available
	 * with EIGEN3 or other default/complete implementation.
	 */
	std::unique_ptr<LinalgBackendBase> cpu_backend;

	/** Pointer to GPU backend.
	 * NULL utill assigned.
	 */
	std::unique_ptr<LinalgBackendBase> gpu_backend;
};
}

namespace shogun
{
	/** Variable that holds the CPU and GPU backends. */
	extern std::unique_ptr<SGLinalg> sg_linalg;
}

#endif //SG_LINALG_H__
