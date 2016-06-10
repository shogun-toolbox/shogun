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

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/CPUBackend.h>
#include <shogun/mathematics/linalg/GPUBackend.h>
#include <shogun/mathematics/linalg/CPUVector.h>
#include <shogun/mathematics/linalg/GPUVector.h>
#include <memory>

#ifndef LINALGR_H__
#define LINALGR_H__

#ifdef HAVE_CXX11

namespace shogun
{

/** Linalg Class **/
class SGLinalg
{
	/** pointer to cpubackend */
	std::shared_ptr<CPUBackend> m_cpubackend;

	/** pointer to gpubackend */
	std::shared_ptr<GPUBackend> m_gpubackend;

public:
	/** Default Constructor */
	SGLinalg();

	/** Constructor: explicitly set CPU Backend */
	SGLinalg(std::shared_ptr<CPUBackend> cpubackend);

	/** Constructor: explicitly set GPU Backend */
	SGLinalg(std::shared_ptr<GPUBackend> gpubackend);

	/** Constructor: explicitly set CPU and GPU Backend */
	SGLinalg(std::shared_ptr<CPUBackend> cpubackend, std::shared_ptr<GPUBackend> gpubackend);

	/** set CPU backend
	 * @param cpubackend cpubackend
	 */
	void set_cpu_backend(std::shared_ptr<CPUBackend> cpubackend);

	/** get CPU backend */
	std::shared_ptr<CPUBackend> get_cpu_backend();

	/** set GPU backend
	 * @param gpubackend gpubackend
	 */
	void set_gpu_backend(std::shared_ptr<GPUBackend> gpubackend);

	/** get GPU backend */
	std::shared_ptr<GPUBackend> get_gpu_backend();


	/**
	 * Wrapper method of implementation of vector dot-product that works
	 * with generic vectors
	 *
	 * @param a first vector
	 * @param b second vector
	 * @return the dot product of \f$\mathbf{a}\f$ and \f$\mathbf{b}\f$, represented
	 * as \f$\sum_i a_i b_i\f$
	 */
	template <class T>
	T dot(BaseVector<T> *a, BaseVector<T> *b) const;

	/**
	 * Method that computes the sum of a vector
	 *
	 * @param vec a vector whose sum has to be computed
	 * @return the vector sum \f$\sum_i a_i\f$
	 */
	template <class T>
	T sum(BaseVector<T> *vec) const;

	/** Check whether gpubackend is registered by user */
	bool hasGPUBackend() const;

};

}

namespace shogun
{
	extern std::unique_ptr<SGLinalg> sg_linalg;
}

#endif //HAVE_CXX11

#endif
