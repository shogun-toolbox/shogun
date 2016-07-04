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

#include <shogun/mathematics/linalg/LinalgBackendBase.h>
#include <shogun/mathematics/linalg/SGLinalg.h>

namespace shogun
{

namespace linalg
{

/** Return the corresponding backend of SGVector
 * @param vec SGVector
 * @return LinalgBackendBase pointer
 */
template <typename Type>
LinalgBackendBase* infer_backend(const SGVector<Type>& vec)
{
	if (vec.on_gpu())
		return sg_linalg->get_gpu_backend();
	else
		return sg_linalg->get_cpu_backend();
}

/** Return the corresponding backend of SGVector
 * Raise error if the two SGVectors are not on the same backend.
 * @param a the fisrt SGVector
 * @param b the second SGVector
 * @return LinalgBackendBase pointer
 */
template <typename Type>
LinalgBackendBase* infer_backend(const SGVector<Type>& a, const SGVector<Type>& b)
{
	if (a.on_gpu() && b.on_gpu())
		return sg_linalg->get_gpu_backend();
	else if (a.on_gpu() || b.on_gpu())
		SG_SERROR("Cannot operate with first vector gpu (%d) and second vector gpu (%d).\n",
					a.on_gpu(), b.on_gpu());

	return sg_linalg->get_cpu_backend();
}

/**
 * Vector dot-product that works with generic vectors.
 *
 * @param a first vector
 * @param b second vector
 * @return the dot product of \f$\mathbf{a}\f$ and \f$\mathbf{b}\f$, represented
 * as \f$\sum_i a_i b_i\f$
 */
template <typename Type>
Type dot(const SGVector<Type>& a, const SGVector<Type>& b)
{
	return infer_backend(a, b)->dot(a, b);
}

/**
 * Transfers data to GPU memory. Does nothing if no GPU backend registered.
 *
 * @param vector SGVector to be transferred
 * @return SGVector with vector on GPU if GPU backend is available
 * and a shallow-copy of SGVector with vector on CPU if GPU backend not available
 */
template <typename T>
SGVector<T> to_gpu(const SGVector<T>& vector)
{
	REQUIRE(!vector.on_gpu(), "The vector is already on GPU.\n");
	LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
	if (gpu_backend)
		return SGVector<T>(gpu_backend->to_gpu(vector), vector.vlen);
	else
	{
		SG_SWARNING("Trying to access GPU memory without GPU backend registered.\n");
		return vector;
	}
}

/**
 * Fetches data from GPU memory.
 *
 * @param vector SGVector to be transferred
 * @return SGVector with vector on CPU if GPU backend is still available
 * and a shallow-copy of SGVector with vector on GPU if GPU backend not available
 */
template <typename T>
SGVector<T> from_gpu(const SGVector<T>& vector)
{
	LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
	if (gpu_backend)
	{
		typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type aligned_t;
		T* data;
		data = reinterpret_cast<T*>(SG_MALLOC(aligned_t, vector.vlen));
		gpu_backend->from_gpu(vector, data);
		return SGVector<T>(data, vector.vlen);
	}
	else
	{
		SG_SWARNING("Trying to run GPU code without GPU backend registered.\n");
		return vector;
	}
}

}

}
