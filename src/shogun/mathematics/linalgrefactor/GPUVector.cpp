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

#include <shogun/mathematics/linalgrefactor/GPUVector.h>
#include <shogun/mathematics/linalgrefactor/GPUArray.h>

namespace shogun
{

template <class T>
GPUVector<T>::GPUVector()
{
	init();
}

template <class T>
GPUVector<T>::GPUVector(const SGVector<T> &vector)
{
	init();
	vlen = vector.vlen;

#ifdef HAVE_VIENNACL
	gpuarray = std::unique_ptr<GPUArray>(new GPUArray(vector));
#else
	SG_SERROR("User did not register GPU backend. \n");
#endif
}

template <class T>
GPUVector<T>::GPUVector(const GPUVector<T> &vector)
{
	init();
	vlen = vector.vlen;
	offset = vector.offset;
#ifdef HAVE_VIENNACL
	gpuarray = std::unique_ptr<GPUArray>(new GPUArray(*(vector.gpuarray)));
#else
	SG_SERROR("User did not register GPU backend. \n");
#endif
}

template <class T>
void GPUVector<T>::init()
{
	vlen = 0;
	offset = 0;
}

template <class T>
GPUVector<T>& GPUVector<T>::operator=(const GPUVector<T> &other)
{
	// check for self-assignment
	if(&other == this)
	{
		return *this;
	}

	// reuse storage when possible
	gpuarray.reset(new GPUArray(*(other.gpuarray)));
	vlen = other.vlen;
	return *this;
}

template <class T>
GPUVector<T>::~GPUVector()
{
}

template struct GPUVector<int32_t>;
template struct GPUVector<float32_t>;

}
