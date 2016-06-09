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

#include <shogun/mathematics/linalg/GPUArray.h>

#ifdef HAVE_CXX11

namespace shogun
{

#ifdef HAVE_VIENNACL

template <class T>
GPUVector<T>::GPUArray::GPUArray(const SGVector<T> &vector)
:GPUptr(new VCLMemoryArray()), vlen(vector.vlen), offset(0)
{
	viennacl::backend::memory_create(*GPUptr, sizeof(T)*vlen,
        	viennacl::context());

	viennacl::backend::memory_write(*GPUptr, 0, vlen*sizeof(T),
        	vector.vector);
}

template <class T>
GPUVector<T>::GPUArray::GPUArray(const GPUVector<T>::GPUArray &array)
{
	GPUptr = array.GPUptr;
	vlen = array.vlen;
	offset = array.offset;
}

template <class T>
typename GPUVector<T>::GPUArray::VCLVectorBase GPUVector<T>::GPUArray::GPUvec()
{
	 return VCLVectorBase(*GPUptr, vlen, offset, 1);
}

template <class T>
typename GPUVector<T>::GPUArray::VCLVector GPUVector<T>::GPUArray::vector()
{
	return VCLVector(GPUVector<T>::GPUArray::GPUvec());
}

#else // HAVE_VIENNACL

template <class T>
GPUVector<T>::GPUArray::GPUArray(const SGVector<T> &vector)
{
	SG_SERROR("User did not register GPU backend. \n");
}

#endif //HAVE_VIENNACL

template struct GPUVector<int32_t>;
template struct GPUVector<float32_t>;

}

#endif //HAVE_CXX11
