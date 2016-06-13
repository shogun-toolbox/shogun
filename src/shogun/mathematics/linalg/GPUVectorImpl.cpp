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

#include <shogun/mathematics/linalg/GPUVectorImpl.h>

#ifdef HAVE_CXX11
#ifdef HAVE_VIENNACL

namespace shogun
{

template<class T>
LinalgVector<T>::GPUVectorImpl::GPUVectorImpl()
{
}

template<class T>
LinalgVector<T>::GPUVectorImpl::GPUVectorImpl(T* data, index_t len)
:m_GPUptr(new VCLMemoryArray()), m_len(len), m_offset(0)
{

	{
		viennacl::backend::memory_create(*m_GPUptr, sizeof(T)*m_len,
	        	viennacl::context());

		viennacl::backend::memory_write(*m_GPUptr, 0, m_len*sizeof(T), data);
	}
}

template<class T>
LinalgVector<T>::GPUVectorImpl::GPUVectorImpl
(const LinalgVector<T>::GPUVectorImpl &array)
{
	m_GPUptr = array.m_GPUptr;
	m_len = array.m_len;
	m_offset = array.m_offset;
}

template<class T>
typename LinalgVector<T>::GPUVectorImpl::VCLVectorBase LinalgVector<T>::GPUVectorImpl::GPUvec()
{
	return VCLVectorBase(*m_GPUptr, m_len, m_offset, 1);
}

template<class T>
typename LinalgVector<T>::GPUVectorImpl::VCLVector LinalgVector<T>::GPUVectorImpl::vector()
{
	return VCLVector(LinalgVector<T>::GPUVectorImpl::GPUvec());
}

template class LinalgVector<int32_t>;
template class LinalgVector<float32_t>;

}

#endif //HAVE_VIENNACL
#endif //HAVE_CXX11
