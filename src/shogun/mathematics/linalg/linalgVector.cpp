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

#include <shogun/mathematics/linalg/linalgVector.h>
#include <shogun/mathematics/linalg/GPUVectorImpl.h>

#ifdef HAVE_CXX11

namespace shogun
{

template<class T>
LinalgVector<T>::LinalgVector()
{
	init();
}

template<class T>
LinalgVector<T>::LinalgVector(SGVector<T> const &vector)
{
	init();
	m_data = reinterpret_cast<T*>(SG_MALLOC(aligned_t, vector.vlen));
	std::copy(vector.vector, vector.vector+vector.vlen, m_data);
	m_len = vector.vlen;
}

template<class T>
LinalgVector<T>::LinalgVector(LinalgVector<T> const &vector)
{
	init();
	m_data = vector.m_data;
	m_len = vector.m_len;
	m_onGPU = vector.m_onGPU;

	if (vector.onGPU())
	{
		m_onGPU = true;
		m_gpu_impl = std::unique_ptr<GPUVectorImpl>(new GPUVectorImpl(*(vector.m_gpu_impl)));
	}
}

template<class T>
LinalgVector<T>::~LinalgVector()
{
	free(m_data);
}

template<class T>
LinalgVector<T>& LinalgVector<T>::operator=(SGVector<T> const &vector)
{
	m_data = reinterpret_cast<T*>(SG_MALLOC(aligned_t, vector.vlen));
	std::copy(vector.vector, vector.vector+vector.vlen, m_data);
	m_len = vector.vlen;
	m_onGPU = false;
	m_gpu_impl.release();
	return *this;
}

template<class T>
LinalgVector<T>& LinalgVector<T>::operator=(LinalgVector<T> const &vector)
{

	m_data = vector.m_data;
	m_len = vector.m_len;
	m_onGPU = vector.m_onGPU;
	if (vector.onGPU())
	{
		m_gpu_impl.reset(new GPUVectorImpl(*(vector.m_gpu_impl)));
	}
	return *this;

}

template<class T>
LinalgVector<T>::operator SGVector<T>() const
{
	return SGVector<T>(m_data, m_len);
}

template<class T>
bool LinalgVector<T>::onGPU() const
{
	return m_onGPU;
}

template<class T>
T* LinalgVector<T>::data()
{
	return m_data;
}

template<class T>
T const * LinalgVector<T>::data() const
{
	return m_data;
}

template<class T>
index_t LinalgVector<T>::size() const
{
	return m_len;
}

template<class T>
void LinalgVector<T>::transferToGPU()
{
#ifdef HAVE_VIENNACL
	m_gpu_impl = std::unique_ptr<GPUVectorImpl>(new GPUVectorImpl(m_data, m_len));
	m_onGPU = true;
#else
	SG_SERROR("User did not register GPU backend. \n");
#endif
}

template<class T>
void LinalgVector<T>::transferToCPU()
{
	if (m_gpu_impl != nullptr)
	{
		m_gpu_impl.release();
	}
	m_onGPU = false;
}

template<class T>
void LinalgVector<T>::init()
{
	m_data = nullptr;
	m_len = 0;
	m_onGPU = false;
}

template class LinalgVector<int32_t>;
template class LinalgVector<float32_t>;

}

#endif //HAVE_CXX11
