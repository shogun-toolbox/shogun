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

#include <shogun/mathematics/linalg/Vector.h>
#include <shogun/mathematics/linalg/GPUVectorImpl.h>

#ifdef HAVE_CXX11

namespace shogun
{

template<class T>
Vector<T>::Vector()
{
	init();
}

template<class T>
Vector<T>::Vector(SGVector<T> const &vector)
{
	init();
	m_data = vector.vector;
	m_len = vector.vlen;
}

template<class T>
Vector<T>::Vector(Vector<T> const &vector)
{
	init();
	m_data = vector.m_data;
	m_len = vector.m_len;
#ifdef HAVE_VIENNACL
	if (vector.onGPU())
	{
		m_onGPU = true;
		m_gpu_impl = std::unique_ptr<GPUVectorImpl>(new GPUVectorImpl(*(vector.m_gpu_impl)));
	}
#endif
}

template<class T>
Vector<T>::~Vector()
{
}

template<class T>
Vector<T>& Vector<T>::operator=(SGVector<T> const &vector)
{
	m_data = vector.vector;
	m_len = vector.vlen;

	m_onGPU = false;
	m_gpu_impl.release();

	return *this;
}

template<class T>
Vector<T>& Vector<T>::operator=(Vector<T> const &vector)
{
	m_data = vector.m_data;
	m_len = vector.m_len;
	m_onGPU = false;
	m_gpu_impl.reset();

#ifdef HAVE_VIENNACL
	if (vector.onGPU())
	{
		m_onGPU = true;
		m_gpu_impl.reset(new GPUVectorImpl(*(vector.m_gpu_impl)));
	}
#endif
	return *this;
}

template<class T>
Vector<T>::operator SGVector<T>()
{
	if (onGPU())
		copy_to_CPU();
	return SGVector<T>(m_data, m_len);
}

template<class T>
bool Vector<T>::onGPU() const
{
	return m_onGPU;
}

template<class T>
T* Vector<T>::data()
{
	return m_data;
}

template<class T>
T const * Vector<T>::data() const
{
	return m_data;
}

template<class T>
index_t Vector<T>::size() const
{
	return m_len; // Assume GPU operations won't change the length
}

template<class T>
void Vector<T>::copy_to_GPU()
{
	if (m_data == nullptr)
	{
		SG_SINFO("There is no data to copy to GPU.\n")
		return;
	}

#ifdef HAVE_VIENNACL
	m_gpu_impl = std::unique_ptr<GPUVectorImpl>(new GPUVectorImpl(m_data, m_len));
	m_onGPU = true;
#else
	SG_SINFO("Transfer incomplete. No registered GPU backend found.\n");
#endif
}

template<class T>
void Vector<T>::move_to_GPU()
{
	if (m_data == nullptr)
	{
		SG_SINFO("There is no data to transfer to GPU.\n")
		return;
	}

#ifdef HAVE_VIENNACL
	m_gpu_impl = std::unique_ptr<GPUVectorImpl>(new GPUVectorImpl(m_data, m_len));
	m_onGPU = true;
	release_from_CPU();
#else
	SG_SINFO("Transfer incomplete. No registered GPU backend found.\n");
#endif
}

template<class T>
void Vector<T>::copy_to_CPU()
{
	if (!onGPU())
	{
		SG_SINFO("There is no data to copy on GPU.\n")
		return;
	}

#ifdef HAVE_VIENNACL
	if (m_data == nullptr)
	{
		m_data = SG_MALLOC(T, m_gpu_impl->size());
		m_len = m_gpu_impl->size();
	}
	m_gpu_impl->transferToCPU(m_data); // Assume the transfer won't change the length of the vector
#endif
}

template<class T>
void Vector<T>::move_to_CPU()
{
	if (!onGPU())
	{
		SG_SINFO("There is no data to transfer on GPU.\n")
		return;
	}

#ifdef HAVE_VIENNACL
	if (m_data == nullptr)
	{
		m_data = SG_MALLOC(T, m_gpu_impl->size());
		m_len = m_gpu_impl->size();
	}
	m_gpu_impl->transferToCPU(m_data); // Assume the transfer won't change the length of the vector
	release_from_GPU();
#endif
}

template<class T>
void Vector<T>::release_from_GPU()
{
	m_gpu_impl.reset();
	m_onGPU = false;
}

template<class T>
void Vector<T>::release_from_CPU()
{
	m_data = nullptr;
	m_len = 0;
}

template<class T>
void Vector<T>::init()
{
	m_data = nullptr;
	m_len = 0;
	m_onGPU = false;
}

template class Vector<int32_t>;
template class Vector<float32_t>;

}

#endif //HAVE_CXX11
