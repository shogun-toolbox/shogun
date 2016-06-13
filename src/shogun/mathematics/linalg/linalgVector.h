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
#include <shogun/lib/SGVector.h>
#include <memory>

#ifndef LINALG_VECTOR_H__
#define LINALG_VECTOR_H__

#ifdef HAVE_CXX11

namespace shogun
{

 /** Vector structure
  * for both CPU and GPU vector
  */
template <class T>
class LinalgVector
{
private:
	class GPUVectorImpl;
	std::unique_ptr<GPUVectorImpl> m_gpu_impl;

	void init();

public:
	/** Default Constructor */
	LinalgVector();

	LinalgVector(SGVector<T> const &vector);

	LinalgVector(LinalgVector<T> const &vector);

	~LinalgVector();

	LinalgVector& operator=(SGVector<T> const &vector);

	LinalgVector& operator=(LinalgVector<T> const &vector);

	operator SGVector<T>() const;

	/** Data Storage
	 * @return whether data is on GPU
	*/
	bool onGPU() const;

	T* data();

	T const * data() const;

	index_t size() const;

	void transferToGPU();

	void transferToCPU();

private:
	bool m_onGPU;
	index_t m_len;  // same logic
	T* m_data;      // non-owning ptr, referring to the SGVector.vector
};
}

 #endif // HAVE_CXX11

 #endif
