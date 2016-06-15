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
class Vector
{
	/** Memory aligned datatype */
	typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type aligned_t;

public:
	/** Default Constructor */
	Vector();

	/** Construct and copy from SGVector */
	Vector(SGVector<T> const &vector);

	/** Copy Constructor */
	Vector(Vector<T> const &vector);

	/** Destructor */
	~Vector();

	/** overload operator = */
	Vector& operator=(SGVector<T> const &vector);

	/** overload operator = */
	Vector& operator=(Vector<T> const &vector);

	/** Converts the vector into an SGVector */
	operator SGVector<T>();

	/** Data Storage */
	bool onGPU() const;

	/** Return vector data */
	T* data();

	/** Return vector data. Read only */
	T const * data() const;

	/** Return vector size. Read only */
	index_t size() const;

	/** Transfer vector from CPU to GPU */
	void transferToGPU();

	/** Transfer vector from GPU to CPU */
	void transferToCPU();

	/** Release vector from GPU */
	void releaseFromGPU();

	/** Operator overload for vector read only access
	 *
	 * @param index dimension to access
	 *
	 */
	T& operator[](index_t index);

	/** Operator overload for vector read only access
	 *
	 * @param index dimension to access
	 *
	 */
	const T& operator[](index_t index) const;

private:
	void init();

	/** Store the position of the data */
	alignas(CPU_CACHE_LINE_SIZE) bool m_onGPU;

	/** Vector length */
	alignas(CPU_CACHE_LINE_SIZE) index_t m_len;

	/** Vector data */
	alignas(CPU_CACHE_LINE_SIZE) std::shared_ptr<T> m_data;

	/** GPU Vector class */
	class GPUVectorImpl;

	/** Pointer to GPU Vector Class */
	alignas(CPU_CACHE_LINE_SIZE) std::unique_ptr<GPUVectorImpl> m_gpu_impl;
};
}

 #endif // HAVE_CXX11

 #endif
