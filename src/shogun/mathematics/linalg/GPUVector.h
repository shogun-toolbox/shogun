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
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/BaseVector.h>
#include <memory>

#ifndef GPU_Vector_H__
#define GPU_Vector_H__

#ifdef HAVE_CXX11

namespace shogun
{

/** GPU vector wrapper structure */
template <class T>
struct GPUVector : public BaseVector<T>
{
	/** Structure for different types of GPU vectors */
	struct GPUArray;

	/** Opaque pointer of GPUArray */
	std::unique_ptr<GPUArray> gpuarray;

	/** Default Constructor */
	GPUVector();

	/** Wrap SGVector */
	GPUVector(const SGVector<T> &vector);

	/** Copy Constructor*/
	GPUVector(const GPUVector<T> &vector);

	/** Empty destructor */
	~GPUVector();

	/** needs to be overridden to initialize empty data */
	void init();

	/** Overload operator "=" */
	GPUVector<T>& operator=(const GPUVector<T> &other);

	/** Data Storage */
	inline bool onGPU()
	{
		return true;
	}

public:
	/** Vector length */
	index_t vlen;

	/** Offset for the memory segment, i.e the data of the vector
	 * starts at vector+offset
	 */
	index_t offset;
};

}

#endif //HAVE_CXX11

#endif
