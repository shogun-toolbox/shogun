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
#include <shogun/mathematics/linalg/GPUVector.h>
#include <memory>

#ifdef HAVE_VIENNACL
#include <viennacl/vector.hpp>
#endif

#ifndef GPU_ARRAY_H__
#define GPU_ARRAY_H__

#ifdef HAVE_CXX11

namespace shogun
{

/** GPU array structure */
template <class T>
struct GPUVector<T>::GPUArray
{

#ifdef HAVE_VIENNACL
	typedef viennacl::backend::mem_handle VCLMemoryArray;
	typedef viennacl::vector_base<T, std::size_t, std::ptrdiff_t> VCLVectorBase;
	typedef viennacl::vector<T> VCLVector;

	/** Memory segment holding the data for the vector */
	std::shared_ptr<VCLMemoryArray> GPUptr;

	/** Vector length */
	index_t vlen;

	/** Offset for the memory segment, i.e the data of the vector
	 * starts at vector+offset
	 */
	index_t offset;

	/** Creates a gpu vector with SGVector */
	GPUArray(const SGVector<T> &vector);

	/** Copy Constructor */
	GPUArray(const GPUVector<T>::GPUArray &array);

	/** Returns a ViennaCL vector wrapped around the data of this vector. Can be
	 * used to call native ViennaCL methods on this vector
	 */
	VCLVectorBase GPUvec();

	/** Cast of VCLVectorBase to VCLVector
	 * allows element access
	 */
	VCLVector vector();

#else //HAVE_VIENNACL
	/** Creates a gpu vector with SGVector */
	GPUArray(const SGVector<T> &vector);
#endif //HAVE_VIENNACL

};

}

#endif //HAVE_CXX11

#endif
