/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its 
 * contributors may be used to endorse or promote products derived from this 
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * Written (W) 2014 Khaled Nasr
 */

#include <shogun/lib/config.h>

#ifdef HAVE_VIENNACL

#include <shogun/lib/GPUVector.h>

namespace shogun
{

template <class T> 
CGPUVector<T>::CGPUVector()
{
	init();
}

template <class T> 
CGPUVector<T>::CGPUVector(index_t length)
{
	init();
	
	vlen = length;
	
	viennacl::backend::memory_create(vector, sizeof(T)*vlen, 
		viennacl::context());
}

template <class T> 
CGPUVector<T>::CGPUVector(VCLMemoryArray mem, index_t length, index_t mem_offset)
{
	init();
	
	vector = mem;
	vlen = length;
	offset = mem_offset;
}

template <class T> 
CGPUVector<T>::CGPUVector(const SGVector<T>& cpu_vec)
{
	init();
	vlen = cpu_vec.vlen;
	
	viennacl::backend::memory_create(vector, sizeof(T)*vlen, 
		viennacl::context());
	
	viennacl::backend::memory_write(vector, 0, vlen*sizeof(T), 
		cpu_vec.vector);
}

#ifdef HAVE_EIGEN3
template <class T> 
CGPUVector<T>::operator Eigen::Matrix<T, Eigen::Dynamic, 1>() const
{
	Eigen::Matrix<T, Eigen::Dynamic, 1> cpu_vec(vlen);
	
	viennacl::backend::memory_read(vector, offset*sizeof(T), vlen*sizeof(T), 
		cpu_vec.data());

	return cpu_vec;
}

template <class T> 
CGPUVector<T>::operator Eigen::Matrix<T, 1, Eigen::Dynamic>() const
{
	Eigen::Matrix<T, 1, Eigen::Dynamic> cpu_vec(vlen);
	
	viennacl::backend::memory_read(vector, offset*sizeof(T), vlen*sizeof(T), 
		cpu_vec.data());

	return cpu_vec;
}
#endif

template <class T> 
CGPUVector<T>::operator SGVector<T>() const
{
	SGVector<T> cpu_vec(vlen);
	
	viennacl::backend::memory_read(vector, offset*sizeof(T), vlen*sizeof(T), 
		cpu_vec.vector);

	return cpu_vec;
}

template <class T> 
void CGPUVector<T>::init()
{
	vlen = 0;
	offset = 0;
}

template class CGPUVector<char>;
template class CGPUVector<uint8_t>;
template class CGPUVector<int16_t>;
template class CGPUVector<uint16_t>;
template class CGPUVector<int32_t>;
template class CGPUVector<uint32_t>;
template class CGPUVector<int64_t>;
template class CGPUVector<uint64_t>;
template class CGPUVector<float32_t>;
template class CGPUVector<float64_t>;
}

#endif
