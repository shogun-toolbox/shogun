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
#ifdef HAVE_CXX11

#include <shogun/lib/GPUVector.h>
#include <viennacl/vector.hpp>

#include <shogun/lib/SGVector.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif

namespace shogun
{

template <class T> 
CGPUVector<T>::CGPUVector()
{
	init();
}

template <class T> 
CGPUVector<T>::CGPUVector(index_t length) : vector(new VCLMemoryArray())
{
	init();
	
	vlen = length;
	
	viennacl::backend::memory_create(*vector, sizeof(T)*vlen, 
		viennacl::context());
}

template <class T> 
CGPUVector<T>::CGPUVector(std::shared_ptr<VCLMemoryArray> mem, index_t length, 
	index_t mem_offset)
{
	init();
	
	vector = mem;
	vlen = length;
	offset = mem_offset;
}

template <class T> 
CGPUVector<T>::CGPUVector(const SGVector<T>& cpu_vec) : vector(new VCLMemoryArray())
{
	init();
	vlen = cpu_vec.vlen;

	viennacl::backend::memory_create(*vector, sizeof(T)*vlen, 
		viennacl::context());
	
	viennacl::backend::memory_write(*vector, 0, vlen*sizeof(T), 
		cpu_vec.vector);
}

#ifdef HAVE_EIGEN3
template <class T> 
CGPUVector<T>::CGPUVector(const EigenVectorXt& cpu_vec)
: vector(new VCLMemoryArray())
{
	init();
	vlen = cpu_vec.size();

	viennacl::backend::memory_create(*vector, sizeof(T)*vlen, 
		viennacl::context());
	
	viennacl::backend::memory_write(*vector, 0, vlen*sizeof(T), 
		cpu_vec.data());
}

template <class T> 
CGPUVector<T>::CGPUVector(const EigenRowVectorXt& cpu_vec)
: vector(new VCLMemoryArray())
{
	init();
	vlen = cpu_vec.size();
	
	viennacl::backend::memory_create(*vector, sizeof(T)*vlen, 
		viennacl::context());
	
	viennacl::backend::memory_write(*vector, 0, vlen*sizeof(T), 
		cpu_vec.data());
}

template <class T> 
CGPUVector<T>::operator EigenVectorXt() const
{
	EigenVectorXt cpu_vec(vlen);
	
	viennacl::backend::memory_read(*vector, offset*sizeof(T), vlen*sizeof(T), 
		cpu_vec.data());

	return cpu_vec;
}

template <class T> 
CGPUVector<T>::operator EigenRowVectorXt() const
{
	EigenRowVectorXt cpu_vec(vlen);
	
	viennacl::backend::memory_read(*vector, offset*sizeof(T), vlen*sizeof(T), 
		cpu_vec.data());

	return cpu_vec;
}
#endif

template <class T> 
CGPUVector<T>::operator SGVector<T>() const
{
	SGVector<T> cpu_vec(vlen);
	
	viennacl::backend::memory_read(*vector, offset*sizeof(T), vlen*sizeof(T), 
		cpu_vec.vector);

	return cpu_vec;
}

template <class T> 
typename CGPUVector<T>::VCLVectorBase CGPUVector<T>::vcl_vector()
{
	return VCLVectorBase(*vector,vlen, offset, 1);
}

template <class T> 
void CGPUVector<T>::display_vector(const char* name) const
{
	((SGVector<T>)*this).display_vector(name);
}

template <class T> 
void CGPUVector<T>::zero()
{
	vcl_vector().clear();
}

template <class T> 
void CGPUVector<T>::set_const(T value)
{
	VCLVectorBase v = vcl_vector();
	viennacl::linalg::vector_assign(v, value);
}

template <class T> 
viennacl::const_entry_proxy< T > CGPUVector<T>::operator[](index_t index) const
{
	return viennacl::const_entry_proxy<T>(offset+index, *vector);
}

template <class T>
viennacl::entry_proxy< T > CGPUVector<T>::operator[](index_t index)
{
	return viennacl::entry_proxy<T>(offset+index, *vector);
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

#endif // HAVE_CXX11
#endif // HAVE_VIENNACL
