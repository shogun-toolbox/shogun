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

#ifndef __GPUVECTOR_H__
#define __GPUVECTOR_H__

#include <shogun/lib/config.h>

#ifdef HAVE_VIENNACL

#include <shogun/lib/SGVector.h>

#include <viennacl/vector.hpp>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif

namespace shogun
{

/** @brief Represents a vector on the GPU 
 * 
 * This class handles vectors on the GPU using [ViennaCL](http://viennacl.sourceforge.net/)
 * as backend for managing GPU memory.
 * 
 * It supports conversion to/from SGVector objects and Eigen3 vectors. Native 
 * ViennaCL methods can also be used on the vector's data through vcl_vector()
 * 
 * Supported scalar types: char, uint8_t, int16_t, uint16_t, int32_t, 
 * uint32_t, int64_t, uint64_t, float32_t, float64_t.
 */
template <class T> class CGPUVector
{
	typedef viennacl::vector_base<T> VCLVectorBase;
	typedef viennacl::backend::mem_handle VCLMemoryArray;
	
public:
	/** Default Constructor */ 
	CGPUVector();
	
	/** Creates a new vector
	 * 
	 * @param length Number of elements
	 */
	CGPUVector(index_t length);
	
	/** Wraps a vector around an existing memery segment
	 * 
	 * @param mem A memory segment 
	 * @param length Number of elements
	 * @param mem_offset Offset for the memory segment, i.e the data of the vector
	 * starts at mem+mem_offset
	 */
	CGPUVector(VCLMemoryArray mem, index_t length, index_t mem_offset=0);
	
	/** Creates a gpu vector using data from an SGVector */
	CGPUVector(const SGVector<T>& cpu_vec);

#ifdef HAVE_EIGEN3
	/** Creates a gpu vector using data from an Eigen3 vector */
	template <class Derived>
	CGPUVector(const Eigen::PlainObjectBase<Derived>& cpu_vec)
	{
		init();
		vlen = cpu_vec.size();
		
		viennacl::backend::memory_create(vector, sizeof(T)*vlen, 
			viennacl::context());
		
		viennacl::backend::memory_write(vector, 0, vlen*sizeof(T), 
			cpu_vec.data());
	}
	
	/** Converts the vector into an Eigen3 column vector */
	operator Eigen::Matrix<T, Eigen::Dynamic, 1>() const;
	
	/** Converts the vector into an Eigen3 row vector */
	operator Eigen::Matrix<T, 1, Eigen::Dynamic>() const;
#endif
	
	/** Converts the vector into an SGVector */
	operator SGVector<T>() const;
	
	/** Returns a ViennaCL vector wrapped around the data of this vector. Can be 
	 * used to call native ViennaCL methods on this vector
	 */
	VCLVectorBase vcl_vector()
	{
		return VCLVectorBase(vector,vlen, offset, 1);
	}
	
	/** Sets all the elements of the vector to zero */
	void zero()
	{
		vcl_vector().clear();
	}
	
	/** Sets all the elements of the vector to a constant value 
	 * 
	 * @param value New value for all the elements in the vector
	 */ 
	void set_const(T value)
	{
		VCLVectorBase v = vcl_vector();
		viennacl::linalg::vector_assign(v, value);
	}
	
	/** Displays the vector */
	void display_vector(const char* name="vector") const
	{
		((SGVector<T>)*this).display_vector(name);
	}
	
	/** Read only memory access. Note that this is very slow as it copies the 
	 * element from the GPU to the CPU
	 * 
	 * @param index Element index
	 */ 
	inline viennacl::const_entry_proxy<T> operator[](index_t index) const
	{
		return viennacl::const_entry_proxy<T>(offset+index, vector);
	}
	
	/** Read/write memory access. Note that this is very slow as it copies the 
	 * element between the GPU and the CPU
	 * 
	 * @param index Element index
	 */ 
	inline viennacl::entry_proxy<T> operator[](index_t index)
	{
		return viennacl::entry_proxy<T>(offset+index, vector);
	}
	
private:
	void init();
	
public:
	/** Memory segment holding the data for the vector */
	VCLMemoryArray vector;
	
	/** Offset for the memory segment, i.e the data of the vector
	 * starts at vector+offset
	 */
	index_t offset;
	
	/** Vector length */
	index_t vlen;
};

}

#endif
#endif
