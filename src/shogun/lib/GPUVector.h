/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Written (W) 2014 Khaled Nasr
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
 */

#ifndef __GPUVECTOR_H__
#define __GPUVECTOR_H__

#include <shogun/lib/config.h>

#ifdef HAVE_VIENNACL
#ifdef HAVE_CXX11

#include <shogun/lib/common.h>
#include <memory>


namespace viennacl
{
	template <class, class, class> class vector_base;
	template <class> class const_entry_proxy;
	template <class> class entry_proxy;

	namespace backend
	{
		class mem_handle;
	}
}

namespace Eigen
{
	template <class, int, int, int, int, int> class Matrix;
}

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
	typedef viennacl::vector_base<T, std::size_t, std::ptrdiff_t> VCLVectorBase;
	typedef viennacl::backend::mem_handle VCLMemoryArray;

	typedef Eigen::Matrix<T,-1,1,0,-1,1> EigenVectorXt;
	typedef Eigen::Matrix<T,1,-1,0x1,1,-1> EigenRowVectorXt;

public:
	/** The scalar type of the vector */
	typedef T Scalar;

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
	CGPUVector(std::shared_ptr<VCLMemoryArray> mem, index_t length, index_t mem_offset=0);

	/** Creates a gpu vector using data from an SGVector */
	CGPUVector(const SGVector<T>& cpu_vec);

#ifndef SWIG // SWIG should skip this part
#ifdef HAVE_EIGEN3
	/** Creates a gpu vector using data from an Eigen3 column vector */
	CGPUVector(const EigenVectorXt& cpu_vec);

	/** Creates a gpu vector using data from an Eigen3 row vector */
	CGPUVector(const EigenRowVectorXt& cpu_vec);

	/** Converts the vector into an Eigen3 column vector */
	operator EigenVectorXt() const;

	/** Converts the vector into an Eigen3 row vector */
	operator EigenRowVectorXt() const;
#endif
#endif

	/** Converts the vector into an SGVector */
	operator SGVector<T>() const;

	/** Returns a ViennaCL vector wrapped around the data of this vector. Can be
	 * used to call native ViennaCL methods on this vector
	 */
	VCLVectorBase vcl_vector();

	/** Sets all the elements of the vector to zero */
	void zero();

	/** Sets all the elements of the vector to a constant value
	 *
	 * @param value New value for all the elements in the vector
	 */
	void set_const(T value);

	/** Displays the vector */
	void display_vector(const char* name="vector") const;

	/** Read only memory access. Note that this is very slow as it copies the
	 * element from the GPU to the CPU
	 *
	 * @param index Element index
	 */
	viennacl::const_entry_proxy<T> operator[](index_t index) const;

	/** Read/write memory access. Note that this is very slow as it copies the
	 * element between the GPU and the CPU
	 *
	 * @param index Element index
	 */
	viennacl::entry_proxy<T> operator[](index_t index);

private:
	void init();

public:
	/** Memory segment holding the data for the vector */
	std::shared_ptr<VCLMemoryArray> vector;

	/** Offset for the memory segment, i.e the data of the vector
	 * starts at vector+offset
	 */
	index_t offset;

	/** Vector length */
	index_t vlen;
};

}

#endif // HAVE_CXX11
#endif // HAVE_VIENNACL
#endif // __GPUVECTOR_H__
