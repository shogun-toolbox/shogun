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

#ifndef __GPUMATRIX_H__
#define __GPUMATRIX_H__

#include <shogun/lib/config.h>

#ifdef HAVE_VIENNACL
#ifdef HAVE_CXX11

#include <shogun/lib/common.h>
#include <memory>

#ifndef SWIG // SWIG should skip this part
namespace viennacl
{
	template <class, class, class, class> class matrix_base;
	template <class> class const_entry_proxy;
	template <class> class entry_proxy;
	class column_major;

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

template <class> class SGMatrix;

/** @brief Represents a column-major matrix on the GPU
 *
 * This class handles matrices on the GPU using [ViennaCL](http://viennacl.sourceforge.net/)
 * as backend for managing GPU memory.
 *
 * It supports conversion to/from SGMatrix objects and Eigen3 matrices. Native
 * ViennaCL methods can also be used on the data of the matrix through vcl_matrix()
 *
 * Supported scalar types: char, uint8_t, int16_t, uint16_t, int32_t,
 * uint32_t, int64_t, uint64_t, float32_t, float64_t.
 */
template <class T> class CGPUMatrix
{

	typedef viennacl::matrix_base<T, viennacl::column_major, std::size_t, std::ptrdiff_t> VCLMatrixBase;
	typedef viennacl::backend::mem_handle VCLMemoryArray;

	typedef Eigen::Matrix<T,-1,-1,0,-1,-1> EigenMatrixXt;

public:
	/** The scalar type of the matrix */
	typedef T Scalar;

	/** The container type for a given template argument */
	template <typename ST> using container_type = CGPUMatrix<ST>;

	/** Default Constructor */
	CGPUMatrix();

	/** Creates a new matrix
	 *
	 * @param nrows Number of rows
	 * @param ncols Number of columns
	 */
	CGPUMatrix(index_t nrows, index_t ncols);

	/** Wraps a matrix around an existing memery segment
	 *
	 * @param mem A memory segment
	 * @param nrows Number of rows
	 * @param ncols Number of columns
	 * @param mem_offset Offset for the memory segment, i.e the data of the matrix
	 * starts at mem+mem_offset
	 */
	CGPUMatrix(std::shared_ptr<VCLMemoryArray> mem, index_t nrows, index_t ncols,
		index_t mem_offset=0);

	/** Creates a gpu matrix using data from an SGMatrix */
	CGPUMatrix(const SGMatrix<T>& cpu_mat);

	/** Converts the matrix into an SGMatrix */
	operator SGMatrix<T>() const;

	/** Creates a gpu matrix using data from an Eigen3 matrix */
	CGPUMatrix(const EigenMatrixXt& cpu_mat);

	/** Converts the matrix into an Eigen3 matrix */
	operator EigenMatrixXt() const;

	/** The data */
	inline VCLMatrixBase data()
	{
		return vcl_matrix();
	}

	/** The size */
	inline uint64_t size() const
	{
		const uint64_t c=num_cols;
		return num_rows*c;
	}

	/** Returns a ViennaCL matrix wrapped around the data of this matrix. Can be
	 * used to call native ViennaCL methods on this matrix
	 */
	VCLMatrixBase vcl_matrix();

	/** Sets all the elements of the matrix to zero */
	void zero();

	/** Sets all the elements of the matrix to a constant value
	 *
	 * @param value New value for all the elements in the matrix
	 */
	void set_const(T value);

	/** Displays the matrix */
	void display_matrix(const char* name="matrix") const;

	/** Read only memory access. Note that this is very slow as it copies the
	 * element from the GPU to the CPU
	 *
	 * @param i Row index
	 * @param j Column index
	 */
	viennacl::const_entry_proxy<T> operator()(index_t i, index_t j) const;

	/** Read/write memory access. Note that this is very slow as it copies the
	 * element between the GPU and the CPU
	 *
	 * @param i Row index
	 * @param j Column index
	 */
	viennacl::entry_proxy<T> operator()(index_t i, index_t j);

	/** Read only memory access. Note that this is very slow as it copies the
	 * element from the GPU to the CPU
	 *
	 * @param index Array index
	 */
	viennacl::const_entry_proxy<T> operator[](index_t index) const;

	/** Read/write memory access. Note that this is very slow as it copies the
	 * element between the GPU and the CPU
	 *
	 * @param index Array index
	 */
	viennacl::entry_proxy<T> operator[](index_t index);

private:
	void init();

public:
	/** Memory segment holding the data for the matrix */
	std::shared_ptr<VCLMemoryArray> matrix;

	/** Offset for the memory segment, i.e the data of the matrix
	 * starts at matrix+offset
	 */
	index_t offset;

	/** Number of rows */
	index_t num_rows;

	/** Number of columns */
	index_t num_cols;
};

}
#endif // SWIG

#endif // HAVE_CXX11
#endif // HAVE_VIENNACL
#endif // __GPUMATRIX_H__
