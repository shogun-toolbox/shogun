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

#ifndef __GPUMATRIX_H__
#define __GPUMATRIX_H__

#include <shogun/lib/config.h>

#ifdef HAVE_VIENNACL

#include <shogun/lib/SGMatrix.h>

#include <viennacl/matrix.hpp>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif

namespace shogun
{

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
	typedef viennacl::matrix_base<T, viennacl::column_major> VCLMatrixBase;
	typedef viennacl::backend::mem_handle VCLMemoryArray;
	
public:
	typedef T Scalar;
	
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
	CGPUMatrix(VCLMemoryArray mem, index_t nrows, index_t ncols, index_t mem_offset=0);
	
	/** Creates a gpu matrix using data from an SGMatrix */
	CGPUMatrix(const SGMatrix<T>& cpu_mat);

#ifdef HAVE_EIGEN3
	/** Creates a gpu matrix using data from an Eigen3 matrix */
	template <class Derived>
	CGPUMatrix(const Eigen::PlainObjectBase<Derived>& cpu_mat)
	{
		init();
		
		num_rows = cpu_mat.rows();
		num_cols = cpu_mat.cols();
		viennacl::backend::memory_create(matrix, sizeof(T)*num_rows*num_cols, 
			viennacl::context());
		
		viennacl::backend::memory_write(matrix, 0, num_rows*num_cols*sizeof(T), 
			cpu_mat.data());
	}
	
	/** Converts the matrix into an Eigen3 matrix */
	operator Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>() const;
#endif
	
	/** Converts the matrix into an SGMatrix */
	operator SGMatrix<T>() const;
	
	/** Returns a ViennaCL matrix wrapped around the data of this matrix. Can be 
	 * used to call native ViennaCL methods on this matrix
	 */
	VCLMatrixBase vcl_matrix()
	{
		return VCLMatrixBase(matrix,num_rows, offset, 1, num_rows, num_cols, 0, 1, num_cols);
	}
	
	/** Sets all the elements of the matrix to zero */
	void zero()
	{
		vcl_matrix().clear();
	}
	
	/** Sets all the elements of the matrix to a constant value 
	 * 
	 * @param value New value for all the elements in the matrix
	 */ 
	void set_const(T value)
	{
		VCLMatrixBase m = vcl_matrix();
		viennacl::linalg::matrix_assign(m, value);
	}
	
	/** Displays the matrix */
	void display_matrix(const char* name="matrix") const
	{
		((SGMatrix<T>)*this).display_matrix(name);
	}
	
	/** Read only memory access. Note that this is very slow as it copies the 
	 * element from the GPU to the CPU
	 * 
	 * @param i Row index
	 * @param j Column index
	 */ 
	inline viennacl::const_entry_proxy<T> operator()(index_t i, index_t j) const
	{
		return viennacl::const_entry_proxy<T>(offset+i+j*num_rows, matrix);
	}
	
	/** Read/write memory access. Note that this is very slow as it copies the 
	 * element between the GPU and the CPU
	 * 
	 * @param i Row index
	 * @param j Column index
	 */ 
	inline viennacl::entry_proxy<T> operator()(index_t i, index_t j)
	{
		return viennacl::entry_proxy<T>(offset+i+j*num_rows, matrix);
	}
	
	/** Read only memory access. Note that this is very slow as it copies the 
	 * element from the GPU to the CPU
	 * 
	 * @param index Array index
	 */ 
	inline viennacl::const_entry_proxy<T> operator[](index_t index) const
	{
		return viennacl::const_entry_proxy<T>(offset+index, matrix);
	}
	
	/** Read/write memory access. Note that this is very slow as it copies the 
	 * element between the GPU and the CPU
	 * 
	 * @param index Array index
	 */ 
	inline viennacl::entry_proxy<T> operator[](index_t index)
	{
		return viennacl::entry_proxy<T>(offset+index, matrix);
	}
	
private:
	void init();
	
public:
	/** Memory segment holding the data for the matrix */
	VCLMemoryArray matrix;
	
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

#endif
#endif
