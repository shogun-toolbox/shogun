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

#include <shogun/lib/GPUMatrix.h>

namespace shogun
{

template <class T> 
CGPUMatrix<T>::CGPUMatrix()
{
	init();
}

template <class T> 
CGPUMatrix<T>::CGPUMatrix(index_t nrows, index_t ncols)
{
	init();
	
	num_rows = nrows;
	num_cols = ncols;
	viennacl::backend::memory_create(matrix, sizeof(T)*num_rows*num_cols, 
		viennacl::context());
}

template <class T> 
CGPUMatrix<T>::CGPUMatrix(VCLMemoryArray mem, index_t nrows, index_t ncols, 
	index_t mem_offset)
{
	init();
	
	matrix = mem;
	num_rows = nrows;
	num_cols = ncols;
	offset = mem_offset;
}

template <class T> 
CGPUMatrix<T>::CGPUMatrix(const SGMatrix< T >& cpu_mat)
{
	init();
	
	num_rows = cpu_mat.num_rows;
	num_cols = cpu_mat.num_cols;
	viennacl::backend::memory_create(matrix, sizeof(T)*num_rows*num_cols, 
		viennacl::context());
	
	viennacl::backend::memory_write(matrix, 0, num_rows*num_cols*sizeof(T), 
		cpu_mat.matrix);
}

#ifdef HAVE_EIGEN3
template <class T> 
CGPUMatrix<T>::operator Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>() const
{
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> cpu_mat(num_rows, num_cols);
	
	viennacl::backend::memory_read(matrix, offset*sizeof(T), num_rows*num_cols*sizeof(T), 
		cpu_mat.data());

	return cpu_mat;
}
#endif

template <class T> 
CGPUMatrix<T>::operator SGMatrix<T>() const
{
	SGMatrix<T> cpu_mat(num_rows, num_cols);
	
	viennacl::backend::memory_read(matrix, offset*sizeof(T), num_rows*num_cols*sizeof(T), 
		cpu_mat.matrix);

	return cpu_mat;
}

template <class T> 
void CGPUMatrix<T>::init()
{
	num_rows = 0;
	num_cols = 0;
	offset = 0;
}

template class CGPUMatrix<char>;
template class CGPUMatrix<uint8_t>;
template class CGPUMatrix<int16_t>;
template class CGPUMatrix<uint16_t>;
template class CGPUMatrix<int32_t>;
template class CGPUMatrix<uint32_t>;
template class CGPUMatrix<int64_t>;
template class CGPUMatrix<uint64_t>;
template class CGPUMatrix<float32_t>;
template class CGPUMatrix<float64_t>;
}

#endif
