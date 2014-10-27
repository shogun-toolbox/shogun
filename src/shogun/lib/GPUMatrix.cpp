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

#include <shogun/lib/GPUMatrix.h>
#include <viennacl/matrix.hpp>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif

#include <shogun/lib/SGMatrix.h>

namespace shogun
{

template <class T> 
CGPUMatrix<T>::CGPUMatrix()
{
	init();
}

template <class T> 
CGPUMatrix<T>::CGPUMatrix(index_t nrows, index_t ncols) : matrix(new VCLMemoryArray())
{
	init();
	
	num_rows = nrows;
	num_cols = ncols;
	
	viennacl::backend::memory_create(*matrix, sizeof(T)*num_rows*num_cols, 
		viennacl::context());
}

template <class T> 
CGPUMatrix<T>::CGPUMatrix(std::shared_ptr<VCLMemoryArray> mem, index_t nrows, index_t ncols, 
	index_t mem_offset)
{
	init();
	
	matrix = mem;
	num_rows = nrows;
	num_cols = ncols;
	offset = mem_offset;
}

template <class T> 
CGPUMatrix<T>::CGPUMatrix(const SGMatrix< T >& cpu_mat) : matrix(new VCLMemoryArray())
{
	init();
	
	num_rows = cpu_mat.num_rows;
	num_cols = cpu_mat.num_cols;

	viennacl::backend::memory_create(*matrix, sizeof(T)*num_rows*num_cols, 
		viennacl::context());
	
	viennacl::backend::memory_write(*matrix, 0, num_rows*num_cols*sizeof(T), 
		cpu_mat.matrix);
}

#ifdef HAVE_EIGEN3
template <class T> 
CGPUMatrix<T>::CGPUMatrix(const EigenMatrixXt& cpu_mat)
: matrix(new VCLMemoryArray())
{
	init();
	
	num_rows = cpu_mat.rows();
	num_cols = cpu_mat.cols();

	viennacl::backend::memory_create(*matrix, sizeof(T)*num_rows*num_cols, 
		viennacl::context());
	
	viennacl::backend::memory_write(*matrix, 0, num_rows*num_cols*sizeof(T), 
		cpu_mat.data());
}

template <class T> 
CGPUMatrix<T>::operator EigenMatrixXt() const
{
	EigenMatrixXt cpu_mat(num_rows, num_cols);
	
	viennacl::backend::memory_read(*matrix, offset*sizeof(T), num_rows*num_cols*sizeof(T), 
		cpu_mat.data());
	
	return cpu_mat;
}
#endif

template <class T> 
CGPUMatrix<T>::operator SGMatrix<T>() const
{
	SGMatrix<T> cpu_mat(num_rows, num_cols);
	
	viennacl::backend::memory_read(*matrix, offset*sizeof(T), num_rows*num_cols*sizeof(T), 
		cpu_mat.matrix);

	return cpu_mat;
}

template <class T> 
typename CGPUMatrix<T>::VCLMatrixBase CGPUMatrix<T>::vcl_matrix()
{
	return VCLMatrixBase(*matrix,num_rows, offset, 1, num_rows, num_cols, 0, 1, num_cols);
}

template <class T> 
void CGPUMatrix<T>::display_matrix(const char* name) const
{
	((SGMatrix<T>)*this).display_matrix(name);
}

template <class T> 
void CGPUMatrix<T>::zero()
{
	vcl_matrix().clear();
}

template <class T> 
void CGPUMatrix<T>::set_const(T value)
{
	VCLMatrixBase m = vcl_matrix();
	viennacl::linalg::matrix_assign(m, value);
}

template <class T> 
viennacl::const_entry_proxy<T> CGPUMatrix<T>::operator()(index_t i, index_t j) const
{
	return viennacl::const_entry_proxy<T>(offset+i+j*num_rows, *matrix);
}

template <class T> 
viennacl::entry_proxy< T > CGPUMatrix<T>::operator()(index_t i, index_t j)
{
	return viennacl::entry_proxy<T>(offset+i+j*num_rows, *matrix);
}

template <class T> 
viennacl::const_entry_proxy< T > CGPUMatrix<T>::operator[](index_t index) const
{
	return viennacl::const_entry_proxy<T>(offset+index, *matrix);
}

template <class T>
viennacl::entry_proxy< T > CGPUMatrix<T>::operator[](index_t index)
{
	return viennacl::entry_proxy<T>(offset+index, *matrix);
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

#endif // HAVE_CXX11
#endif // HAVE_VIENNACL
