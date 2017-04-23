/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
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
 * Authors: 2016 Pan Deng, Soumyajit De, Heiko Strathmann, Viktor Gal
 */

#ifndef GPU_MEMORY_VIENNACL_H__
#define GPU_MEMORY_VIENNACL_H__

#include <shogun/lib/common.h>

#ifdef HAVE_VIENNACL
#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <memory>

namespace shogun
{

/** @brief ViennaCL memory structure.
 * Saves data to GPU and clone data.
 * @see SGVector
 */
template <typename T>
struct GPUMemoryViennaCL : public GPUMemoryBase<T>
{
	friend class LinalgBackendViennaCL;

	typedef viennacl::backend::mem_handle VCLMemoryArray;
	typedef viennacl::vector_base<T, std::size_t, std::ptrdiff_t> VCLVectorBase;

/** @see <a href="http://viennacl.sourceforge.net/doc/changelog.html"> */
#if VIENNACL_VERSION >= 10600
	typedef viennacl::matrix_base<T, std::size_t, std::ptrdiff_t> VCLMatrixBase;
#else
	typedef viennacl::matrix_base<T, viennacl::column_major, std::size_t, std::ptrdiff_t> VCLMatrixBase;
#endif

	/** Default constructor */
	GPUMemoryViennaCL() : m_data(new VCLMemoryArray())
	{
		init();
	};

	/** Create a new vector
	 *
	 * @param len Number of elements
	 */
	GPUMemoryViennaCL(index_t len): m_data(new VCLMemoryArray())
	{
		init();
		viennacl::backend::memory_create(*m_data, sizeof(T)*len,
			viennacl::context());
	}

	/** Wrap a vector around an existing memory segment
	 *
	 * @param gpu_ptr GPUMemoryBase pointer
	 */
	GPUMemoryViennaCL(GPUMemoryBase<T>* gpu_ptr) : m_data(new VCLMemoryArray())
	{
		GPUMemoryViennaCL<T>* temp_ptr = static_cast<GPUMemoryViennaCL<T>*>(gpu_ptr);
		init();
		m_data = temp_ptr->m_data;
		m_offset = temp_ptr->m_offset;
	};

	/** Clone GPU vector
	 *
	 * @param vector GPUMemoryBase pointer
	 * @param vlen Length of the vector
	 */
	GPUMemoryBase<T>* clone_vector(GPUMemoryBase<T>* vector, index_t vlen) const
	{
		GPUMemoryViennaCL<T>* src_ptr = static_cast<GPUMemoryViennaCL<T>*>(vector);
		GPUMemoryViennaCL<T>* gpu_ptr = new GPUMemoryViennaCL<T>();

		viennacl::backend::memory_create(*(gpu_ptr->m_data), sizeof(T)*vlen,
			viennacl::context());
		viennacl::backend::memory_copy(*(src_ptr->m_data), *(gpu_ptr->m_data),
			0, 0, vlen*sizeof(T));

		return gpu_ptr;
	}

	/** ViennaCL Vector structure that saves the data
	 *
	 * @param len Number of elements
	 */
	VCLVectorBase data_vector(index_t len)
	{
		return VCLVectorBase(*m_data, len, m_offset, 1);
	}

	/** ViennaCL Vector structure that saves the data
	 *
	 * @param nrows Row number of the matrix
	 * @param ncols Column number of the matrix
	 */
	VCLMatrixBase data_matrix(index_t nrows, index_t ncols)
	{
	#if VIENNACL_VERSION >= 10600
		return VCLMatrixBase(*m_data, nrows, m_offset, 1, nrows, ncols, 0, 1, ncols, false);
	#else
		return VCLMatrixBase(*m_data, nrows, m_offset, 1, nrows, ncols, 0, 1, ncols);
	#endif
	}

private:
	void init()
	{
		m_offset = 0;
	}

	/** Memory segment holding the data for the vector */
	alignas(CPU_CACHE_LINE_SIZE) std::shared_ptr<VCLMemoryArray> m_data;

	/** Offset for the memory segment, i.e the data of the vector
	 * starts at vector+offset
	 */
	alignas(CPU_CACHE_LINE_SIZE) index_t m_offset;
};

}
#endif // HAVE_VIENNACL

#endif //GPU_MEMORY_VIENNACL_H__
