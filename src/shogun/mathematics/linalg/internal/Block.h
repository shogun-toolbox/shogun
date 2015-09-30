/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * Written (w) 2014 Khaled Nasr
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef BLOCK_H_
#define BLOCK_H_

#include <shogun/lib/config.h>

namespace shogun
{

namespace linalg
{

/**
 * @brief Generic class Block which wraps a matrix class and contains block
 * specific information, providing a uniform way to deal with matrix blocks
 * for all supported backend matrices
 */
template <class Matrix>
struct Block
{
	/** scalar type */
	typedef typename Matrix::Scalar Scalar;
	
	/**
	 * constructor
	 *
	 * @param matrix the matrix on which the block is defined
	 * @param row_begin the row index at which the block starts
	 * @param col_begin the col index at which the block starts
	 * @param row_size the number of rows in the block
	 * @param col_size the number of cols in the block
	 *
	 * For example, row_begin 0, col_begin 4 and row_size 5, col_size 6
	 * represents the block that starts at index (0,4) in the matrix and
	 * goes upto (0+5-1,4+6-1) i.e. (4,9) both inclusive
	 */
	Block<Matrix>(Matrix matrix,
			index_t row_begin, index_t col_begin,
			index_t row_size, index_t col_size)
		: m_matrix(matrix), m_row_begin(row_begin), m_col_begin(col_begin),
		m_row_size(row_size), m_col_size(col_size)
	{
	}

	/** the matrix on which the block is defined */
	Matrix m_matrix;

	/** the row index at which the block starts */
	index_t m_row_begin;

	/** the col index at which the block starts */
	index_t m_col_begin;

	/** the number of rows in the block */
	index_t m_row_size;

	/** the number of cols in the block */
	index_t m_col_size;
};

/**
 * Method that returns a block object. Suited for Eigen3/SGMatrix
 *
 * @param matrix the matrix on which the block is defined
 * @param row_begin the row index at which the block starts
 * @param col_begin the col index at which the block starts
 * @param row_size the number of rows in the block
 * @param col_size the number of cols in the block
 * @return a block object on this matrix
 */
template <class Matrix>
Block<Matrix> block(Matrix matrix, index_t row_begin,
		index_t col_begin, index_t row_size, index_t col_size)
{
	return Block<Matrix>(matrix, row_begin, col_begin, row_size, col_size);
}

}

}
#endif // BLOCK_H_
