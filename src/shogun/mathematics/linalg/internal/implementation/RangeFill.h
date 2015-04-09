/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Kunal Arora
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

#ifndef RANGE_FILL_IMPL_H_
#define RANGE_FILL_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

#include <numeric>

namespace shogun
{

namespace linalg
{

namespace implementation
{

/**
 * @brief Generic class which is specialized for different backends to perform addition
 */
template <enum Backend, class Matrix>
struct range_fill
{
	/**Scalar type */
	typedef typename Matrix::Scalar T;

	/**Range fill a vector or a matrix with start...start+len-1
     * @param A - the matrix to be filled
     * @param len - length of the matrix to be filled
	 * @param start - value to be assigned to first element of vector or matrix
	 */	
	static void compute(Matrix A, T start);

};

/**
 *@brief Partial specialization of add for the Native backend
 */
template <class Matrix>
struct range_fill<Backend::NATIVE, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Range fill a matrix with start...start+len-1
     * @param A - the matrix to be filled
     * @param len - length of the matrix to be filled
	 * @param start - value to be assigned to first element of vector or matrix
	 */	
	static void compute(SGMatrix<T> A, T start)
	{
		compute(A.matrix, A.num_rows*A.num_cols, start);
	}

	/** Range fill a vector with start...start+len-1
     * @param A - the matrix to be filled
     * @param len - length of the matrix to be filled
	 * @param start - value to be assigned to first element of vector or matrix
	 */	
	static void compute(SGVector<T> A, T start)
	{
		compute(A.vector, A.vlen, start);
	}

	/**Range fill a vector or a matrix with start...start+len-1
     * @param A - the matrix to be filled
     * @param len - length of the matrix to be filled
	 * @param start - value to be assigned to first element of vector or matrix
	 */	
	static void compute(T* A, index_t len, T start)
	{
		std::iota(A, A+len, start);
	}

};

/**
 * @brief Generic class which is specialized for different backends to perform addition
 */
template <enum Backend, class Vector>
struct range_fill_vec
{
	static void compute(Vector* A, index_t len, Vector start);
};

/**
 *@brief Partial specialization of add for the Native backend
 */
template <class Vector>
struct range_fill_vec<Backend::NATIVE, Vector>
{
	static void compute(Vector* A, index_t len, Vector start)
	{
		std::iota(A, A+len, start);
	}	
};

}

}

}
#endif //RANGE_FILL_IMPL_H_
