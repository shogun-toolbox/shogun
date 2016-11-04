/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Kunal Arora
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
 * @brief Generic class which is specialized for different backends to perform the Range fill operation
 */
template <enum Backend, class Matrix>
struct range_fill
{
	/**Scalar type */
	typedef typename Matrix::Scalar T;

	/**Range fill a vector with start...start+len-1
	 * @param A - the matrix to be filled
	 * @param start - value to be assigned to first element of vector
	 */
	static void compute(Matrix A, T start);

};

/**
 * @brief Partial specialization of add for the Eigen3 backend
 */
template <class Matrix>
struct range_fill<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Eigen3 vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/**Range fill a vector with start...start+len-1
	 * @param A - the vector to be filled
	 * @param start - value to be assigned to first element of vector or matrix
	 */
	static void compute(SGVector<T> A, T start)
	{
		compute(A, A.size(), start);
	}

	/** Range fill a vector array with start...start+len-1
	 * @param A - the array to be filled
	 * @param len - length of the array to be filled
	 * @param start - value to be assigned to first element of array
	 */
	static void compute(SGVector<T> A, index_t len, T start)
	{
		Eigen::Map<VectorXt> A_eig=A;
		A_eig.setLinSpaced(len, start, A.size()+start-1);
	}

};

}

}

}
#endif //RANGE_FILL_IMPL_H_
