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

#ifndef REDUX_H_
#define REDUX_H_

#include <shogun/mathematics/linalg/internal/implementation/Dot.h>
#include <shogun/mathematics/linalg/internal/implementation/Sum.h>
#include <shogun/mathematics/linalg/internal/implementation/VectorSum.h>
#include <shogun/mathematics/linalg/internal/implementation/Max.h>

namespace shogun
{

namespace linalg
{

/**
 * Wrapper method for internal implementation of vector dot-product that works
 * with generic vectors.
 *
 * @param a first vector
 * @param b second vector
 * @return the dot product of \f$\mathbf{a}\f$ and \f$\mathbf{b}\f$, represented
 * as \f$\sum_i a_i b_i\f$
 */
template <Backend backend=linalg_traits<Redux>::backend, class Vector>
typename Vector::Scalar dot(Vector a, Vector b)
{
	return implementation::dot<backend,Vector>::compute(a, b);
}

/**
 * Wrapper method for internal implementation of matrix sum of values that works
 * with generic dense matrices
 * 
 * @param m the matrix whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
 */
template <Backend backend=linalg_traits<Redux>::backend, class Matrix>
typename Matrix::Scalar sum(Matrix m, bool no_diag=false)
{
	return implementation::sum<backend,Matrix>::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of symmetric matrix sum of values that works
 * with generic dense matrices 
 * 
 * @param m the matrix whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
 */
template <Backend backend=linalg_traits<Redux>::backend,class Matrix>
typename Matrix::Scalar sum_symmetric(Matrix m, bool no_diag=false)
{
	return implementation::sum_symmetric<backend,Matrix>::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of symmetric matrix-block sum of values that works
 * with generic dense matrix blocks 
 *
 * @param b the matrix-block whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
 */
template <Backend backend=linalg_traits<Redux>::backend,class Matrix>
typename Matrix::Scalar sum_symmetric(Block<Matrix> b, bool no_diag=false)
{
	return implementation::sum_symmetric<backend,Matrix>
		::compute(b, no_diag);
}

/**
 * Wrapper method for internal implementation of matrix colwise sum of values that works
 * with generic dense matrices
 *
 * @param m the matrix whose colwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}m_{i,j}\f$
 */
template <Backend backend=linalg_traits<Redux>::backend,class Matrix>
typename implementation::colwise_sum<backend,Matrix>::ReturnType colwise_sum(
	Matrix m, bool no_diag=false)
{
	return implementation::colwise_sum<backend,Matrix>::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of matrix colwise sum of values that works
 * with generic dense matrices
 *
 * @param m the matrix whose colwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @param result Pre-allocated vector for the result of the computation
 */
template <Backend backend=linalg_traits<Redux>::backend,class Matrix, class Vector>
void colwise_sum(Matrix m, Vector result, bool no_diag=false)
{
	implementation::colwise_sum<backend,Matrix>::compute(m, result, no_diag);
}

/**
 * Wrapper method for internal implementation of matrix rowwise sum of values that works
 * with generic dense matrices
 *
 * @param m the matrix whose rowwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}m_{i,j}\f$
 */
template <Backend backend=linalg_traits<Redux>::backend,class Matrix>
typename implementation::rowwise_sum<backend,Matrix>::ReturnType rowwise_sum(
	Matrix m, bool no_diag=false)
{
	return implementation::rowwise_sum<backend,Matrix>::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of matrix rowwise sum of values that works
 * with generic dense matrices
 *
 * @param m the matrix whose rowwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @param result Pre-allocated vector for the result of the computation
 */
template <Backend backend=linalg_traits<Redux>::backend,class Matrix, class Vector>
void rowwise_sum(Matrix m, Vector result, bool no_diag=false)
{
	implementation::rowwise_sum<backend,Matrix>::compute(m, result, no_diag);
}

/**
 * Wrapper method for internal implementation of vector sum of values that works
 * with generic dense vectors

 * @param a vector whose sum has to be computed
 * @return the vector sum \f$\sum_i a_i\f$
 */
template <Backend backend=linalg_traits<Redux>::backend, class Vector>
typename Vector::Scalar vector_sum(Vector a)
{
	return implementation::vector_sum<backend,Vector>::compute(a);
}

/** Returns the largest element in a matrix or vector */
template <Backend backend=linalg_traits<Redux>::backend, class Matrix>
typename Matrix::Scalar max(Matrix m)
{
	return implementation::max<backend,Matrix>::compute(m);
}

}

}
#endif // REDUX_H_
