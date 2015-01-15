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

#ifndef SPECIAL_PURPOSE_H_
#define SPECIAL_PURPOSE_H_

#ifdef HAVE_LINALG_LIB

#include <shogun/mathematics/linalg/internal/implementation/SpecialPurpose.h>

namespace shogun
{

namespace linalg
{

/** Contains special purpose, algorithm specific functions. Uses the same
 * backend as the Core module
 */
namespace special_purpose
{

/** Applies the elementwise logistic function f(x) = 1/(1+exp(-x)) to a matrix */
template <Backend backend=linalg_traits<Core>::backend,class Matrix>
void logistic(Matrix A, Matrix result)
{
	implementation::special_purpose::logistic<backend, Matrix>::compute(A, result);
}

/** Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j) for all i and j*/
template <Backend backend=linalg_traits<Core>::backend,class Matrix>
void multiply_by_logistic_derivative(Matrix A, Matrix C)
{
	implementation::special_purpose::multiply_by_logistic_derivative<backend, Matrix>::compute(A, C);
}

/** Applies the elementwise rectified linear function f(x) = max(0,x) to a matrix */
template <Backend backend=linalg_traits<Core>::backend,class Matrix>
void rectified_linear(Matrix A, Matrix result)
{
	implementation::special_purpose::rectified_linear<backend, Matrix>::compute(A, result);
}

/** Performs the operation C(i,j) = C(i,j) * (A(i,j)!=0) for all i and j*/
template <Backend backend=linalg_traits<Core>::backend,class Matrix>
void multiply_by_rectified_linear_derivative(Matrix A, Matrix C)
{
	implementation::special_purpose::multiply_by_rectified_linear_derivative<backend, Matrix>::compute(A, C);
}

/** Applies the softmax function inplace to a matrix. The softmax function is
 * defined as \f$ f(A[i,j]) = \frac{exp(A[i,j])}{\sum_i exp(A[i,j])} \f$
 */
template <Backend backend=linalg_traits<Core>::backend,class Matrix>
void softmax(Matrix A)
{
	implementation::special_purpose::softmax<backend, Matrix>::compute(A);
}

/** Returns the cross entropy between P and Q. The cross entropy is defined as
 * \f$ H(P,Q) = - \sum_{ij} P[i,j]log(Q[i,j]) \f$
 */
template <Backend backend=linalg_traits<Core>::backend,class Matrix>
typename Matrix::Scalar cross_entropy(Matrix P, Matrix Q)
{
	return implementation::special_purpose::cross_entropy<backend, Matrix>::compute(P,Q);
}

/** Returns the squared error between P and Q. The squared error is defined as
 * \f$ E(P,Q) = \frac{1}{2} \sum_{ij} (P[i,j]-Q[i,j])^2 \f$
 */
template <Backend backend=linalg_traits<Core>::backend,class Matrix>
typename Matrix::Scalar squared_error(Matrix P, Matrix Q)
{
	return implementation::special_purpose::squared_error<backend, Matrix>::compute(P,Q);
}

}

}

}
#endif // HAVE_LINALG_LIB
#endif // SPECIAL_PURPOSE_H_
