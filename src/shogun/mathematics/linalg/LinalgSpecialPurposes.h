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
 * Authors: 2017 Pan Deng, 2014 Khaled Nasr
 */

#ifndef LINALG_SPECIAL_PURPOSE_H_
#define LINALG_SPECIAL_PURPOSE_H_

#include <shogun/mathematics/linalg/LinalgNamespace.h>

namespace shogun
{

namespace linalg
{

/** Applies the elementwise logistic function f(x) = 1/(1+exp(-x)) to a matrix
 *  This method returns the result in-place.
 *
 * @param a The input matrix
 * @param result The output matrix
 */
template <typename T>
void logistic(SGMatrix<T>& a, SGMatrix<T>& result)
{
	REQUIRE((a.num_rows == result.num_rows),
		"Number of rows of matrix a (%d) must match matrix result (%d).\n",
		a.num_rows, result.num_rows);
	REQUIRE(
		(a.num_cols == result.num_cols),
		"Number of columns of matrix a (%d) must match matrix result (%d).\n",
		a.num_cols, result.num_cols);

	infer_backend(a, result)->logistic(a, result);
}

/** Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j)) for all i and
 * j
 *  This method returns the result in-place.
 *
 * @param a The input matrix
 * @param result The output matrix
 */
template <typename T>
void multiply_by_logistic_derivative(SGMatrix<T>& a, SGMatrix<T>& result)
{
	REQUIRE(
		(a.num_rows == result.num_rows),
		"Number of rows of matrix a (%d) must match matrix result (%d).\n",
		a.num_rows, result.num_rows);
	REQUIRE(
		(a.num_cols == result.num_cols),
		"Number of columns of matrix a (%d) must match matrix result (%d).\n",
		a.num_cols, result.num_cols);

	infer_backend(a, result)->multiply_by_logistic_derivative(a, result);
}

/** Performs the operation C(i,j) = C(i,j) * (A(i,j)!=0) for all i and j
 *  This method returns the result in-place.
 *
 * @param a The input matrix
 * @param result The output matrix
 */
template <typename T>
void multiply_by_rectified_linear_derivative(
	SGMatrix<T>& a, SGMatrix<T>& result)
{
	REQUIRE(
		(a.num_rows == result.num_rows),
		"Number of rows of matrix a (%d) must match matrix result (%d).\n",
		a.num_rows, result.num_rows);
	REQUIRE(
		(a.num_cols == result.num_cols),
		"Number of columns of matrix a (%d) must match matrix result (%d).\n",
		a.num_cols, result.num_cols);

	infer_backend(a, result)->multiply_by_rectified_linear_derivative(
		a, result);
}

/** Applies the elementwise rectified linear function f(x) = max(0,x) to a
 * matrix
 *
 * @param a The input matrix
 * @param result The output matrix
 */
template <typename T>
void rectified_linear(SGMatrix<T>& a, SGMatrix<T>& result)
{
	REQUIRE(
		(a.num_rows == result.num_rows),
		"Number of rows of matrix a (%d) must match matrix result (%d).\n",
		a.num_rows, result.num_rows);
	REQUIRE(
		(a.num_cols == result.num_cols),
		"Number of columns of matrix a (%d) must match matrix result (%d).\n",
		a.num_cols, result.num_cols);

	infer_backend(a, result)->rectified_linear(a, result);
}

/** Applies the softmax function inplace to a matrix. The softmax function is
 * defined as \f$ f(A[i,j]) = \frac{exp(A[i,j])}{\sum_i exp(A[i,j])} \f$
 *  This method returns the result in-place.
 *
 * @param a The input matrix
 */
template <typename T>
void softmax(SGMatrix<T>& a)
{
	infer_backend(a)->softmax(a);
}

/** Returns the cross entropy between P and Q. The cross entropy is defined as
 * \f$ H(P,Q) = - \sum_{ij} P[i,j]log(Q[i,j]) \f$
 *
 * @param p Input matrix 1
 * @param q Input matrix 2
 */
template <typename T>
T cross_entropy(const SGMatrix<T> p, const SGMatrix<T> q)
{
	REQUIRE(
		(p.num_rows == q.num_rows),
		"Number of rows of matrix p (%d) must match matrix q (%d).\n",
		p.num_rows, q.num_rows);
	REQUIRE(
		(p.num_cols == q.num_cols),
		"Number of columns of matrix p (%d) must match matrix q (%d).\n",
		p.num_cols, q.num_cols);

	return infer_backend(p, q)->cross_entropy(p, q);
}

/** Returns the squared error between P and Q. The squared error is defined as
 * \f$ E(P,Q) = \frac{1}{2} \sum_{ij} (P[i,j]-Q[i,j])^2 \f$
 *
 * @param p Input matrix 1
 * @param q Input matrix 2
 */
template <typename T>
T squared_error(const SGMatrix<T> p, const SGMatrix<T> q)
{
	REQUIRE(
		(p.num_rows == q.num_rows),
		"Number of rows of matrix p (%d) must match matrix q (%d).\n",
		p.num_rows, q.num_rows);
	REQUIRE(
		(p.num_cols == q.num_cols),
		"Number of columns of matrix p (%d) must match matrix q (%d).\n",
		p.num_cols, q.num_cols);

	return infer_backend(p, q)->squared_error(p, q);
}
}

}

#endif //LINALG_SPECIAL_PURPOSE_H_
