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

#ifndef LINALG_BACKEND_BASE_H__
#define LINALG_BACKEND_BASE_H__

#include <memory>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/GPUMemoryBase.h>
#include <shogun/mathematics/linalg/LinalgEnums.h>
#include <shogun/mathematics/linalg/internal/Block.h>

namespace shogun
{

	/** @brief Base interface of generic linalg methods
	 * and generic memory transfer methods.
	 */
	class LinalgBackendBase
	{
	public:
#define DEFINE_FOR_ALL_PTYPE(METHODNAME, Container)                            \
	METHODNAME(bool, Container);                                               \
	METHODNAME(char, Container);                                               \
	METHODNAME(int8_t, Container);                                             \
	METHODNAME(uint8_t, Container);                                            \
	METHODNAME(int16_t, Container);                                            \
	METHODNAME(uint16_t, Container);                                           \
	METHODNAME(int32_t, Container);                                            \
	METHODNAME(uint32_t, Container);                                           \
	METHODNAME(int64_t, Container);                                            \
	METHODNAME(uint64_t, Container);                                           \
	METHODNAME(float32_t, Container);                                          \
	METHODNAME(float64_t, Container);                                          \
	METHODNAME(floatmax_t, Container);                                         \
	METHODNAME(complex128_t, Container);

#define DEFINE_FOR_REAL_PTYPE(METHODNAME, Container)                           \
	METHODNAME(bool, Container);                                               \
	METHODNAME(char, Container);                                               \
	METHODNAME(int8_t, Container);                                             \
	METHODNAME(uint8_t, Container);                                            \
	METHODNAME(int16_t, Container);                                            \
	METHODNAME(uint16_t, Container);                                           \
	METHODNAME(int32_t, Container);                                            \
	METHODNAME(uint32_t, Container);                                           \
	METHODNAME(int64_t, Container);                                            \
	METHODNAME(uint64_t, Container);                                           \
	METHODNAME(float32_t, Container);                                          \
	METHODNAME(float64_t, Container);                                          \
	METHODNAME(floatmax_t, Container);

#define DEFINE_FOR_NON_INTEGER_PTYPE(METHODNAME, Container)                    \
	METHODNAME(float32_t, Container);                                          \
	METHODNAME(float64_t, Container);                                          \
	METHODNAME(floatmax_t, Container);                                         \
	METHODNAME(complex128_t, Container);

/**
 * Wrapper method of add operation the operation result = alpha*a + beta*b.
 *
 * @see linalg::add
 */
#define BACKEND_GENERIC_IN_PLACE_ADD(Type, Container)                          \
	virtual void add(                                                          \
	    const Container<Type>& a, const Container<Type>& b, Type alpha,        \
	    Type beta, Container<Type>& result) const                              \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ADD, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ADD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_ADD

/**
 * Wrapper method of add column vector result = alpha*A.col(i) + beta*b.
 *
 * @see linalg::add_col_vec
 */
#define BACKEND_GENERIC_ADD_COL_VEC(Type, Container)                           \
	virtual void add_col_vec(                                                  \
	    const SGMatrix<Type>& A, index_t i, const SGVector<Type>& b,           \
	    Container<Type>& result, Type alpha, Type beta) const                  \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return;                                                                \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ADD_COL_VEC, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ADD_COL_VEC, SGMatrix)
#undef BACKEND_GENERIC_ADD_COL_VEC

/**
 * Wrapper method of add diagonal vector A.diagonal = alpha * A.diagonal +
 * beta * b.
 *
 * @see linalg::add_diag
 */
#define BACKEND_GENERIC_ADD_DIAG(Type, Container)                              \
	virtual void add_diag(                                                     \
	    SGMatrix<Type>& A, const SGVector<Type>& b, Type alpha, Type beta)     \
	    const                                                                  \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return;                                                                \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ADD_DIAG, SGMatrix);
#undef BACKEND_GENERIC_ADD_DIAG

/**
 * Wrapper method of add vector to each column of matrix.
 *
 * @see linalg::add_vector
 */
#define BACKEND_GENERIC_ADD_VECTOR(Type, Container)                            \
	virtual void add_vector(                                                   \
	    const SGMatrix<Type>& A, const SGVector<Type>& b,                      \
	    SGMatrix<Type>& result, Type alpha, Type beta) const                   \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return;                                                                \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ADD_VECTOR, SGMatrix)
#undef BACKEND_GENERIC_ADD_VECTOR

/**
 * Wrapper method of add scalar operation.
 *
 * @see linalg::add_scalar
 */
#define BACKEND_GENERIC_ADD_SCALAR(Type, Container)                            \
	virtual void add_scalar(Container<Type>& a, Type b) const                  \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ADD_SCALAR, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ADD_SCALAR, SGMatrix)
#undef BACKEND_GENERIC_ADD_SCALAR

/**
 * Wrapper method of center matrix operation.
 *
 * @see linalg::center_matrix
 */
#define BACKEND_GENERIC_CENTER_MATRIX(Type, Container)                         \
	virtual void center_matrix(Container<Type>& A) const                       \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CENTER_MATRIX, SGMatrix)
#undef BACKEND_GENERIC_CENTER_MATRIX

/**
 * Wrapper method of Cholesky decomposition.
 *
 * @see linalg::cholesky_factor
 */
#define BACKEND_GENERIC_CHOLESKY_FACTOR(Type, Container)                       \
	virtual Container<Type> cholesky_factor(                                   \
	    const Container<Type>& A, const bool lower) const                      \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CHOLESKY_FACTOR, SGMatrix)
#undef BACKEND_GENERIC_CHOLESKY_FACTOR

/**
 * Wrapper triangular solver with Choleksy decomposition.
 *
 * @see linalg::cholesky_solver
 */
#define BACKEND_GENERIC_CHOLESKY_SOLVER(Type, Container)                       \
	virtual SGVector<Type> cholesky_solver(                                    \
	    const Container<Type>& L, const SGVector<Type>& b, const bool lower)   \
	    const                                                                  \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CHOLESKY_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_CHOLESKY_SOLVER

/**
 * Wrapper method of LDLT Cholesky decomposition
 *
 * @see linalg::ldlt_factor
 */
#define BACKEND_GENERIC_LDLT_FACTOR(Type, Container)                           \
	virtual void ldlt_factor(                                                  \
	    const Container<Type>& A, Container<Type>& L, SGVector<Type>& d,       \
	    SGVector<index_t>& p, const bool lower) const                          \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_LDLT_FACTOR, SGMatrix)
#undef BACKEND_GENERIC_LDLT_FACTOR

/**
 * Wrapper method of LDLT Cholesky solver
 *
 * @see linalg::ldlt_solver
 */
#define BACKEND_GENERIC_LDLT_SOLVER(Type, Container)                           \
	virtual SGVector<Type> ldlt_solver(                                        \
	    const Container<Type>& L, const SGVector<Type>& d,                     \
	    const SGVector<index_t>& p, const SGVector<Type>& b, const bool lower) \
	    const                                                                  \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_LDLT_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_LDLT_SOLVER

/**
 * Wrapper method of cross entropy.
 *
 * @see linalg::cross_entropy
 */
#define BACKEND_GENERIC_CROSS_ENTROPY(Type, Container)                         \
	virtual Type cross_entropy(                                                \
	    const Container<Type>& P, const Container<Type>& Q) const              \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_CROSS_ENTROPY, SGMatrix)
#undef BACKEND_GENERIC_CROSS_ENTROPY

/**
 * Wrapper method of vector dot-product that works with generic vectors.
 *
 * @see linalg::dot
 */
#define BACKEND_GENERIC_DOT(Type, Container)                                   \
	virtual Type dot(const Container<Type>& a, const Container<Type>& b) const \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_DOT, SGVector)
#undef BACKEND_GENERIC_DOT

/**
 * Wrapper method of eigenvalues and eigenvectors computation.
 *
 * @see linalg::eigen_solver
 */
#define BACKEND_GENERIC_EIGEN_SOLVER(Type, Container)                          \
	virtual void eigen_solver(                                                 \
	    const Container<Type>& A, SGVector<Type>& eigenvalues,                 \
	    SGMatrix<Type>& eigenvectors) const                                    \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_EIGEN_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_EIGEN_SOLVER

/**
 * Wrapper method of eigenvalues and eigenvectors computation
 * for symmetric matrices.
 *
 * @see linalg::eigen_solver_symmetric
 */
#define BACKEND_GENERIC_EIGEN_SOLVER_SYMMETRIC(Type, Container)                \
	virtual void eigen_solver_symmetric(                                       \
	    const Container<Type>& A, SGVector<Type>& eigenvalues,                 \
	    SGMatrix<Type>& eigenvectors, index_t k) const                         \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_NON_INTEGER_PTYPE(
		    BACKEND_GENERIC_EIGEN_SOLVER_SYMMETRIC, SGMatrix)
#undef BACKEND_GENERIC_EIGEN_SOLVER_SYMMETRIC

/**
 * Wrapper method of in-place matrix/vector elementwise product.
 *
 * @see linalg::element_prod
 */
#define BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD(Type, Container)                 \
	virtual void element_prod(                                                 \
	    const Container<Type>& a, const Container<Type>& b,                    \
	    Container<Type>& result) const                                         \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD, SGMatrix)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD, SGVector)
#undef BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD

/**
 * Wrapper method of in-place matrix block elementwise product.
 *
 * @see linalg::element_prod
 */
#define BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD(Type, Container)           \
	virtual void element_prod(                                                 \
	    const linalg::Block<Container<Type>>& a,                               \
	    const linalg::Block<Container<Type>>& b, Container<Type>& result)      \
	    const                                                                  \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(
		    BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD

/**
 * Wrapper method of in-place exponent method.
 *
 * @see linalg::exponent
 */
#define BACKEND_GENERIC_EXPONENT(Type, Container)                              \
	virtual void exponent(const Container<Type>& a, Container<Type>& result)   \
	    const                                                                  \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_EXPONENT, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_EXPONENT, SGMatrix)
#undef BACKEND_GENERIC_EXPONENT

/**
 * Wrapper method of set matrix to identity.
 *
 * @see linalg::identity
 */
#define BACKEND_GENERIC_IDENTITY(Type, Container)                              \
	virtual void identity(Container<Type>& identity_matrix) const              \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return;                                                                \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IDENTITY, SGMatrix)
#undef BACKEND_GENERIC_IDENTITY

/**
 * Wrapper method of logistic function f(x) = 1/(1+exp(-x))
 *
 * @see linalg::logistic
 */
#define BACKEND_GENERIC_LOGISTIC(Type, Container)                              \
	virtual void logistic(Container<Type>& a, Container<Type>& result) const   \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_LOGISTIC, SGMatrix)
#undef BACKEND_GENERIC_LOGISTIC

/**
 * Wrapper method of matrix product method.
 *
 * @see linalg::matrix_prod
 */
#define BACKEND_GENERIC_IN_PLACE_MATRIX_PROD(Type, Container)                  \
	virtual void matrix_prod(                                                  \
	    const SGMatrix<Type>& a, const Container<Type>& b,                     \
	    Container<Type>& result, bool transpose_A, bool transpose_B) const     \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_MATRIX_PROD, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_MATRIX_PROD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_MATRIX_PROD

/**
 * Wrapper method of max method. Return the largest element in a vector or
 * matrix.
 *
 * @see linalg::max
 */
#define BACKEND_GENERIC_MAX(Type, Container)                                   \
	virtual Type max(const Container<Type>& a) const                           \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_MAX, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_MAX, SGMatrix)
#undef BACKEND_GENERIC_MAX

/**
* Wrapper method that computes mean of SGVectors and SGMatrices
* that are composed of real numbers.
*
* @see linalg::mean
*/
#define BACKEND_GENERIC_REAL_MEAN(Type, Container)                             \
	virtual float64_t mean(const Container<Type>& a) const                     \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_REAL_MEAN, SGVector)
		DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_REAL_MEAN, SGMatrix)
#undef BACKEND_GENERIC_REAL_MEAN

/**
* Wrapper method that computes mean of SGVectors and SGMatrices
* that are composed of complex numbers.
*
* @see linalg::mean
*/
#define BACKEND_GENERIC_COMPLEX_MEAN(Container)                                \
	virtual complex128_t mean(const Container<complex128_t>& a) const          \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		BACKEND_GENERIC_COMPLEX_MEAN(SGVector)
		BACKEND_GENERIC_COMPLEX_MEAN(SGMatrix)
#undef BACKEND_GENERIC_COMPLEX_MEAN

/**
 * Wrapper method of multiply_by_logistic_derivative function f(x) =
 * 1/(1+exp(-x))
 *
 * @see linalg::multiply_by_logistic_derivative
 */
#define BACKEND_GENERIC_MULTIPLY_BY_LOGISTIC_DERIV(Type, Container)            \
	virtual void multiply_by_logistic_derivative(                              \
	    Container<Type>& a, Container<Type>& result) const                     \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(
		    BACKEND_GENERIC_MULTIPLY_BY_LOGISTIC_DERIV, SGMatrix)
#undef BACKEND_GENERIC_MULTIPLY_BY_LOGISTIC_DERIV

/**
 * Wrapper method of multiply_by_rectified_linear_derivative
 *
 * @see linalg::multiply_by_rectified_linear_derivative
 */
#define BACKEND_GENERIC_MULTIPLY_BY_RECTIFIED_LINEAR_DERIV(Type, Container)    \
	virtual void multiply_by_rectified_linear_derivative(                      \
	    Container<Type>& a, Container<Type>& result) const                     \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(
		    BACKEND_GENERIC_MULTIPLY_BY_RECTIFIED_LINEAR_DERIV, SGMatrix)
#undef BACKEND_GENERIC_MULTIPLY_BY_RECTIFIED_LINEAR_DERIV

/**
* Wrapper method that range fills a vector of matrix.
*
* @see linalg::range_fill
*/
#define BACKEND_GENERIC_RANGE_FILL(Type, Container)                            \
	virtual void range_fill(Container<Type>& a, const Type start) const        \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_RANGE_FILL, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_RANGE_FILL, SGMatrix)
#undef BACKEND_GENERIC_RANGE_FILL

/**
 * Wrapper method of rectified_linear method f(x) = max(0, x)
 *
 * @see linalg::rectified_linear
 */
#define BACKEND_GENERIC_RECTIFIED_LINEAR(Type, Container)                      \
	virtual void rectified_linear(Container<Type>& a, Container<Type>& result) \
	    const                                                                  \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_RECTIFIED_LINEAR, SGMatrix)
#undef BACKEND_GENERIC_RECTIFIED_LINEAR

/**
 * Wrapper method that solves a system of linear equations
 * using QR decomposition.
 *
 * @see linalg::qr_solver
 */
#define BACKEND_GENERIC_QR_SOLVER(Type, Container)                             \
	virtual Container<Type> qr_solver(                                         \
	    const SGMatrix<Type>& A, const Container<Type>& b) const               \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_QR_SOLVER, SGVector)
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_QR_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_QR_SOLVER

/**
 * Wrapper method of scale operation the operation result = alpha*A.
 *
 * @see linalg::scale
 */
#define BACKEND_GENERIC_IN_PLACE_SCALE(Type, Container)                        \
	virtual void scale(                                                        \
	    const Container<Type>& a, Type alpha, Container<Type>& result) const   \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_SCALE, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_SCALE, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_SCALE

/**
 * Wrapper method that sets const values to vectors or matrices.
 *
 * @see linalg::set_const
 */
#define BACKEND_GENERIC_SET_CONST(Type, Container)                             \
	virtual void set_const(Container<Type>& a, const Type value) const         \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SET_CONST, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SET_CONST, SGMatrix)
#undef BACKEND_GENERIC_SET_CONST

/**
* Wrapper method of sum that works with generic vectors or matrices.
*
* @see linalg::sum
*/
#define BACKEND_GENERIC_SUM(Type, Container)                                   \
	virtual Type sum(const Container<Type>& a, bool no_diag) const             \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGMatrix)
#undef BACKEND_GENERIC_SUM

/**
 * Wrapper method of softmax method.
 *
 * @see linalg::softmax
 */
#define BACKEND_GENERIC_SOFTMAX(Type, Container)                               \
	virtual void softmax(Container<Type>& a) const                             \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SOFTMAX, SGMatrix)
#undef BACKEND_GENERIC_SOFTMAX

/**
* Wrapper method of squared error method.
*
* @see linalg::squared_error
*/
#define BACKEND_GENERIC_SQUARED_ERROR(Type, Container)                         \
	virtual Type squared_error(                                                \
	    const Container<Type>& P, const Container<Type>& Q) const              \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SQUARED_ERROR, SGMatrix)
#undef BACKEND_GENERIC_SQUARED_ERROR

/**
* Wrapper method of sum that works with matrix blocks.
*
* @see linalg::sum
*/
#define BACKEND_GENERIC_BLOCK_SUM(Type, Container)                             \
	virtual Type sum(const linalg::Block<Container<Type>>& a, bool no_diag)    \
	    const                                                                  \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_SUM

/**
* Wrapper method of sum that works with symmetric matrices.
*
* @see linalg::sum_symmetric
*/
#define BACKEND_GENERIC_SYMMETRIC_SUM(Type, Container)                         \
	virtual Type sum_symmetric(const Container<Type>& a, bool no_diag) const   \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SYMMETRIC_SUM, SGMatrix)
#undef BACKEND_GENERIC_SYMMETRIC_SUM

/**
* Wrapper method of sum that works with symmetric matrix blocks.
*
* @see linalg::sum
*/
#define BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM(Type, Container)                   \
	virtual Type sum_symmetric(                                                \
	    const linalg::Block<Container<Type>>& a, bool no_diag) const           \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM, SGMatrix)
#undef BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM

/**
 * Wrapper method of matrix rowwise sum that works with dense matrices.
 *
 * @see linalg::colwise_sum
 */
#define BACKEND_GENERIC_COLWISE_SUM(Type, Container)                           \
	virtual SGVector<Type> colwise_sum(const Container<Type>& a, bool no_diag) \
	    const                                                                  \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_COLWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_COLWISE_SUM

/**
* Wrapper method of matrix colwise sum that works with dense matrices.
*
* @see linalg::colwise_sum
*/
#define BACKEND_GENERIC_BLOCK_COLWISE_SUM(Type, Container)                     \
	virtual SGVector<Type> colwise_sum(                                        \
	    const linalg::Block<Container<Type>>& a, bool no_diag) const           \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_COLWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_COLWISE_SUM

/**
 * Wrapper method of matrix rowwise sum that works with dense matrices.
 *
 * @see linalg::rowwise_sum
 */
#define BACKEND_GENERIC_ROWWISE_SUM(Type, Container)                           \
	virtual SGVector<Type> rowwise_sum(const Container<Type>& a, bool no_diag) \
	    const                                                                  \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ROWWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_ROWWISE_SUM

/**
* Wrapper method of matrix rowwise sum that works with dense matrices.
*
* @see linalg::rowwise_sum
*/
#define BACKEND_GENERIC_BLOCK_ROWWISE_SUM(Type, Container)                     \
	virtual SGVector<Type> rowwise_sum(                                        \
	    const linalg::Block<Container<Type>>& a, bool no_diag) const           \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_ROWWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_ROWWISE_SUM

/**
 * Wrapper method of svd computation.
 *
 * @see linalg::svd
 */
#define BACKEND_GENERIC_SVD(Type, Container)                                   \
	virtual void svd(                                                          \
	    const Container<Type>& A, SGVector<Type> s, SGMatrix<Type> U,          \
	    bool thin_U, linalg::SVDAlgorithm alg) const                           \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_SVD, SGMatrix)
#undef BACKEND_GENERIC_SVD

/**
 * Wrapper method of trace computation.
 *
 * @see linalg::trace
 */
#define BACKEND_GENERIC_TRACE(Type, Container)                                 \
	virtual Type trace(const Container<Type>& A) const                         \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_TRACE, SGMatrix)
#undef BACKEND_GENERIC_TRACE

/**
 * Wrapper method of trace computation.
 *
 * @see linalg::transpose_matrix
 */
#define BACKEND_GENERIC_TRANSPOSE_MATRIX(Type, Container)                      \
	virtual Container<Type> transpose_matrix(const Container<Type>& A) const   \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_TRANSPOSE_MATRIX, SGMatrix)
#undef BACKEND_GENERIC_TRANSPOSE_MATRIX

/**
 * Wrapper method of triangular solver.
 *
 * @see linalg::triangular_solver
 */
#define BACKEND_GENERIC_TRIANGULAR_SOLVER(Type, Container)                     \
	virtual Container<Type> triangular_solver(                                 \
	    const SGMatrix<Type>& L, const Container<Type>& b,                     \
	    const bool lower = true) const                                         \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_NON_INTEGER_PTYPE(
		    BACKEND_GENERIC_TRIANGULAR_SOLVER, SGVector)
		DEFINE_FOR_NON_INTEGER_PTYPE(
		    BACKEND_GENERIC_TRIANGULAR_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_TRIANGULAR_SOLVER

/**
 * Wrapper method of set vector or matrix to zero.
 *
 * @see linalg::zero
 */
#define BACKEND_GENERIC_ZERO(Type, Container)                                  \
	virtual void zero(Container<Type>& a) const                                \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return;                                                                \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ZERO, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ZERO, SGMatrix)
#undef BACKEND_GENERIC_ZERO

/**
 * Wrapper method of Transferring data to GPU memory.
 * Does nothing if no GPU backend registered.
 *
 * @see linalg::to_gpu
 */
#define BACKEND_GENERIC_TO_GPU(Type, Container)                                \
	virtual GPUMemoryBase<Type>* to_gpu(const Container<Type>&) const          \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
		return 0;                                                              \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_TO_GPU, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_TO_GPU, SGMatrix)
#undef BACKEND_GENERIC_TO_GPU

/**
 * Wrapper method of fetching data from GPU memory.
 *
 * @see linalg::from_gpu
 */
#define BACKEND_GENERIC_FROM_GPU(Type, Container)                              \
	virtual void from_gpu(const Container<Type>&, Type* data) const            \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_FROM_GPU, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_FROM_GPU, SGMatrix)
#undef BACKEND_GENERIC_FROM_GPU

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_REAL_PTYPE
#undef DEFINE_FOR_NON_INTEGER_PTYPE
	};
}

#endif // LINALG_BACKEND_BASE_H__
