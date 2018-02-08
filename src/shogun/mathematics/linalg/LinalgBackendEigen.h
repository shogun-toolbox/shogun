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

#ifndef LINALG_BACKEND_EIGEN_H__
#define LINALG_BACKEND_EIGEN_H__

#include <numeric>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/LinalgBackendBase.h>
#include <shogun/mathematics/linalg/LinalgEnums.h>
#include <shogun/mathematics/linalg/LinalgMacros.h>

namespace shogun
{

	/** @brief Linalg methods with Eigen3 backend */
	class LinalgBackendEigen : public LinalgBackendBase
	{
	public:
/** Implementation of @see LinalgBackendBase::add */
#define BACKEND_GENERIC_IN_PLACE_ADD(Type, Container)                          \
	virtual void add(                                                          \
	    const Container<Type>& a, const Container<Type>& b, Type alpha,        \
	    Type beta, Container<Type>& result) const;
		DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_IN_PLACE_ADD, SGVector)
		DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_IN_PLACE_ADD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_ADD

/** Implementation of @see LinalgBackendBase::add_col_vec */
#define BACKEND_GENERIC_ADD_COL_VEC(Type, Container)                           \
	virtual void add_col_vec(                                                  \
	    const SGMatrix<Type>& A, index_t i, const SGVector<Type>& b,           \
	    Container<Type>& result, Type alpha, Type beta) const;
		DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_COL_VEC, SGVector)
		DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_COL_VEC, SGMatrix)
#undef BACKEND_GENERIC_ADD_COL_VEC

/** Implementation of @see LinalgBackendBase::add_diag */
#define BACKEND_GENERIC_ADD_DIAG(Type, Container)                              \
	virtual void add_diag(                                                     \
	    SGMatrix<Type>& A, const SGVector<Type>& b, Type alpha, Type beta)     \
	    const;
		DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_DIAG, SGMatrix)
#undef BACKEND_GENERIC_ADD_DIAG

/** Implementation of @see LinalgBackendBase::add_vector */
#define BACKEND_GENERIC_ADD(Type, Container)                                   \
	virtual void add_vector(                                                   \
	    const SGMatrix<Type>& A, const SGVector<Type>& b,                      \
	    SGMatrix<Type>& result, Type alpha, Type beta) const;
		DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD, SGMatrix)
#undef BACKEND_GENERIC_ADD

/** Implementation of @see LinalgBackendBase::add_scalar */
#define BACKEND_GENERIC_ADD_SCALAR(Type, Container)                            \
	virtual void add_scalar(Container<Type>& a, Type b) const;
		DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_SCALAR, SGVector)
		DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_SCALAR, SGMatrix)
#undef BACKEND_GENERIC_ADD_SCALAR

/** Implementation of @see LinalgBackendBase::center_matrix */
#define BACKEND_GENERIC_CENTER_MATRIX(Type, Container)                         \
	virtual void center_matrix(Container<Type>& A) const;
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CENTER_MATRIX, SGMatrix)
#undef BACKEND_GENERIC_CENTER_MATRIX

/** Implementation of @see LinalgBackendBase::cholesky_factor */
#define BACKEND_GENERIC_CHOLESKY_FACTOR(Type, Container)                       \
	virtual Container<Type> cholesky_factor(                                   \
	    const Container<Type>& A, const bool lower) const;
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CHOLESKY_FACTOR, SGMatrix)
#undef BACKEND_GENERIC_CHOLESKY_FACTOR

/** Implementation of @see LinalgBackendBase::cholesky_solver */
#define BACKEND_GENERIC_CHOLESKY_SOLVER(Type, Container)                       \
	virtual SGVector<Type> cholesky_solver(                                    \
	    const Container<Type>& L, const SGVector<Type>& b, const bool lower)   \
	    const;
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CHOLESKY_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_CHOLESKY_SOLVER

/** Implementation of @see LinalgBackendBase::ldlt_factor */
#define BACKEND_GENERIC_LDLT_FACTOR(Type, Container)                           \
	virtual void ldlt_factor(                                                  \
	    const Container<Type>& A, Container<Type>& L, SGVector<Type>& d,       \
	    SGVector<index_t>& p, const bool lower) const;
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_LDLT_FACTOR, SGMatrix)
#undef BACKEND_GENERIC_LDLT_FACTOR

/** Implementation of @see LinalgBackendBase::ldlt_solver */
#define BACKEND_GENERIC_LDLT_SOLVER(Type, Container)                           \
	virtual SGVector<Type> ldlt_solver(                                        \
	    const Container<Type>& L, const SGVector<Type>& d,                     \
	    const SGVector<index_t>& p, const SGVector<Type>& b, const bool lower) \
	    const;
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_LDLT_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_LDLT_SOLVER

/** Implementation of @see linalg::cross_entropy */
#define BACKEND_GENERIC_CROSS_ENTROPY(Type, Container)                         \
	virtual Type cross_entropy(                                                \
	    const Container<Type>& P, const Container<Type>& Q) const;
		DEFINE_FOR_NON_INTEGER_REAL_PTYPE(
		    BACKEND_GENERIC_CROSS_ENTROPY, SGMatrix)
#undef BACKEND_GENERIC_CROSS_ENTROPY

/** Implementation of @see LinalgBackendBase::dot */
#define BACKEND_GENERIC_DOT(Type, Container)                                   \
	virtual Type dot(const Container<Type>& a, const Container<Type>& b) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_DOT, SGVector)
#undef BACKEND_GENERIC_DOT

/** Implementation of @see LinalgBackendBase::eigen_solver */
#define BACKEND_GENERIC_EIGEN_SOLVER(Type, Container)                          \
	virtual void eigen_solver(                                                 \
	    const Container<Type>& A, SGVector<Type>& eigenvalues,                 \
	    SGMatrix<Type>& eigenvectors) const;
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_EIGEN_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_EIGEN_SOLVER

/** Implementation of @see LinalgBackendBase::eigen_solver_symmetric */
#define BACKEND_GENERIC_EIGEN_SOLVER_SYMMETRIC(Type, Container)                \
	virtual void eigen_solver_symmetric(                                       \
	    const Container<Type>& A, SGVector<Type>& eigenvalues,                 \
	    SGMatrix<Type>& eigenvectors, index_t k) const;
		DEFINE_FOR_NON_INTEGER_PTYPE(
		    BACKEND_GENERIC_EIGEN_SOLVER_SYMMETRIC, SGMatrix)
#undef BACKEND_GENERIC_EIGEN_SOLVER_SYMMETRIC

/** Implementation of @see LinalgBackendBase::element_prod */
#define BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD(Type, Container)                 \
	virtual void element_prod(                                                 \
	    const Container<Type>& a, const Container<Type>& b,                    \
	    Container<Type>& result) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD, SGMatrix)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD, SGVector)
#undef BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD

/** Implementation of @see LinalgBackendBase::element_prod */
#define BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD(Type, Container)           \
	virtual void element_prod(                                                 \
	    const linalg::Block<Container<Type>>& a,                               \
	    const linalg::Block<Container<Type>>& b, Container<Type>& result)      \
	    const;
		DEFINE_FOR_ALL_PTYPE(
		    BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD

/** Implementation of @see linalg::exponent */
#define BACKEND_GENERIC_EXPONENT(Type, Container)                              \
	virtual void exponent(const Container<Type>& a, Container<Type>& result)   \
	    const
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_EXPONENT, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_EXPONENT, SGMatrix)
#undef BACKEND_GENERIC_EXPONENT

/** Implementation of @see LinalgBackendBase::identity */
#define BACKEND_GENERIC_IDENTITY(Type, Container)                              \
	virtual void identity(Container<Type>& identity_matrix) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IDENTITY, SGMatrix)
#undef BACKEND_GENERIC_IDENTITY

/** Implementation of @see LinalgBackendBase::logistic */
#define BACKEND_GENERIC_LOGISTIC(Type, Container)                              \
	virtual void logistic(const Container<Type>& a, Container<Type>& result)   \
	    const;
		DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_LOGISTIC, SGMatrix)
#undef BACKEND_GENERIC_LOGISTIC

/** Implementation of @see LinalgBackendBase::matrix_prod */
#define BACKEND_GENERIC_IN_PLACE_MATRIX_PROD(Type, Container)                  \
	virtual void matrix_prod(                                                  \
	    const SGMatrix<Type>& a, const Container<Type>& b,                     \
	    Container<Type>& result, bool transpose_A, bool transpose_B) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_MATRIX_PROD, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_MATRIX_PROD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_MATRIX_PROD

/** Implementation of @see LinalgBackendBase::max */
#define BACKEND_GENERIC_MAX(Type, Container)                                   \
	virtual Type max(const Container<Type>& a) const;
		DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_MAX, SGVector)
		DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_MAX, SGMatrix)
#undef BACKEND_GENERIC_MAX

/** Implementation of @see LinalgBackendBase::mean */
#define BACKEND_GENERIC_REAL_MEAN(Type, Container)                             \
	virtual float64_t mean(const Container<Type>& a) const;
		DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_REAL_MEAN, SGVector)
		DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_REAL_MEAN, SGMatrix)
#undef BACKEND_GENERIC_REAL_MEAN

/** Implementation of @see LinalgBackendBase::mean */
#define BACKEND_GENERIC_COMPLEX_MEAN(Container)                                \
	virtual complex128_t mean(const Container<complex128_t>& a) const;
		BACKEND_GENERIC_COMPLEX_MEAN(SGVector)
		BACKEND_GENERIC_COMPLEX_MEAN(SGMatrix)
#undef BACKEND_GENERIC_COMPLEX_MEAN

/** Implementation of @see linalg::multiply_by_logistic_derivative */
#define BACKEND_GENERIC_MULTIPLY_BY_LOGISTIC_DERIV(Type, Container)            \
	virtual void multiply_by_logistic_derivative(                              \
	    const Container<Type>& a, Container<Type>& result) const;
		DEFINE_FOR_NUMERIC_PTYPE(
		    BACKEND_GENERIC_MULTIPLY_BY_LOGISTIC_DERIV, SGMatrix)
#undef BACKEND_GENERIC_MULTIPLY_BY_LOGISTIC_DERIV

/** Implementation of @see linalg::multiply_by_rectified_linear_derivative */
#define BACKEND_GENERIC_MULTIPLY_BY_RECTIFIED_LINEAR_DERIV(Type, Container)    \
	virtual void multiply_by_rectified_linear_derivative(                      \
	    const Container<Type>& a, Container<Type>& result) const;
		DEFINE_FOR_NON_INTEGER_REAL_PTYPE(
		    BACKEND_GENERIC_MULTIPLY_BY_RECTIFIED_LINEAR_DERIV, SGMatrix)
#undef BACKEND_GENERIC_MULTIPLY_BY_RECTIFIED_LINEAR_DERIV

/** Implementation of @see LinalgBackendBase::qr_solver */
#define BACKEND_GENERIC_QR_SOLVER(Type, Container)                             \
	virtual Container<Type> qr_solver(                                         \
	    const SGMatrix<Type>& A, const Container<Type>& b) const;
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_QR_SOLVER, SGVector)
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_QR_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_QR_SOLVER

/** Implementation of @see LinalgBackendBase::range_fill */
#define BACKEND_GENERIC_RANGE_FILL(Type, Container)                            \
	virtual void range_fill(Container<Type>& a, const Type start) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_RANGE_FILL, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_RANGE_FILL, SGMatrix)
#undef BACKEND_GENERIC_RANGE_FILL

/** Implementation of @see linalg::rectified_linear */
#define BACKEND_GENERIC_RECTIFIED_LINEAR(Type, Container)                      \
	virtual void rectified_linear(                                             \
	    const Container<Type>& a, Container<Type>& result) const;
		DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_RECTIFIED_LINEAR, SGMatrix)
#undef BACKEND_GENERIC_RECTIFIED_LINEAR

/** Implementation of @see linalg::scale */
#define BACKEND_GENERIC_IN_PLACE_SCALE(Type, Container)                        \
	virtual void scale(                                                        \
	    const Container<Type>& a, Type alpha, Container<Type>& result) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_SCALE, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_SCALE, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_SCALE

/** Implementation of @see LinalgBackendBase::set_const */
#define BACKEND_GENERIC_SET_CONST(Type, Container)                             \
	virtual void set_const(Container<Type>& a, const Type value) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SET_CONST, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SET_CONST, SGMatrix)
#undef BACKEND_GENERIC_SET_CONST

/** Implementation of @see linalg::softmax */
#define BACKEND_GENERIC_SOFTMAX(Type, Container)                               \
	virtual void softmax(Container<Type>& a) const;
		DEFINE_FOR_NON_INTEGER_REAL_PTYPE(BACKEND_GENERIC_SOFTMAX, SGMatrix)
#undef BACKEND_GENERIC_SOFTMAX

/** Implementation of @see linalg::squared_error */
#define BACKEND_GENERIC_SQUARED_ERROR(Type, Container)                         \
	virtual Type squared_error(                                                \
	    const Container<Type>& P, const Container<Type>& Q) const;
		DEFINE_FOR_NON_INTEGER_REAL_PTYPE(
		    BACKEND_GENERIC_SQUARED_ERROR, SGMatrix)
#undef BACKEND_GENERIC_SQUARED_ERROR

/** Implementation of @see LinalgBackendBase::sum */
#define BACKEND_GENERIC_SUM(Type, Container)                                   \
	virtual Type sum(const Container<Type>& a, bool no_diag) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGMatrix)
#undef BACKEND_GENERIC_SUM

/** Implementation of @see LinalgBackendBase::sum */
#define BACKEND_GENERIC_BLOCK_SUM(Type, Container)                             \
	virtual Type sum(const linalg::Block<Container<Type>>& a, bool no_diag)    \
	    const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_SUM

/** Implementation of @see LinalgBackendBase::sum_symmetric */
#define BACKEND_GENERIC_SYMMETRIC_SUM(Type, Container)                         \
	virtual Type sum_symmetric(const Container<Type>& a, bool no_diag) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SYMMETRIC_SUM, SGMatrix)
#undef BACKEND_GENERIC_SYMMETRIC_SUM

/** Implementation of @see LinalgBackendBase::sum_symmetric */
#define BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM(Type, Container)                   \
	virtual Type sum_symmetric(                                                \
	    const linalg::Block<Container<Type>>& a, bool no_diag) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM, SGMatrix)
#undef BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM

/** Implementation of @see LinalgBackendBase::colwise_sum */
#define BACKEND_GENERIC_COLWISE_SUM(Type, Container)                           \
	virtual SGVector<Type> colwise_sum(const Container<Type>& a, bool no_diag) \
	    const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_COLWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_COLWISE_SUM

/** Implementation of @see LinalgBackendBase::colwise_sum */
#define BACKEND_GENERIC_BLOCK_COLWISE_SUM(Type, Container)                     \
	virtual SGVector<Type> colwise_sum(                                        \
	    const linalg::Block<Container<Type>>& a, bool no_diag) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_COLWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_COLWISE_SUM

/** Implementation of @see LinalgBackendBase::rowwise_sum */
#define BACKEND_GENERIC_ROWWISE_SUM(Type, Container)                           \
	virtual SGVector<Type> rowwise_sum(const Container<Type>& a, bool no_diag) \
	    const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ROWWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_ROWWISE_SUM

/** Implementation of @see LinalgBackendBase::rowwise_sum */
#define BACKEND_GENERIC_BLOCK_ROWWISE_SUM(Type, Container)                     \
	virtual SGVector<Type> rowwise_sum(                                        \
	    const linalg::Block<Container<Type>>& a, bool no_diag) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_ROWWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_ROWWISE_SUM

/** Implementation of @see LinalgBackendBase::svd */
#define BACKEND_GENERIC_SVD(Type, Container)                                   \
	virtual void svd(                                                          \
	    const Container<Type>& A, SGVector<Type> s, Container<Type> U,         \
	    bool thin_U, linalg::SVDAlgorithm alg) const;
		DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_SVD, SGMatrix)
#undef BACKEND_GENERIC_SVD

/** Implementation of @see LinalgBackendBase::trace */
#define BACKEND_GENERIC_TRACE(Type, Container)                                 \
	virtual Type trace(const Container<Type>& A) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_TRACE, SGMatrix)
#undef BACKEND_GENERIC_TRACE

/** Implementation of @see LinalgBackendBase::transpose_matrix */
#define BACKEND_GENERIC_TRANSPOSE_MATRIX(Type, Container)                      \
	virtual Container<Type> transpose_matrix(const Container<Type>& A) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_TRANSPOSE_MATRIX, SGMatrix)
#undef BACKEND_GENERIC_TRANSPOSE_MATRIX

/** Implementation of @see LinalgBackendBase::triangular_solver */
#define BACKEND_GENERIC_TRIANGULAR_SOLVER(Type, Container)                     \
	virtual Container<Type> triangular_solver(                                 \
	    const SGMatrix<Type>& L, const Container<Type>& b, const bool lower)   \
	    const;
		DEFINE_FOR_NON_INTEGER_PTYPE(
		    BACKEND_GENERIC_TRIANGULAR_SOLVER, SGVector)
		DEFINE_FOR_NON_INTEGER_PTYPE(
		    BACKEND_GENERIC_TRIANGULAR_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_TRIANGULAR_SOLVER

/** Implementation of @see LinalgBackendBase::zero */
#define BACKEND_GENERIC_ZERO(Type, Container)                                  \
	virtual void zero(Container<Type>& a) const;
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ZERO, SGVector)
		DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ZERO, SGMatrix)
#undef BACKEND_GENERIC_ZERO

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_REAL_PTYPE
#undef DEFINE_FOR_NON_INTEGER_PTYPE
#undef DEFINE_FOR_NUMERIC_PTYPE

	private:
		/** Eigen3 vector result = alpha*A + beta*B method */
		template <typename T>
		void add_impl(
		    const SGVector<T>& a, const SGVector<T>& b, T alpha, T beta,
		    SGVector<T>& result) const;

		/** Eigen3 matrix result = alpha*A + beta*B method */
		template <typename T>
		void add_impl(
		    const SGMatrix<T>& a, const SGMatrix<T>& b, T alpha, T beta,
		    SGMatrix<T>& result) const;

		/** Eigen3 add column vector method */
		template <typename T>
		void add_col_vec_impl(
		    const SGMatrix<T>& A, index_t i, const SGVector<T>& b,
		    SGMatrix<T>& result, T alpha, T beta) const;

		/** Eigen3 add column vector method */
		template <typename T>
		void add_col_vec_impl(
		    const SGMatrix<T>& A, index_t i, const SGVector<T>& b,
		    SGVector<T>& result, T alpha, T beta) const;

		/** Eigen3 add diagonal vector method */
		template <typename T>
		void add_diag_impl(
		    SGMatrix<T>& A, const SGVector<T>& b, T alpha, T beta) const;

		/** Eigen3 add vector to each column of matrix method */
		template <typename T>
		void add_vector_impl(
		    const SGMatrix<T>& A, const SGVector<T>& b, SGMatrix<T>& result,
		    T alpha, T beta) const;

		/** Eigen3 vector add scalar method */
		template <typename T>
		void add_scalar_impl(SGVector<T>& a, T b) const;

		/** Eigen3 matrix add scalar method */
		template <typename T>
		void add_scalar_impl(SGMatrix<T>& a, T b) const;

		/** Eigen3 center matrix method */
		template <typename T>
		void center_matrix_impl(SGMatrix<T>& A) const;

		/** Eigen3 Cholesky decomposition */
		template <typename T>
		SGMatrix<T>
		cholesky_factor_impl(const SGMatrix<T>& A, const bool lower) const;

		/** Eigen3 Cholesky solver */
		template <typename T>
		SGVector<T> cholesky_solver_impl(
		    const SGMatrix<T>& L, const SGVector<T>& b, const bool lower) const;

		/** Eigen3 LDLT Cholesky decomposition */
		template <typename T>
		void ldlt_factor_impl(
		    const SGMatrix<T>& A, SGMatrix<T>& L, SGVector<T>& d,
		    SGVector<index_t>& p, const bool lower) const;

		/** Eigen3 LDLT Cholesky solver */
		template <typename T>
		SGVector<T> ldlt_solver_impl(
		    const SGMatrix<T>& L, const SGVector<T>& d,
		    const SGVector<index_t>& p, const SGVector<T>& b,
		    const bool lower) const;

		/** Eigen3 cross_entropy method
		 * The cross entropy is defined as \f$ H(P,Q) = - \sum_{ij}
		 * P[i,j]log(Q[i,j]) \f$
		 */
		template <typename T>
		T cross_entropy_impl(const SGMatrix<T>& p, const SGMatrix<T>& q) const;

		/** Eigen3 vector dot-product method */
		template <typename T>
		T dot_impl(const SGVector<T>& a, const SGVector<T>& b) const;

		/** Eigen3 eigenvalues and eigenvectors computation for real matrices.
		 */
		template <typename T>
		void eigen_solver_impl(
		    const SGMatrix<T>& A, SGVector<T>& eigenvalues,
		    SGMatrix<T>& eigenvectors) const;

		/** Eigen3 eigenvalues and eigenvectors computation for complex
		 * matrices. */
		void eigen_solver_impl(
		    const SGMatrix<complex128_t>& A,
		    SGVector<complex128_t>& eigenvalues,
		    SGMatrix<complex128_t>& eigenvectors) const;

		/** Eigen3 eigenvalues and eigenvectors computation of symmetric
		 * matrices */
		template <typename T>
		void eigen_solver_symmetric_impl(
		    const SGMatrix<T>& A, SGVector<T>& eigenvalues,
		    SGMatrix<T>& eigenvectors, index_t k) const;

		/** Eigen3 matrix in-place elementwise product method */
		template <typename T>
		void element_prod_impl(
		    const SGMatrix<T>& a, const SGMatrix<T>& b,
		    SGMatrix<T>& result) const;

		/** Eigen3 matrix block in-place elementwise product method */
		template <typename T>
		void element_prod_impl(
		    const linalg::Block<SGMatrix<T>>& a,
		    const linalg::Block<SGMatrix<T>>& b, SGMatrix<T>& result) const;

		/** Eigen3 vector in-place elementwise product method */
		template <typename T>
		void element_prod_impl(
		    const SGVector<T>& a, const SGVector<T>& b,
		    SGVector<T>& result) const;

		/** Eigen3 vector exponent method */
		template <typename T>
		void exponent_impl(const SGVector<T>& a, SGVector<T>& result) const;

		/** Eigen3 matrix exponent method */
		template <typename T>
		void exponent_impl(const SGMatrix<T>& a, SGMatrix<T>& result) const;

		/** Eigen3 set matrix to identity method */
		template <typename T>
		void identity_impl(SGMatrix<T>& identity_matrix) const;

		/** Eigen3 logistic method. Calculates f(x) = 1/(1+exp(-x)) */
		template <typename T>
		void logistic_impl(const SGMatrix<T>& a, SGMatrix<T>& result) const;

		/** Eigen3 matrix * vector in-place product method */
		template <typename T>
		void matrix_prod_impl(
		    const SGMatrix<T>& a, const SGVector<T>& b, SGVector<T>& result,
		    bool transpose, bool transpose_B = false) const;

		/** Eigen3 matrix in-place product method */
		template <typename T>
		void matrix_prod_impl(
		    const SGMatrix<T>& a, const SGMatrix<T>& b, SGMatrix<T>& result,
		    bool transpose_A, bool transpose_B) const;

		/** Return the largest element in the vector with Eigen3 library */
		template <typename T>
		T max_impl(const SGVector<T>& vec) const;

		/** Return the largest element in the matrix with Eigen3 library */
		template <typename T>
		T max_impl(const SGMatrix<T>& mat) const;

		/** Real eigen3 vector and matrix mean method */
		template <typename T, template <typename> class Container>
		typename std::enable_if<!std::is_same<T, complex128_t>::value,
		                        float64_t>::type
		mean_impl(const Container<T>& a) const;

		/** Complex eigen3 vector and matrix mean method */
		template <template <typename> class Container>
		complex128_t mean_impl(const Container<complex128_t>& a) const;

		/** Eigen3 multiply_by_logistic_derivative method
		 * Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j)) for
		 * all i
		 * and j
		 */
		template <typename T>
		void multiply_by_logistic_derivative_impl(
		    const SGMatrix<T>& a, SGMatrix<T>& result) const;

		/** Eigen3 multiply_by_rectified_linear_derivative method
		 * Performs the operation C(i,j) = C(i,j) * (A(i,j)!=0) for all i and j
		 */
		template <typename T>
		void multiply_by_rectified_linear_derivative_impl(
		    const SGMatrix<T>& a, SGMatrix<T>& result) const;

		/** Eigen3 vector QR solver. */
		template <typename T>
		SGVector<T>
		qr_solver_impl(const SGMatrix<T>& A, const SGVector<T>& b) const;

		/** Eigen3 matrix QR solver. */
		template <typename T>
		SGMatrix<T>
		qr_solver_impl(const SGMatrix<T>& A, const SGMatrix<T> b) const;

		/** Range fill a vector or matrix with start...start+len-1. */
		template <typename T, template <typename> class Container>
		void range_fill_impl(Container<T>& a, const T start) const;

		/** Applies the elementwise rectified linear function f(x) = max(0,x) */
		template <typename T>
		void
		rectified_linear_impl(const SGMatrix<T>& a, SGMatrix<T>& result) const;

		/** Eigen3 vector inplace scale method: result = alpha * A */
		template <typename T>
		void
		scale_impl(const SGVector<T>& a, T alpha, SGVector<T>& result) const;

		/** Eigen3 matrix inplace scale method: result = alpha * A */
		template <typename T>
		void
		scale_impl(const SGMatrix<T>& a, T alpha, SGMatrix<T>& result) const;

		/** Eigen3 set const method */
		template <typename T, template <typename> class Container>
		void set_const_impl(Container<T>& a, T value) const;

		/** Eigen3 softmax method */
		template <typename T, template <typename> class Container>
		void softmax_impl(Container<T>& a) const;

		/** Eigen3 squared error method
		 * The squared error is defined as \f$ E(P,Q) = \frac{1}{2} \sum_{ij}
		 * (P[i,j]-Q[i,j])^2 \f$
		 */
		template <typename T>
		T squared_error_impl(const SGMatrix<T>& p, const SGMatrix<T>& q) const;

		/** Eigen3 vector sum method */
		template <typename T>
		T sum_impl(const SGVector<T>& vec, bool no_diag = false) const;

		/** Eigen3 matrix sum method */
		template <typename T>
		T sum_impl(const SGMatrix<T>& mat, bool no_diag = false) const;

		/** Eigen3 matrix block sum method */
		template <typename T>
		T sum_impl(
		    const linalg::Block<SGMatrix<T>>& mat, bool no_diag = false) const;

		/** Eigen3 symmetric matrix sum method */
		template <typename T>
		T
		sum_symmetric_impl(const SGMatrix<T>& mat, bool no_diag = false) const;

		/** Eigen3 symmetric matrix block sum method */
		template <typename T>
		T sum_symmetric_impl(
		    const linalg::Block<SGMatrix<T>>& mat, bool no_diag = false) const;

		/** Eigen3 matrix colwise sum method */
		template <typename T>
		SGVector<T>
		colwise_sum_impl(const SGMatrix<T>& mat, bool no_diag) const;

		/** Eigen3 matrix block colwise sum method */
		template <typename T>
		SGVector<T> colwise_sum_impl(
		    const linalg::Block<SGMatrix<T>>& mat, bool no_diag) const;

		/** Eigen3 matrix rowwise sum method */
		template <typename T>
		SGVector<T>
		rowwise_sum_impl(const SGMatrix<T>& mat, bool no_diag) const;

		/** Eigen3 matrix block rowwise sum method */
		template <typename T>
		SGVector<T> rowwise_sum_impl(
		    const linalg::Block<SGMatrix<T>>& mat, bool no_diag) const;

		/** Eigen3 compute svd method */
		template <typename T>
		void svd_impl(
		    const SGMatrix<T>& A, SGVector<T>& s, SGMatrix<T>& U, bool thin_U,
		    linalg::SVDAlgorithm alg) const;

		/** Eigen3 compute trace method */
		template <typename T>
		T trace_impl(const SGMatrix<T>& A) const;

		/** Eigen3 transpose matrix method */
		template <typename T>
		SGMatrix<T> transpose_matrix_impl(const SGMatrix<T>& A) const;

		/** Eigen3 triangular solver method */
		template <typename T>
		SGMatrix<T> triangular_solver_impl(
		    const SGMatrix<T>& L, const SGMatrix<T>& b, const bool lower) const;

		/** Eigen3 triangular solver method */
		template <typename T>
		SGVector<T> triangular_solver_impl(
		    const SGMatrix<T>& L, const SGVector<T>& b, const bool lower) const;

		/** Eigen3 set vector to zero method */
		template <typename T>
		void zero_impl(SGVector<T>& a) const;

		/** Eigen3 set matrix to zero method */
		template <typename T>
		void zero_impl(SGMatrix<T>& a) const;
	};

/*
 * Eigen's symmetric eigensolver uses a slower algorithm in comparison
 * to LAPACK's dsyevr, so if LAPACK is available we use it for float64 type.
 * This should be removed if eventually Eigen will provide a faster
 * symmetric eigensolver (@see
 * http://eigen.tuxfamily.org/bz/show_bug.cgi?id=522).
 */
#ifdef HAVE_LAPACK
	template <>
	void LinalgBackendEigen::eigen_solver_symmetric_impl<float64_t>(
	    const SGMatrix<float64_t>& A, SGVector<float64_t>& eigenvalues,
	    SGMatrix<float64_t>& eigenvectors, index_t k) const;
#endif
}

#endif // LINALG_BACKEND_EIGEN_H__
