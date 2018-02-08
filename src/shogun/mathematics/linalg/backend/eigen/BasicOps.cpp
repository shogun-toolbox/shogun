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

#include <shogun/mathematics/linalg/LinalgBackendEigen.h>
#include <shogun/mathematics/linalg/LinalgMacros.h>

using namespace shogun;

#define BACKEND_GENERIC_IN_PLACE_ADD(Type, Container)                          \
	void LinalgBackendEigen::add(                                              \
	    const Container<Type>& a, const Container<Type>& b, Type alpha,        \
	    Type beta, Container<Type>& result) const                              \
	{                                                                          \
		add_impl(a, b, alpha, beta, result);                                   \
	}
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_IN_PLACE_ADD, SGVector)
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_IN_PLACE_ADD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_ADD

#define BACKEND_GENERIC_ADD_COL_VEC(Type, Container)                           \
	void LinalgBackendEigen::add_col_vec(                                      \
	    const SGMatrix<Type>& A, index_t i, const SGVector<Type>& b,           \
	    Container<Type>& result, Type alpha, Type beta) const                  \
	{                                                                          \
		add_col_vec_impl(A, i, b, result, alpha, beta);                        \
	}
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_COL_VEC, SGVector)
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_COL_VEC, SGMatrix)
#undef BACKEND_GENERIC_ADD_COL_VEC

#define BACKEND_GENERIC_ADD_DIAG(Type, Container)                              \
	void LinalgBackendEigen::add_diag(                                         \
	    SGMatrix<Type>& A, const SGVector<Type>& b, Type alpha, Type beta)     \
	    const                                                                  \
	{                                                                          \
		add_diag_impl(A, b, alpha, beta);                                      \
	}
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_DIAG, SGMatrix)
#undef BACKEND_GENERIC_ADD_DIAG

#define BACKEND_GENERIC_ADD_VECTOR(Type, Container)                            \
	void LinalgBackendEigen::add_vector(                                       \
	    const SGMatrix<Type>& A, const SGVector<Type>& b,                      \
	    SGMatrix<Type>& result, Type alpha, Type beta) const                   \
	{                                                                          \
		add_vector_impl(A, b, result, alpha, beta);                            \
	}
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_VECTOR, SGMatrix)
#undef BACKEND_GENERIC_ADD_VECTOR

#define BACKEND_GENERIC_ADD_SCALAR(Type, Container)                            \
	void LinalgBackendEigen::add_scalar(Container<Type>& a, Type b) const      \
	{                                                                          \
		add_scalar_impl(a, b);                                                 \
	}
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_SCALAR, SGVector)
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_SCALAR, SGMatrix)
#undef BACKEND_GENERIC_ADD_SCALAR

#define BACKEND_GENERIC_DOT(Type, Container)                                   \
	Type LinalgBackendEigen::dot(                                              \
	    const Container<Type>& a, const Container<Type>& b) const              \
	{                                                                          \
		return dot_impl(a, b);                                                 \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_DOT, SGVector)
#undef BACKEND_GENERIC_DOT

#define BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD(Type, Container)                 \
	void LinalgBackendEigen::element_prod(                                     \
	    const Container<Type>& a, const Container<Type>& b,                    \
	    Container<Type>& result) const                                         \
	{                                                                          \
		element_prod_impl(a, b, result);                                       \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD, SGMatrix)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD, SGVector)
#undef BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD

#define BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD(Type, Container)           \
	void LinalgBackendEigen::element_prod(                                     \
	    const linalg::Block<Container<Type>>& a,                               \
	    const linalg::Block<Container<Type>>& b, Container<Type>& result)      \
	    const                                                                  \
	{                                                                          \
		element_prod_impl(a, b, result);                                       \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD

#define BACKEND_GENERIC_EXPONENT(Type, Container)                              \
	void LinalgBackendEigen::exponent(                                         \
	    const Container<Type>& a, Container<Type>& result) const               \
	{                                                                          \
		exponent_impl(a, result);                                              \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_EXPONENT, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_EXPONENT, SGMatrix)
#undef BACKEND_GENERIC_EXPONENT

#define BACKEND_GENERIC_IN_PLACE_MATRIX_PROD(Type, Container)                  \
	void LinalgBackendEigen::matrix_prod(                                      \
	    const SGMatrix<Type>& a, const Container<Type>& b,                     \
	    Container<Type>& result, bool transpose_A, bool transpose_B) const     \
	{                                                                          \
		matrix_prod_impl(a, b, result, transpose_A, transpose_B);              \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_MATRIX_PROD, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_MATRIX_PROD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_MATRIX_PROD

#define BACKEND_GENERIC_IN_PLACE_SCALE(Type, Container)                        \
	void LinalgBackendEigen::scale(                                            \
	    const Container<Type>& a, Type alpha, Container<Type>& result) const   \
	{                                                                          \
		scale_impl(a, alpha, result);                                          \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_SCALE, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_SCALE, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_SCALE

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_REAL_PTYPE
#undef DEFINE_FOR_NON_INTEGER_PTYPE
#undef DEFINE_FOR_NUMERIC_PTYPE

template <typename T>
void LinalgBackendEigen::add_impl(
    const SGVector<T>& a, const SGVector<T>& b, T alpha, T beta,
    SGVector<T>& result) const
{
	typename SGVector<T>::EigenVectorXtMap a_eig = a;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = alpha * a_eig + beta * b_eig;
}

template <typename T>
void LinalgBackendEigen::add_impl(
    const SGMatrix<T>& a, const SGMatrix<T>& b, T alpha, T beta,
    SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap b_eig = b;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig = alpha * a_eig + beta * b_eig;
}

template <typename T>
void LinalgBackendEigen::add_col_vec_impl(
    const SGMatrix<T>& A, index_t i, const SGVector<T>& b, SGMatrix<T>& result,
    T alpha, T beta) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig.col(i) = alpha * A_eig.col(i) + beta * b_eig;
}

template <typename T>
void LinalgBackendEigen::add_diag_impl(
    SGMatrix<T>& A, const SGVector<T>& b, T alpha, T beta) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;

	A_eig.diagonal() = alpha * A_eig.diagonal() + beta * b_eig;
}

template <typename T>
void LinalgBackendEigen::add_col_vec_impl(
    const SGMatrix<T>& A, index_t i, const SGVector<T>& b, SGVector<T>& result,
    T alpha, T beta) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = alpha * A_eig.col(i) + beta * b_eig;
}

template <typename T>
void LinalgBackendEigen::add_vector_impl(
    const SGMatrix<T>& A, const SGVector<T>& b, SGMatrix<T>& result, T alpha,
    T beta) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig = (alpha * A_eig).colwise() + beta * b_eig;
}

template <typename T>
void LinalgBackendEigen::add_scalar_impl(SGVector<T>& a, T b) const
{
	typename SGVector<T>::EigenVectorXtMap a_eig = a;
	a_eig = a_eig.array() + b;
}

template <typename T>
void LinalgBackendEigen::add_scalar_impl(SGMatrix<T>& a, T b) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	a_eig = a_eig.array() + b;
}

template <typename T>
T LinalgBackendEigen::dot_impl(const SGVector<T>& a, const SGVector<T>& b) const
{
	return (typename SGVector<T>::EigenVectorXtMap(a))
	    .dot(typename SGVector<T>::EigenVectorXtMap(b));
}

template <typename T>
void LinalgBackendEigen::element_prod_impl(
    const SGMatrix<T>& a, const SGMatrix<T>& b, SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap b_eig = b;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig = a_eig.array() * b_eig.array();
}

template <typename T>
void LinalgBackendEigen::element_prod_impl(
    const linalg::Block<SGMatrix<T>>& a, const linalg::Block<SGMatrix<T>>& b,
    SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a.m_matrix;
	typename SGMatrix<T>::EigenMatrixXtMap b_eig = b.m_matrix;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	Eigen::Block<typename SGMatrix<T>::EigenMatrixXtMap> a_block =
	    a_eig.block(a.m_row_begin, a.m_col_begin, a.m_row_size, a.m_col_size);
	Eigen::Block<typename SGMatrix<T>::EigenMatrixXtMap> b_block =
	    b_eig.block(b.m_row_begin, b.m_col_begin, b.m_row_size, b.m_col_size);

	result_eig = a_block.array() * b_block.array();
}

template <typename T>
void LinalgBackendEigen::element_prod_impl(
    const SGVector<T>& a, const SGVector<T>& b, SGVector<T>& result) const
{
	typename SGVector<T>::EigenVectorXtMap a_eig = a;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = a_eig.array() * b_eig.array();
}

template <typename T>
void LinalgBackendEigen::exponent_impl(
    const SGVector<T>& a, SGVector<T>& result) const
{
	typename SGVector<T>::EigenVectorXtMap a_eig = a;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;
	result_eig = a_eig.array().exp();
}

template <typename T>
void LinalgBackendEigen::exponent_impl(
    const SGMatrix<T>& a, SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;
	result_eig = a_eig.array().exp();
}

template <typename T>
void LinalgBackendEigen::matrix_prod_impl(
    const SGMatrix<T>& a, const SGVector<T>& b, SGVector<T>& result,
    bool transpose, bool transpose_B) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	if (transpose)
		result_eig = a_eig.transpose() * b_eig;
	else
		result_eig = a_eig * b_eig;
}

template <typename T>
void LinalgBackendEigen::matrix_prod_impl(
    const SGMatrix<T>& a, const SGMatrix<T>& b, SGMatrix<T>& result,
    bool transpose_A, bool transpose_B) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap b_eig = b;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	if (transpose_A && transpose_B)
		result_eig = a_eig.transpose() * b_eig.transpose();

	else if (transpose_A)
		result_eig = a_eig.transpose() * b_eig;

	else if (transpose_B)
		result_eig = a_eig * b_eig.transpose();

	else
		result_eig = a_eig * b_eig;
}

template <typename T>
void LinalgBackendEigen::scale_impl(
    const SGVector<T>& a, T alpha, SGVector<T>& result) const
{
	typename SGVector<T>::EigenVectorXtMap a_eig = a;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = alpha * a_eig;
}

template <typename T>
void LinalgBackendEigen::scale_impl(
    const SGMatrix<T>& a, T alpha, SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig = alpha * a_eig;
}
