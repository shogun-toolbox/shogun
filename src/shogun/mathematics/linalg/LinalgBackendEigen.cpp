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

#define BACKEND_GENERIC_IN_PLACE_ADD(Type, Container) \
void LinalgBackendEigen::add(Container<Type>& a, Container<Type>& b, Type alpha, \
Type beta, Container<Type>& result) const \
{  \
add_impl(a, b, alpha, beta, result); \
}
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_IN_PLACE_ADD, SGVector)
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_IN_PLACE_ADD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_ADD

#define BACKEND_GENERIC_ADD_COL_VEC(Type, Container) \
void LinalgBackendEigen::add_col_vec(const SGMatrix<Type>& A, index_t i, \
	const SGVector<Type>& b, Container<Type>& result, Type alpha, Type beta) const \
{  \
	add_col_vec_impl(A, i, b, result, alpha, beta); \
}
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_COL_VEC, SGVector)
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_ADD_COL_VEC, SGMatrix)
#undef BACKEND_GENERIC_ADD_COL_VEC

#define BACKEND_GENERIC_CHOLESKY_FACTOR(Type, Container) \
Container<Type> LinalgBackendEigen::cholesky_factor(const Container<Type>& A, \
	const bool lower) const \
{  \
	return cholesky_factor_impl(A, lower); \
}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CHOLESKY_FACTOR, SGMatrix)
#undef BACKEND_GENERIC_CHOLESKY_FACTOR

#define BACKEND_GENERIC_CHOLESKY_SOLVER(Type, Container) \
SGVector<Type> LinalgBackendEigen::cholesky_solver(const Container<Type>& L, \
	const SGVector<Type>& b, const bool lower) const \
{  \
	return cholesky_solver_impl(L, b, lower); \
}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CHOLESKY_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_CHOLESKY_SOLVER

#define BACKEND_GENERIC_DOT(Type, Container) \
Type LinalgBackendEigen::dot(const Container<Type>& a, const Container<Type>& b) const \
{  \
	return dot_impl(a, b);  \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_DOT, SGVector)
#undef BACKEND_GENERIC_DOT

#define BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD(Type, Container) \
void LinalgBackendEigen::element_prod(Container<Type>& a, Container<Type>& b,\
	Container<Type>& result) const \
{  \
	element_prod_impl(a, b, result); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_ELEMENT_PROD

#define BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD(Type, Container) \
void LinalgBackendEigen::element_prod(linalg::Block<Container<Type>>& a, \
	linalg::Block<Container<Type>>& b, Container<Type>& result) const \
{  \
	element_prod_impl(a, b, result); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_BLOCK_ELEMENT_PROD

#define BACKEND_GENERIC_IDENTITY(Type, Container) \
void LinalgBackendEigen::identity(Container<Type>& I) const \
{  \
	identity_impl(I); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IDENTITY, SGMatrix)
#undef BACKEND_GENERIC_IDENTITY

#define BACKEND_GENERIC_LOGISTIC(Type, Container) \
void LinalgBackendEigen::logistic(Container<Type>& a, Container<Type>& result) const \
{  \
	logistic_impl(a, result); \
}
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_LOGISTIC, SGMatrix)
#undef BACKEND_GENERIC_LOGISTIC

#define BACKEND_GENERIC_IN_PLACE_MATRIX_PROD(Type, Container) \
void LinalgBackendEigen::matrix_prod(SGMatrix<Type>& a, Container<Type>& b,\
	Container<Type>& result, bool transpose_A, bool transpose_B) const \
{  \
	matrix_prod_impl(a, b, result, transpose_A, transpose_B); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_MATRIX_PROD, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_MATRIX_PROD, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_MATRIX_PROD

#define BACKEND_GENERIC_MAX(Type, Container) \
Type LinalgBackendEigen::max(const Container<Type>& a) const \
{  \
	return max_impl(a); \
}
DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_MAX, SGVector)
DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_MAX, SGMatrix)
#undef BACKEND_GENERIC_MAX

#define BACKEND_GENERIC_REAL_MEAN(Type, Container) \
float64_t LinalgBackendEigen::mean(const Container<Type>& a) const \
{  \
	return mean_impl(a);  \
}
DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_REAL_MEAN, SGVector)
DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_REAL_MEAN, SGMatrix)
#undef BACKEND_GENERIC_REAL_MEAN

#define BACKEND_GENERIC_COMPLEX_MEAN(Container) \
complex128_t LinalgBackendEigen::mean(const Container<complex128_t>& a) const \
{  \
	return mean_impl(a);  \
}
	BACKEND_GENERIC_COMPLEX_MEAN(SGVector)
	BACKEND_GENERIC_COMPLEX_MEAN(SGMatrix)
#undef BACKEND_GENERIC_COMPLEX_MEAN

#define BACKEND_GENERIC_RANGE_FILL(Type, Container) \
void LinalgBackendEigen::range_fill(Container<Type>& a, const Type start) const \
{  \
	range_fill_impl(a, start); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_RANGE_FILL, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_RANGE_FILL, SGMatrix)
#undef BACKEND_GENERIC_RANGE_FILL

#define BACKEND_GENERIC_IN_PLACE_SCALE(Type, Container) \
void LinalgBackendEigen::scale(Container<Type>& a, Type alpha, Container<Type>& result) const \
{  \
	scale_impl(a, alpha, result); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_SCALE, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IN_PLACE_SCALE, SGMatrix)
#undef BACKEND_GENERIC_IN_PLACE_SCALE

#define BACKEND_GENERIC_SET_CONST(Type, Container) \
void LinalgBackendEigen::set_const(Container<Type>& a, const Type value) const \
{  \
	set_const_impl(a, value); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SET_CONST, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SET_CONST, SGMatrix)
#undef BACKEND_GENERIC_SET_CONST

#define BACKEND_GENERIC_SUM(Type, Container) \
Type LinalgBackendEigen::sum(const Container<Type>& a, bool no_diag) const \
{  \
	return sum_impl(a, no_diag);  \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGMatrix)
#undef BACKEND_GENERIC_SUM

#define BACKEND_GENERIC_BLOCK_SUM(Type, Container) \
Type LinalgBackendEigen::sum(const linalg::Block<Container<Type>>& a, bool no_diag) const \
{  \
	return sum_impl(a, no_diag); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_SUM

#define BACKEND_GENERIC_SYMMETRIC_SUM(Type, Container) \
Type LinalgBackendEigen::sum_symmetric(const Container<Type>& a, bool no_diag) const \
{  \
	return sum_symmetric_impl(a, no_diag); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SYMMETRIC_SUM, SGMatrix)
#undef BACKEND_GENERIC_SYMMETRIC_SUM

#define BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM(Type, Container) \
Type LinalgBackendEigen::sum_symmetric(const linalg::Block<Container<Type>>& a, bool no_diag) const \
{  \
	return sum_symmetric_impl(a, no_diag); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM, SGMatrix)
#undef BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM

#define BACKEND_GENERIC_COLWISE_SUM(Type, Container) \
SGVector<Type> LinalgBackendEigen::colwise_sum(const Container<Type>& a, bool no_diag) const \
{  \
	return colwise_sum_impl(a, no_diag); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_COLWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_COLWISE_SUM

#define BACKEND_GENERIC_BLOCK_COLWISE_SUM(Type, Container) \
SGVector<Type> LinalgBackendEigen::colwise_sum(const linalg::Block<Container<Type>>& a, bool no_diag) const \
{  \
	return colwise_sum_impl(a, no_diag); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_COLWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_COLWISE_SUM

#define BACKEND_GENERIC_ROWWISE_SUM(Type, Container) \
SGVector<Type> LinalgBackendEigen::rowwise_sum(const Container<Type>& a, bool no_diag) const \
{  \
	return rowwise_sum_impl(a, no_diag); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ROWWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_ROWWISE_SUM

#define BACKEND_GENERIC_BLOCK_ROWWISE_SUM(Type, Container) \
SGVector<Type> LinalgBackendEigen::rowwise_sum(const linalg::Block<Container<Type>>& a, bool no_diag) const \
{  \
	return rowwise_sum_impl(a, no_diag); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_ROWWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_ROWWISE_SUM

#define BACKEND_GENERIC_TRACE(Type, Container) \
Type LinalgBackendEigen::trace(const Container<Type>& A) const \
{  \
	return trace_impl(A); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_TRACE, SGMatrix)
#undef BACKEND_GENERIC_TRACE

#define BACKEND_GENERIC_ZERO(Type, Container) \
void LinalgBackendEigen::zero(Container<Type>& a) const \
{  \
	zero_impl(a); \
}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ZERO, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ZERO, SGMatrix)
#undef BACKEND_GENERIC_ZERO

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_REAL_PTYPE
#undef DEFINE_FOR_NON_INTEGER_PTYPE
#undef DEFINE_FOR_NUMERIC_PTYPE

template <typename T>
void LinalgBackendEigen::add_impl(SGVector<T>& a, SGVector<T>& b, T alpha, T beta, SGVector<T>& result) const
{
	typename SGVector<T>::EigenVectorXtMap a_eig = a;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = alpha * a_eig + beta * b_eig;
}

template <typename T>
void LinalgBackendEigen::add_impl(SGMatrix<T>& a, SGMatrix<T>& b, T alpha, T beta, SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap b_eig = b;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig = alpha * a_eig + beta * b_eig;
}

template <typename T>
void LinalgBackendEigen::add_col_vec_impl(const SGMatrix<T>& A, index_t i, const SGVector<T>& b,
                      SGMatrix<T>& result, T alpha, T beta) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig.col(i) = alpha * A_eig.col(i) + beta * b_eig;
}

template <typename T>
void LinalgBackendEigen::add_col_vec_impl(const SGMatrix<T>& A, index_t i, const SGVector<T>& b,
                      SGVector<T>& result, T alpha, T beta) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = alpha * A_eig.col(i) + beta * b_eig;
}

template <typename T>
SGMatrix<T> LinalgBackendEigen::cholesky_factor_impl(const SGMatrix<T>& A, const bool lower) const
{
	SGMatrix<T> c(A.num_rows, A.num_cols);
	set_const_impl<T>(c, 0);
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGMatrix<T>::EigenMatrixXtMap c_eig = c;

	Eigen::LLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > llt(A_eig);

	//compute matrix L or U
	if(lower==false)
		c_eig = llt.matrixU();
	else
		c_eig = llt.matrixL();

	/*
	 * checking for success
	 *
	 * 0: Eigen::Success. Decomposition was successful
	 * 1: Eigen::NumericalIssue. The provided data did not satisfy the prerequisites.
	 */
	REQUIRE(llt.info()!=Eigen::NumericalIssue, "Matrix is not Hermitian positive definite!\n");

	return c;
}

template <typename T>
SGVector<T> LinalgBackendEigen::cholesky_solver_impl(const SGMatrix<T>& L, const SGVector<T>& b,
                                 const bool lower) const
{
	SGVector<T> x(b.size());
	set_const_impl<T>(x, 0);
	typename SGMatrix<T>::EigenMatrixXtMap L_eig = L;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGVector<T>::EigenVectorXtMap x_eig = x;

	if (lower == false)
	{
		Eigen::TriangularView<Eigen::Map<typename SGMatrix<T>::EigenMatrixXt,
			0, Eigen::Stride<0,0> >, Eigen::Upper> tlv(L_eig);

		x_eig = (tlv.transpose()).solve(tlv.solve(b_eig));
	}
	else
	{
		Eigen::TriangularView<Eigen::Map<typename SGMatrix<T>::EigenMatrixXt,
			0, Eigen::Stride<0,0> >, Eigen::Lower> tlv(L_eig);
		x_eig = (tlv.transpose()).solve(tlv.solve(b_eig));
	}

	return x;
}

template <typename T>
T LinalgBackendEigen::dot_impl(const SGVector<T>& a, const SGVector<T>& b) const
{
	return (typename SGVector<T>::EigenVectorXtMap(a)).dot(typename SGVector<T>::EigenVectorXtMap(b));
}

template <typename T>
void LinalgBackendEigen::element_prod_impl(SGMatrix<T>& a, SGMatrix<T>& b, SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap b_eig = b;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig = a_eig.array() * b_eig.array();
}

template <typename T>
void LinalgBackendEigen::identity_impl(SGMatrix<T>& I) const
{
	typename SGMatrix<T>::EigenMatrixXtMap I_eig = I;
	I_eig.setIdentity();
}

template <typename T>
void LinalgBackendEigen::logistic_impl(SGMatrix<T>& a, SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig = (T)1 / (1 + ((-1 * a_eig).array()).exp());
}

template <typename T>
void LinalgBackendEigen::element_prod_impl(linalg::Block<SGMatrix<T>>& a,
                       linalg::Block<SGMatrix<T>>& b, SGMatrix<T>& result) const
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
void LinalgBackendEigen::matrix_prod_impl(SGMatrix<T>& a, SGVector<T>& b, SGVector<T>& result,
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
void LinalgBackendEigen::matrix_prod_impl(SGMatrix<T>& a, SGMatrix<T>& b, SGMatrix<T>& result,
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
T LinalgBackendEigen::max_impl(const SGVector<T>& vec) const
{
	return (typename SGVector<T>::EigenVectorXtMap(vec)).maxCoeff();
}

template <typename T>
T LinalgBackendEigen::max_impl(const SGMatrix<T>& mat) const
{
	return (typename SGMatrix<T>::EigenMatrixXtMap(mat)).maxCoeff();
}

template <typename T, template <typename> class Container>
typename std::enable_if<!std::is_same<T, complex128_t>::value, float64_t>::type
LinalgBackendEigen::mean_impl(const Container<T>& a) const
{
	return sum_impl(a)/(float64_t(a.size()));
}

template<template <typename> class Container>
complex128_t LinalgBackendEigen::mean_impl(const Container<complex128_t>& a) const
{
	return sum_impl(a)/(complex128_t(a.size()));
}

template <typename T, template <typename> class Container>
void LinalgBackendEigen::range_fill_impl(Container<T>& a, const T start) const
{
	for (decltype(a.size()) i = 0; i < a.size(); ++i)
		a[i] = start + T(i);
}

template <typename T>
void LinalgBackendEigen::scale_impl(SGVector<T>& a, T alpha, SGVector<T>& result) const
{
	typename SGVector<T>::EigenVectorXtMap a_eig = a;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = alpha * a_eig;
}

template <typename T>
void LinalgBackendEigen::scale_impl(SGMatrix<T>& a, T alpha, SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig = alpha * a_eig;
}

template <typename T, template <typename> class Container>
void LinalgBackendEigen::set_const_impl(Container<T>& a, T value) const
{
	for (decltype(a.size()) i = 0; i < a.size(); ++i)
		a[i] = value;
}

template <typename T>
T LinalgBackendEigen::sum_impl(const SGVector<T>& vec, bool no_diag) const
{
	return (typename SGVector<T>::EigenVectorXtMap(vec)).sum();
}

template <typename T>
T LinalgBackendEigen::sum_impl(const SGMatrix<T>& mat, bool no_diag) const
{
	typename SGMatrix<T>::EigenMatrixXtMap m = mat;
	T result = m.sum();
	if (no_diag)
		result -= m.diagonal().sum();

	return result;
}

template <typename T>
T LinalgBackendEigen::sum_impl(const linalg::Block<SGMatrix<T>>& mat, bool no_diag) const
{
	typename SGMatrix<T>::EigenMatrixXtMap m = mat.m_matrix;
	Eigen::Block<typename SGMatrix<T>::EigenMatrixXtMap> m_block = m.block(
		mat.m_row_begin, mat.m_col_begin, mat.m_row_size, mat.m_col_size);

	T result = m_block.sum();
	if (no_diag)
		result -= m_block.diagonal().sum();

	return result;
}

template <typename T>
T LinalgBackendEigen::sum_symmetric_impl(const SGMatrix<T>& mat, bool no_diag) const
{
	typename SGMatrix<T>::EigenMatrixXtMap m = mat;

	// since the matrix is symmetric with main diagonal inside, we can save half
	// the computation with using only the upper triangular part.
	typename SGMatrix<T>::EigenMatrixXt m_upper =
		m.template triangularView<Eigen::StrictlyUpper>();
	T result = m_upper.sum();
	result += result;

	if (!no_diag)
		result += m.diagonal().sum();
	return result;
}

template <typename T>
T LinalgBackendEigen::sum_symmetric_impl(const linalg::Block<SGMatrix<T>>& mat, bool no_diag) const
{
	typename SGMatrix<T>::EigenMatrixXtMap m = mat.m_matrix;
	Eigen::Block<typename SGMatrix<T>::EigenMatrixXtMap> m_block = m.block(
		mat.m_row_begin, mat.m_col_begin, mat.m_row_size, mat.m_col_size);

	// since the matrix is symmetric with main diagonal inside, we can save half
	// the computation with using only the upper triangular part.
	typename SGMatrix<T>::EigenMatrixXt m_upper =
		m_block.template triangularView<Eigen::StrictlyUpper>();
	T result = m_upper.sum();
	result += result;

	if (!no_diag)
		result += m_block.diagonal().sum();
	return result;
}

template <typename T>
SGVector<T> LinalgBackendEigen::colwise_sum_impl(const SGMatrix<T>& mat, bool no_diag) const
{
	SGVector<T> result(mat.num_cols);

	typename SGMatrix<T>::EigenMatrixXtMap mat_eig = mat;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = mat_eig.colwise().sum();

	// remove the main diagonal elements if required
	if (no_diag)
	{
		index_t len_major_diag = mat_eig.rows() < mat_eig.cols()
		                         ? mat_eig.rows() : mat_eig.cols();
		for (index_t i = 0; i < len_major_diag; ++i)
			result_eig[i] -= mat_eig(i,i);
	}

	return result;
}

template <typename T>
SGVector<T> LinalgBackendEigen::colwise_sum_impl(const linalg::Block<SGMatrix<T>>& mat, bool no_diag) const
{
	SGVector<T> result(mat.m_col_size);

	typename SGMatrix<T>::EigenMatrixXtMap m = mat.m_matrix;
	Eigen::Block<typename SGMatrix<T>::EigenMatrixXtMap> m_block = m.block(
		mat.m_row_begin, mat.m_col_begin, mat.m_row_size, mat.m_col_size);
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = m_block.colwise().sum();

	// remove the main diagonal elements if required
	if (no_diag)
	{
		index_t len_major_diag = m_block.rows() < m_block.cols()
		                         ? m_block.rows() : m_block.cols();
		for (index_t i = 0; i < len_major_diag; ++i)
			result_eig[i] -= m_block(i,i);
	}

	return result;
}

template <typename T>
SGVector<T> LinalgBackendEigen::rowwise_sum_impl(const SGMatrix<T>& mat, bool no_diag) const
{
	SGVector<T> result(mat.num_rows);

	typename SGMatrix<T>::EigenMatrixXtMap mat_eig = mat;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = mat_eig.rowwise().sum();

	// remove the main diagonal elements if required
	if (no_diag)
	{
		index_t len_major_diag = mat_eig.rows() < mat_eig.cols()
		                         ? mat_eig.rows() : mat_eig.cols();
		for (index_t i = 0; i < len_major_diag; ++i)
			result_eig[i] -= mat_eig(i,i);
	}

	return result;
}

template <typename T>
SGVector<T> LinalgBackendEigen::rowwise_sum_impl(const linalg::Block<SGMatrix<T>>& mat, bool no_diag) const
{
	SGVector<T> result(mat.m_row_size);

	typename SGMatrix<T>::EigenMatrixXtMap m = mat.m_matrix;
	Eigen::Block<typename SGMatrix<T>::EigenMatrixXtMap> m_block = m.block(
		mat.m_row_begin, mat.m_col_begin, mat.m_row_size, mat.m_col_size);
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = m_block.rowwise().sum();

	// remove the main diagonal elements if required
	if (no_diag)
	{
		index_t len_major_diag = m_block.rows() < m_block.cols()
		                         ? m_block.rows() : m_block.cols();
		for (index_t i = 0; i < len_major_diag; ++i)
			result_eig[i] -= m_block(i,i);
	}

	return result;
}

template <typename T>
T LinalgBackendEigen::trace_impl(const SGMatrix<T>& A) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	return A_eig.trace();
}

template <typename T>
void LinalgBackendEigen::zero_impl(SGVector<T>& a) const
{
	typename SGVector<T>::EigenVectorXtMap a_eig = a;
	a_eig.setZero();
}

template <typename T>
void LinalgBackendEigen::zero_impl(SGMatrix<T>& a) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	a_eig.setZero();
}
