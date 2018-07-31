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

#define BACKEND_GENERIC_MAX(Type, Container)                                   \
	Type LinalgBackendEigen::max(const Container<Type>& a) const               \
	{                                                                          \
		return max_impl(a);                                                    \
	}
DEFINE_FOR_NON_COMPLEX_PTYPE(BACKEND_GENERIC_MAX, SGVector)
DEFINE_FOR_NON_COMPLEX_PTYPE(BACKEND_GENERIC_MAX, SGMatrix)
#undef BACKEND_GENERIC_MAX

#define BACKEND_GENERIC_REAL_MEAN(Type, Container)                             \
	float64_t LinalgBackendEigen::mean(const Container<Type>& a) const         \
	{                                                                          \
		return mean_impl(a);                                                   \
	}
DEFINE_FOR_NON_COMPLEX_PTYPE(BACKEND_GENERIC_REAL_MEAN, SGVector)
DEFINE_FOR_NON_COMPLEX_PTYPE(BACKEND_GENERIC_REAL_MEAN, SGMatrix)
#undef BACKEND_GENERIC_REAL_MEAN

#define BACKEND_GENERIC_COMPLEX_MEAN(Container)                                \
	complex128_t LinalgBackendEigen::mean(const Container<complex128_t>& a)    \
	    const                                                                  \
	{                                                                          \
		return mean_impl(a);                                                   \
	}
BACKEND_GENERIC_COMPLEX_MEAN(SGVector)
BACKEND_GENERIC_COMPLEX_MEAN(SGMatrix)
#undef BACKEND_GENERIC_COMPLEX_MEAN

#define BACKEND_GENERIC_SUM(Type, Container)                                   \
	Type LinalgBackendEigen::sum(const Container<Type>& a, bool no_diag) const \
	{                                                                          \
		return sum_impl(a, no_diag);                                           \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SUM, SGMatrix)
#undef BACKEND_GENERIC_SUM

#define BACKEND_GENERIC_BLOCK_SUM(Type, Container)                             \
	Type LinalgBackendEigen::sum(                                              \
	    const linalg::Block<Container<Type>>& a, bool no_diag) const           \
	{                                                                          \
		return sum_impl(a, no_diag);                                           \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_SUM

#define BACKEND_GENERIC_SYMMETRIC_SUM(Type, Container)                         \
	Type LinalgBackendEigen::sum_symmetric(                                    \
	    const Container<Type>& a, bool no_diag) const                          \
	{                                                                          \
		return sum_symmetric_impl(a, no_diag);                                 \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SYMMETRIC_SUM, SGMatrix)
#undef BACKEND_GENERIC_SYMMETRIC_SUM

#define BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM(Type, Container)                   \
	Type LinalgBackendEigen::sum_symmetric(                                    \
	    const linalg::Block<Container<Type>>& a, bool no_diag) const           \
	{                                                                          \
		return sum_symmetric_impl(a, no_diag);                                 \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM, SGMatrix)
#undef BACKEND_GENERIC_SYMMETRIC_BLOCK_SUM

#define BACKEND_GENERIC_COLWISE_SUM(Type, Container)                           \
	SGVector<Type> LinalgBackendEigen::colwise_sum(                            \
	    const Container<Type>& a, bool no_diag) const                          \
	{                                                                          \
		return colwise_sum_impl(a, no_diag);                                   \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_COLWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_COLWISE_SUM

#define BACKEND_GENERIC_BLOCK_COLWISE_SUM(Type, Container)                     \
	SGVector<Type> LinalgBackendEigen::colwise_sum(                            \
	    const linalg::Block<Container<Type>>& a, bool no_diag) const           \
	{                                                                          \
		return colwise_sum_impl(a, no_diag);                                   \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_COLWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_COLWISE_SUM

#define BACKEND_GENERIC_ROWWISE_SUM(Type, Container)                           \
	SGVector<Type> LinalgBackendEigen::rowwise_sum(                            \
	    const Container<Type>& a, bool no_diag) const                          \
	{                                                                          \
		return rowwise_sum_impl(a, no_diag);                                   \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ROWWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_ROWWISE_SUM

#define BACKEND_GENERIC_BLOCK_ROWWISE_SUM(Type, Container)                     \
	SGVector<Type> LinalgBackendEigen::rowwise_sum(                            \
	    const linalg::Block<Container<Type>>& a, bool no_diag) const           \
	{                                                                          \
		return rowwise_sum_impl(a, no_diag);                                   \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_BLOCK_ROWWISE_SUM, SGMatrix)
#undef BACKEND_GENERIC_BLOCK_ROWWISE_SUM

#define BACKEND_GENERIC_TRACE(Type, Container)                                 \
	Type LinalgBackendEigen::trace(const Container<Type>& A) const             \
	{                                                                          \
		return trace_impl(A);                                                  \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_TRACE, SGMatrix)
#undef BACKEND_GENERIC_TRACE

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_NON_COMPLEX_PTYPE
#undef DEFINE_FOR_NON_INTEGER_PTYPE
#undef DEFINE_FOR_NUMERIC_PTYPE

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
	return sum_impl(a, false) / (float64_t(a.size()));
}

template <template <typename> class Container>
complex128_t
LinalgBackendEigen::mean_impl(const Container<complex128_t>& a) const
{
	return sum_impl(a, false) / (complex128_t(a.size()));
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
T LinalgBackendEigen::sum_impl(
    const linalg::Block<SGMatrix<T>>& mat, bool no_diag) const
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
T LinalgBackendEigen::sum_symmetric_impl(
    const SGMatrix<T>& mat, bool no_diag) const
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
T LinalgBackendEigen::sum_symmetric_impl(
    const linalg::Block<SGMatrix<T>>& mat, bool no_diag) const
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
SGVector<T>
LinalgBackendEigen::colwise_sum_impl(const SGMatrix<T>& mat, bool no_diag) const
{
	SGVector<T> result(mat.num_cols);

	typename SGMatrix<T>::EigenMatrixXtMap mat_eig = mat;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = mat_eig.colwise().sum();

	// remove the main diagonal elements if required
	if (no_diag)
	{
		index_t len_major_diag =
		    mat_eig.rows() < mat_eig.cols() ? mat_eig.rows() : mat_eig.cols();
		for (index_t i = 0; i < len_major_diag; ++i)
			result_eig[i] -= mat_eig(i, i);
	}

	return result;
}

template <typename T>
SGVector<T> LinalgBackendEigen::colwise_sum_impl(
    const linalg::Block<SGMatrix<T>>& mat, bool no_diag) const
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
		index_t len_major_diag =
		    m_block.rows() < m_block.cols() ? m_block.rows() : m_block.cols();
		for (index_t i = 0; i < len_major_diag; ++i)
			result_eig[i] -= m_block(i, i);
	}

	return result;
}

template <typename T>
SGVector<T>
LinalgBackendEigen::rowwise_sum_impl(const SGMatrix<T>& mat, bool no_diag) const
{
	SGVector<T> result(mat.num_rows);

	typename SGMatrix<T>::EigenMatrixXtMap mat_eig = mat;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = mat_eig.rowwise().sum();

	// remove the main diagonal elements if required
	if (no_diag)
	{
		index_t len_major_diag =
		    mat_eig.rows() < mat_eig.cols() ? mat_eig.rows() : mat_eig.cols();
		for (index_t i = 0; i < len_major_diag; ++i)
			result_eig[i] -= mat_eig(i, i);
	}

	return result;
}

template <typename T>
SGVector<T> LinalgBackendEigen::rowwise_sum_impl(
    const linalg::Block<SGMatrix<T>>& mat, bool no_diag) const
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
		index_t len_major_diag =
		    m_block.rows() < m_block.cols() ? m_block.rows() : m_block.cols();
		for (index_t i = 0; i < len_major_diag; ++i)
			result_eig[i] -= m_block(i, i);
	}

	return result;
}

template <typename T>
T LinalgBackendEigen::trace_impl(const SGMatrix<T>& A) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	return A_eig.trace();
}
