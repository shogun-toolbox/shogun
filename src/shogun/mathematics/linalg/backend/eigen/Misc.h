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

#ifndef EIGEN_MISC_H
#define EIGEN_MISC_H

template <typename T>
void LinalgBackendEigen::center_matrix(SGMatrix<T>& A, derived_tag) const
{
	index_t n = A.num_cols;
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;

	typename SGVector<T>::EigenVectorXt rows_sum =
	    (A_eig.rowwise().sum()).array() / (T)n;
	typename SGVector<T>::EigenRowVectorXt cols_sum =
	    (A_eig.colwise().sum()).array() / (T)n;
	T m = rows_sum.sum() / (T)n;

	A_eig = ((A_eig.array() + m).matrix().rowwise() - cols_sum).colwise() -
	        rows_sum;
}

template <typename T>
void LinalgBackendEigen::identity(
    SGMatrix<T>& identity_matrix, derived_tag) const
{
	typename SGMatrix<T>::EigenMatrixXtMap I_eig = identity_matrix;
	I_eig.setIdentity();
}

template <typename T>
void LinalgBackendEigen::set_const(
    SGVector<T>& a, T value, derived_tag) const
{
	typename SGVector<T>::EigenVectorXtMap a_eig = a;
	a_eig.setConstant(value);
}

template <typename T>
void LinalgBackendEigen::set_const(
    SGMatrix<T>& a, T value, derived_tag) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	a_eig.setConstant(value);
}

template <typename T, template <typename> class Container>
void LinalgBackendEigen::range_fill(
    Container<T>& a, const T start, derived_tag) const
{
	for (decltype(a.size()) i = 0; i < a.size(); ++i)
		a[i] = start + T(i);
}

template <typename T>
SGMatrix<T> LinalgBackendEigen::transpose_matrix(
    const SGMatrix<T>& A, derived_tag) const
{
	SGMatrix<T> tr(A.num_cols, A.num_rows);
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGMatrix<T>::EigenMatrixXtMap tr_eig = tr;

	tr_eig = A_eig.transpose();
	return tr;
}

template <typename T>
void LinalgBackendEigen::zero(SGVector<T>& a, derived_tag) const
{
	typename SGVector<T>::EigenVectorXtMap a_eig = a;
	a_eig.setZero();
}

template <typename T>
void LinalgBackendEigen::zero(SGMatrix<T>& a, derived_tag) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	a_eig.setZero();
}

#endif