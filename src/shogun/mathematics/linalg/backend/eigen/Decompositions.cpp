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
#include <shogun/mathematics/linalg/LinalgEnums.h>
#include <shogun/mathematics/linalg/LinalgMacros.h>

using namespace shogun;

#define BACKEND_GENERIC_CHOLESKY_FACTOR(Type, Container)                       \
	Container<Type> LinalgBackendEigen::cholesky_factor(                       \
	    const Container<Type>& A, const bool lower) const                      \
	{                                                                          \
		return cholesky_factor_impl(A, lower);                                 \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CHOLESKY_FACTOR, SGMatrix)
#undef BACKEND_GENERIC_CHOLESKY_FACTOR

#define BACKEND_GENERIC_CHOLESKY_RANK_UPDATE(Type, Container)                  \
	void LinalgBackendEigen::cholesky_rank_update(                             \
	    Container<Type>& L, const SGVector<Type>& b, Type alpha,               \
	    const bool lower) const                                                \
	{                                                                          \
		cholesky_rank_update_impl(L, b, alpha, lower);                         \
	}
DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_CHOLESKY_RANK_UPDATE, SGMatrix)
#undef BACKEND_GENERIC_CHOLESKY_RANK_UPDATE

#define BACKEND_GENERIC_LDLT_FACTOR(Type, Container)                           \
	void LinalgBackendEigen::ldlt_factor(                                      \
	    const Container<Type>& A, Container<Type>& L, SGVector<Type>& d,       \
	    SGVector<index_t>& p, const bool lower) const                          \
	{                                                                          \
		return ldlt_factor_impl(A, L, d, p, lower);                            \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_LDLT_FACTOR, SGMatrix)
#undef BACKEND_GENERIC_LDLT_FACTOR

#define BACKEND_GENERIC_SVD(Type, Container)                                   \
	void LinalgBackendEigen::svd(                                              \
	    const Container<Type>& A, SGVector<Type> s, Container<Type> U,         \
	    bool thin_U, linalg::SVDAlgorithm alg) const                           \
	{                                                                          \
		return svd_impl(A, s, U, thin_U, alg);                                 \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_SVD, SGMatrix)
#undef BACKEND_GENERIC_SVD

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_NON_COMPLEX_PTYPE
#undef DEFINE_FOR_NON_INTEGER_PTYPE
#undef DEFINE_FOR_NUMERIC_PTYPE

template <typename T>
SGMatrix<T> LinalgBackendEigen::cholesky_factor_impl(
    const SGMatrix<T>& A, const bool lower) const
{
	SGMatrix<T> c(A.num_rows, A.num_cols);
	set_const(c, 0);
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGMatrix<T>::EigenMatrixXtMap c_eig = c;

	Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> llt(A_eig);

	// compute matrix L or U
	if (lower == false)
		c_eig = llt.matrixU();
	else
		c_eig = llt.matrixL();

	/*
	 * checking for success
	 *
	 * 0: Eigen::Success. Decomposition was successful
	 * 1: Eigen::NumericalIssue. The provided data did not satisfy the
	 * prerequisites.
	 */
	REQUIRE(
	    llt.info() != Eigen::NumericalIssue,
	    "Matrix is not Hermitian positive definite!\n");

	return c;
}

#include <iostream>

template <typename T>
void LinalgBackendEigen::cholesky_rank_update_impl(
    SGMatrix<T>& L, const SGVector<T>& b, T alpha, bool lower) const
{
	typename SGMatrix<T>::EigenMatrixXtMap L_eig = L;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;

	if (lower == false)
	{
		auto U = L_eig.transpose();
		Eigen::internal::llt_rank_update_lower(U, b_eig, alpha);
	}
	else
		Eigen::internal::llt_rank_update_lower(L_eig, b_eig, alpha);
}

template <typename T>
void LinalgBackendEigen::ldlt_factor_impl(
    const SGMatrix<T>& A, SGMatrix<T>& L, SGVector<T>& d, SGVector<index_t>& p,
    const bool lower) const
{
	set_const(L, 0);
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGMatrix<T>::EigenMatrixXtMap L_eig = L;
	typename SGVector<T>::EigenVectorXtMap d_eig = d;
	typename SGVector<index_t>::EigenVectorXtMap p_eig = p;

	Eigen::LDLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ldlt(A_eig);

	d_eig = ldlt.vectorD().template cast<T>();
	if (lower)
		L_eig = ldlt.matrixL();
	else
		L_eig = ldlt.matrixU();

	// flatten N*1 matrix into vector
	p_eig = ldlt.transpositionsP().indices().template cast<index_t>();

	REQUIRE(
	    ldlt.info() != Eigen::NumericalIssue,
	    "The factorization failed because of a zero pivot.\n");
}

template <typename T>
void LinalgBackendEigen::svd_impl(
    const SGMatrix<T>& A, SGVector<T>& s, SGMatrix<T>& U, bool thin_U,
    linalg::SVDAlgorithm alg) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGVector<T>::EigenVectorXtMap s_eig = s;
	typename SGMatrix<T>::EigenMatrixXtMap U_eig = U;

	switch (alg)
	{
	case linalg::SVDAlgorithm::BidiagonalDivideConquer:
	{
// Building BDC-SVD templates OOMs on 32 Bit ARM hardware
#if	(defined(__arm__) || defined (__thumb__) || defined(__TARGET_ARCH_ARM) ||	\
	 defined(__TARGET_ARCH_THUMB) || defined (_ARM) || defined(_M_ARM) ||		\
	 defined(_M_ARMT) || defined(__arm)) && !defined(__aarch64__)
		SG_SWARNING(
		    "BDC-SVD is not supported on 32 Bit ARM hardware.\n"
		    "Falling back on Jacobi-SVD.\n")
#elif EIGEN_VERSION_AT_LEAST(3, 3, 0)
		auto svd_eig =
		    A_eig.bdcSvd(thin_U ? Eigen::ComputeThinU : Eigen::ComputeFullU);
		s_eig = svd_eig.singularValues().template cast<T>();
		U_eig = svd_eig.matrixU().template cast<T>();
		break;
#else
		SG_SWARNING(
		    "At least Eigen 3.3 is required for BDC-SVD.\n"
		    "Falling back on Jacobi-SVD.\n")
#endif
	}

	case linalg::SVDAlgorithm::Jacobi:
	{
		auto svd_eig =
		    A_eig.jacobiSvd(thin_U ? Eigen::ComputeThinU : Eigen::ComputeFullU);
		s_eig = svd_eig.singularValues().template cast<T>();
		U_eig = svd_eig.matrixU().template cast<T>();
		break;
	}
	}
}
