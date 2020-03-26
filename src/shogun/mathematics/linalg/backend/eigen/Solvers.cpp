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

#include <cmath>
#include <shogun/base/range.h>
#include <shogun/mathematics/linalg/LinalgBackendEigen.h>
#include <shogun/mathematics/linalg/LinalgMacros.h>

using namespace shogun;

#define BACKEND_GENERIC_CHOLESKY_SOLVER(Type, Container)                       \
	SGVector<Type> LinalgBackendEigen::cholesky_solver(                        \
	    const Container<Type>& L, const SGVector<Type>& b, const bool lower)   \
	    const                                                                  \
	{                                                                          \
		return cholesky_solver_impl(L, b, lower);                              \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CHOLESKY_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_CHOLESKY_SOLVER

#define BACKEND_GENERIC_LDLT_SOLVER(Type, Container)                           \
	SGVector<Type> LinalgBackendEigen::ldlt_solver(                            \
	    const Container<Type>& L, const SGVector<Type>& d,                     \
	    const SGVector<index_t>& p, const SGVector<Type>& b, const bool lower) \
	    const                                                                  \
	{                                                                          \
		return ldlt_solver_impl(L, d, p, b, lower);                            \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_LDLT_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_LDLT_SOLVER

#define BACKEND_GENERIC_QR_SOLVER(Type, Container)                             \
	Container<Type> LinalgBackendEigen::qr_solver(                             \
	    const SGMatrix<Type>& A, const Container<Type>& b) const               \
	{                                                                          \
		return qr_solver_impl(A, b);                                           \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_QR_SOLVER, SGVector)
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_QR_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_QR_SOLVER

#define BACKEND_GENERIC_TRIANGULAR_SOLVER(Type, Container)                     \
	Container<Type> LinalgBackendEigen::triangular_solver(                     \
	    const SGMatrix<Type>& L, const Container<Type>& b, const bool lower)   \
	    const                                                                  \
	{                                                                          \
		return triangular_solver_impl(L, b, lower);                            \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_TRIANGULAR_SOLVER, SGVector)
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_TRIANGULAR_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_TRIANGULAR_SOLVER

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_NON_COMPLEX_PTYPE
#undef DEFINE_FOR_NON_INTEGER_PTYPE
#undef DEFINE_FOR_NUMERIC_PTYPE
#undef DEFINE_FOR_ALL_PTYPE_EXCEPT_FLOAT64

template <typename T>
SGVector<T> LinalgBackendEigen::cholesky_solver_impl(
    const SGMatrix<T>& L, const SGVector<T>& b, const bool lower) const
{
	SGVector<T> x(b.size());
	set_const(x, 0);
	typename SGMatrix<T>::EigenMatrixXtMap L_eig = L;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGVector<T>::EigenVectorXtMap x_eig = x;

	if (lower == false)
	{
		Eigen::TriangularView<typename SGMatrix<T>::EigenMatrixXtMap,
		                      Eigen::Upper>
		    tlv(L_eig);

		x_eig = (tlv.transpose()).solve(tlv.solve(b_eig));
	}
	else
	{

		Eigen::TriangularView<typename SGMatrix<T>::EigenMatrixXtMap,
		                      Eigen::Lower>
		    tlv(L_eig);
		x_eig = (tlv.transpose()).solve(tlv.solve(b_eig));
	}

	return x;
}

template <typename T>
SGVector<T> LinalgBackendEigen::ldlt_solver_impl(
    const SGMatrix<T>& L, const SGVector<T>& d, const SGVector<index_t>& p,
    const SGVector<T>& b, const bool lower) const
{
	SGVector<T> result(b.vlen);
	set_const(result, 0);

	typename SGMatrix<T>::EigenMatrixXtMap L_eig = L;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;
	typename SGVector<index_t>::EigenVectorXtMap p_eig = p;
	Eigen::Transpositions<Eigen::Dynamic> transpositions(p_eig);

	// result = P b
	result_eig = transpositions * b_eig;

	// result = L^-1 (P b)
	if (lower)
		Eigen::TriangularView<typename SGMatrix<T>::EigenMatrixXtMap,
		                      Eigen::Lower>(L_eig)
		    .solveInPlace(result_eig);
	else
		Eigen::TriangularView<typename SGMatrix<T>::EigenMatrixXtMap,
		                      Eigen::Upper>(L_eig)
		    .transpose()
		    .solveInPlace(result_eig);

	auto tolerance =
	    1.0 / Eigen::NumTraits<typename Eigen::NumTraits<T>::Real>::highest();

	// result = D^-1 L^-1 P b
	for (auto i : range(d.vlen))
	{
		if (std::abs(d[i]) > tolerance)
			result_eig.row(i) /= d[i];
		else
			result_eig.row(i).setZero();
	}

	// result = U^-1 (D^-1 L^-1 P b)
	if (lower)
		Eigen::TriangularView<typename SGMatrix<T>::EigenMatrixXtMap,
		                      Eigen::Lower>(L_eig)
		    .transpose()
		    .solveInPlace(result_eig);
	else
		Eigen::TriangularView<typename SGMatrix<T>::EigenMatrixXtMap,
		                      Eigen::Upper>(L_eig)
		    .solveInPlace(result_eig);

	// result = P^-1 (U^-1 D^-1 L^-1 P b) = A^-1 b
	result_eig = transpositions.transpose() * result_eig;

	return result;
}

template <typename T>
SGVector<T> LinalgBackendEigen::qr_solver_impl(
    const SGMatrix<T>& A, const SGVector<T>& b) const
{
	SGVector<T> result(b.vlen);
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGVector<T>::EigenVectorXtMap result_eig = result;

	result_eig = (A_eig.householderQr().solve(b_eig));

	return result;
}

template <typename T>
SGMatrix<T> LinalgBackendEigen::qr_solver_impl(
    const SGMatrix<T>& A, const SGMatrix<T> b) const
{
	SGMatrix<T> result(b.num_rows, b.num_cols);
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGMatrix<T>::EigenMatrixXtMap b_eig = b;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig = (A_eig.householderQr().solve(b_eig));

	return result;
}

template <typename T>
SGMatrix<T> LinalgBackendEigen::triangular_solver_impl(
    const SGMatrix<T>& L, const SGMatrix<T>& b, const bool lower) const
{
	SGMatrix<T> x(b.num_rows, b.num_cols);
	typename SGMatrix<T>::EigenMatrixXtMap L_eig = L;
	typename SGMatrix<T>::EigenMatrixXtMap b_eig = b;
	typename SGMatrix<T>::EigenMatrixXtMap x_eig = x;

	if (lower == false)
	{
		Eigen::TriangularView<typename SGMatrix<T>::EigenMatrixXtMap,
		                      Eigen::Upper>
		    tlv(L_eig);
		x_eig = tlv.solve(b_eig);
	}
	else
	{
		Eigen::TriangularView<typename SGMatrix<T>::EigenMatrixXtMap,
		                      Eigen::Lower>
		    tlv(L_eig);
		x_eig = tlv.solve(b_eig);
	}

	return x;
}

template <typename T>
SGVector<T> LinalgBackendEigen::triangular_solver_impl(
    const SGMatrix<T>& L, const SGVector<T>& b, const bool lower) const
{
	SGVector<T> x(b.size());
	typename SGMatrix<T>::EigenMatrixXtMap L_eig = L;
	typename SGVector<T>::EigenVectorXtMap b_eig = b;
	typename SGVector<T>::EigenVectorXtMap x_eig = x;

	if (lower == false)
	{
		Eigen::TriangularView<typename SGMatrix<T>::EigenMatrixXtMap,
		                      Eigen::Upper>
		    tlv(L_eig);
		x_eig = tlv.solve(b_eig);
	}
	else
	{
		Eigen::TriangularView<typename SGMatrix<T>::EigenMatrixXtMap,
		                      Eigen::Lower>
		    tlv(L_eig);
		x_eig = tlv.solve(b_eig);
	}

	return x;
}
