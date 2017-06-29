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

#define BACKEND_GENERIC_CHOLESKY_FACTOR(Type, Container)                       \
	Container<Type> LinalgBackendEigen::cholesky_factor(                       \
	    const Container<Type>& A, const bool lower) const                      \
	{                                                                          \
		return cholesky_factor_impl(A, lower);                                 \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CHOLESKY_FACTOR, SGMatrix)
#undef BACKEND_GENERIC_CHOLESKY_FACTOR

#define BACKEND_GENERIC_EIGEN_SOLVER(Type, Container)                          \
	void LinalgBackendEigen::eigen_solver(                                     \
	    const Container<Type>& A, SGVector<Type>& eigenvalues,                 \
	    SGMatrix<Type>& eigenvectors) const                                    \
	{                                                                          \
		eigen_solver_impl(A, eigenvalues, eigenvectors);                       \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_EIGEN_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_EIGEN_SOLVER

#define BACKEND_GENERIC_EIGEN_SOLVER_SYMMETRIC(Type, Container)                \
	void LinalgBackendEigen::eigen_solver_symmetric(                           \
	    const Container<Type>& A, SGVector<Type>& eigenvalues,                 \
	    SGMatrix<Type>& eigenvectors) const                                    \
	{                                                                          \
		eigen_solver_symmetric_impl(A, eigenvalues, eigenvectors);             \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_EIGEN_SOLVER_SYMMETRIC, SGMatrix)
#undef BACKEND_GENERIC_EIGEN_SOLVER_SYMMETRIC

#define BACKEND_GENERIC_SVD(Type, Container)                                   \
	void LinalgBackendEigen::svd(                                              \
	    const Container<Type>& A, SGVector<Type> s, Container<Type> U,         \
	    bool thin_U) const                                                     \
	{                                                                          \
		return svd_impl(A, s, U, thin_U);                                      \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_SVD, SGMatrix)
#undef BACKEND_GENERIC_SVD

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_REAL_PTYPE
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

template <typename T>
void LinalgBackendEigen::eigen_solver_impl(
    const SGMatrix<T>& A, SGVector<T>& eigenvalues,
    SGMatrix<T>& eigenvectors) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGMatrix<T>::EigenMatrixXtMap eigenvectors_eig = eigenvectors;
	typename SGVector<T>::EigenVectorXtMap eigenvalues_eig = eigenvalues;

	Eigen::EigenSolver<typename SGMatrix<T>::EigenMatrixXt> solver(A_eig);

	/*
	 * checking for success
	 *
	 * 0: Eigen::Success. Decomposition was successful
	 * 1: Eigen::NumericalIssue. The input contains INF or NaN values or
	 * overflow occured
	 */
	REQUIRE(
	    solver.info() != Eigen::NumericalIssue,
	    "The input contains INF or NaN values or overflow occured.\n");

	eigenvalues_eig = solver.eigenvalues().real();
	eigenvectors_eig = solver.eigenvectors().real();
}

void LinalgBackendEigen::eigen_solver_impl(
    const SGMatrix<complex128_t>& A, SGVector<complex128_t>& eigenvalues,
    SGMatrix<complex128_t>& eigenvectors) const
{
	typename SGMatrix<complex128_t>::EigenMatrixXtMap A_eig = A;
	typename SGMatrix<complex128_t>::EigenMatrixXtMap eigenvectors_eig =
	    eigenvectors;
	typename SGVector<complex128_t>::EigenVectorXtMap eigenvalues_eig =
	    eigenvalues;

	Eigen::ComplexEigenSolver<typename SGMatrix<complex128_t>::EigenMatrixXt>
	    solver(A_eig);

	REQUIRE(
	    solver.info() != Eigen::NumericalIssue,
	    "The input contains INF or NaN values or overflow occured.\n");

	eigenvalues_eig = solver.eigenvalues();
	eigenvectors_eig = solver.eigenvectors();
}

template <typename T>
void LinalgBackendEigen::eigen_solver_symmetric_impl(
    const SGMatrix<T>& A, SGVector<T>& eigenvalues,
    SGMatrix<T>& eigenvectors) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGMatrix<T>::EigenMatrixXtMap eigenvectors_eig = eigenvectors;
	typename SGVector<T>::EigenVectorXtMap eigenvalues_eig = eigenvalues;

	Eigen::SelfAdjointEigenSolver<typename SGMatrix<T>::EigenMatrixXt> solver(
	    A_eig);

	/*
	 * checking for success
	 *
	 * 0: Eigen::Success. Eigenvalues computation was successful
	 * 2: Eigen::NoConvergence. Iterative procedure did not converge.
	 */
	REQUIRE(
	    solver.info() != Eigen::NoConvergence,
	    "Iterative procedure did not converge!\n");

	eigenvalues_eig = solver.eigenvalues().template cast<T>();
	eigenvectors_eig = solver.eigenvectors().template cast<T>();
}

template <typename T>
void LinalgBackendEigen::svd_impl(
    const SGMatrix<T>& A, SGVector<T>& s, SGMatrix<T>& U, bool thin_U) const
{
	typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
	typename SGVector<T>::EigenVectorXtMap s_eig = s;
	typename SGMatrix<T>::EigenMatrixXtMap U_eig = U;

	auto svd_eig =
	    A_eig.jacobiSvd(thin_U ? Eigen::ComputeThinU : Eigen::ComputeFullU);
	s_eig = svd_eig.singularValues().template cast<T>();
	U_eig = svd_eig.matrixU().template cast<T>();
}