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

#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/linalg/LinalgBackendEigen.h>
#include <shogun/mathematics/linalg/LinalgMacros.h>

using namespace shogun;

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
	    SGMatrix<Type>& eigenvectors, index_t k) const                         \
	{                                                                          \
		eigen_solver_symmetric_impl(A, eigenvalues, eigenvectors, k);          \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_EIGEN_SOLVER_SYMMETRIC, SGMatrix)
#undef BACKEND_GENERIC_EIGEN_SOLVER_SYMMETRIC

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_NON_COMPLEX_PTYPE
#undef DEFINE_FOR_NON_INTEGER_PTYPE
#undef DEFINE_FOR_NUMERIC_PTYPE

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
