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

#ifndef EIGEN_EIGENSOLVERS_H
#define EIGEN_EIGENSOLVERS_H

template <typename T>
void LinalgBackendEigen::eigen_solver(
    const SGMatrix<T>& A, SGVector<T>& eigenvalues, SGMatrix<T>& eigenvectors,
    derived_tag) const
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

#endif