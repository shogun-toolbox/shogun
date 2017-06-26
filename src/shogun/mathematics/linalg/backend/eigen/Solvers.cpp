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

#define BACKEND_GENERIC_CHOLESKY_SOLVER(Type, Container)                       \
	SGVector<Type> LinalgBackendEigen::cholesky_solver(                        \
	    const Container<Type>& L, const SGVector<Type>& b, const bool lower)   \
	    const                                                                  \
	{                                                                          \
		return cholesky_solver_impl(L, b, lower);                              \
	}
DEFINE_FOR_NON_INTEGER_PTYPE(BACKEND_GENERIC_CHOLESKY_SOLVER, SGMatrix)
#undef BACKEND_GENERIC_CHOLESKY_SOLVER

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_REAL_PTYPE
#undef DEFINE_FOR_NON_INTEGER_PTYPE
#undef DEFINE_FOR_NUMERIC_PTYPE

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
		Eigen::TriangularView<Eigen::Map<typename SGMatrix<T>::EigenMatrixXt, 0,
		                                 Eigen::Stride<0, 0>>,
		                      Eigen::Upper>
		    tlv(L_eig);

		x_eig = (tlv.transpose()).solve(tlv.solve(b_eig));
	}
	else
	{
		Eigen::TriangularView<Eigen::Map<typename SGMatrix<T>::EigenMatrixXt, 0,
		                                 Eigen::Stride<0, 0>>,
		                      Eigen::Lower>
		    tlv(L_eig);
		x_eig = (tlv.transpose()).solve(tlv.solve(b_eig));
	}

	return x;
}
