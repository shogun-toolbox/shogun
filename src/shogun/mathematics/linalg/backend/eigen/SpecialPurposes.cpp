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

#define BACKEND_GENERIC_CROSS_ENTROPY(Type, Container)                         \
	Type LinalgBackendEigen::cross_entropy(                                    \
	    const Container<Type>& P, const Container<Type>& Q) const              \
	{                                                                          \
		return cross_entropy_impl(P, Q);                                       \
	}
DEFINE_FOR_NON_INTEGER_REAL_PTYPE(BACKEND_GENERIC_CROSS_ENTROPY, SGMatrix)
#undef BACKEND_GENERIC_CROSS_ENTROPY

#define BACKEND_GENERIC_LOGISTIC(Type, Container)                              \
	void LinalgBackendEigen::logistic(                                         \
	    const Container<Type>& a, Container<Type>& result) const               \
	{                                                                          \
		logistic_impl(a, result);                                              \
	}
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_LOGISTIC, SGMatrix)
#undef BACKEND_GENERIC_LOGISTIC

#define BACKEND_GENERIC_MULTIPLY_BY_LOGISTIC_DERIV(Type, Container)            \
	void LinalgBackendEigen::multiply_by_logistic_derivative(                  \
	    const Container<Type>& a, Container<Type>& result) const               \
	{                                                                          \
		multiply_by_logistic_derivative_impl(a, result);                       \
	}
DEFINE_FOR_NUMERIC_PTYPE(BACKEND_GENERIC_MULTIPLY_BY_LOGISTIC_DERIV, SGMatrix)
#undef BACKEND_GENERIC_MULTIPLY_BY_LOGISTIC_DERIV

#define BACKEND_GENERIC_MULTIPLY_BY_RECTIFIED_LINEAR_DERIV(Type, Container)    \
	void LinalgBackendEigen::multiply_by_rectified_linear_derivative(          \
	    const Container<Type>& a, Container<Type>& result) const               \
	{                                                                          \
		multiply_by_rectified_linear_derivative_impl(a, result);               \
	}
DEFINE_FOR_NON_INTEGER_REAL_PTYPE(
    BACKEND_GENERIC_MULTIPLY_BY_RECTIFIED_LINEAR_DERIV, SGMatrix)
#undef BACKEND_GENERIC_MULTIPLY_BY_RECTIFIED_LINEAR_DERIV

#define BACKEND_GENERIC_RECTIFIED_LINEAR(Type, Container)                      \
	void LinalgBackendEigen::rectified_linear(                                 \
	    const Container<Type>& a, Container<Type>& result) const               \
	{                                                                          \
		rectified_linear_impl(a, result);                                      \
	}
DEFINE_FOR_REAL_PTYPE(BACKEND_GENERIC_RECTIFIED_LINEAR, SGMatrix)
#undef BACKEND_GENERIC_RECTIFIED_LINEAR

#define BACKEND_GENERIC_SOFTMAX(Type, Container)                               \
	void LinalgBackendEigen::softmax(Container<Type>& a) const                 \
	{                                                                          \
		softmax_impl(a);                                                       \
	}
DEFINE_FOR_NON_INTEGER_REAL_PTYPE(BACKEND_GENERIC_SOFTMAX, SGMatrix)
#undef BACKEND_GENERIC_SOFTMAX

#define BACKEND_GENERIC_SQUARED_ERROR(Type, Container)                         \
	Type LinalgBackendEigen::squared_error(                                    \
	    const Container<Type>& P, const Container<Type>& Q) const              \
	{                                                                          \
		return squared_error_impl(P, Q);                                       \
	}
DEFINE_FOR_NON_INTEGER_REAL_PTYPE(BACKEND_GENERIC_SQUARED_ERROR, SGMatrix)
#undef BACKEND_GENERIC_SQUARED_ERROR

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_REAL_PTYPE
#undef DEFINE_FOR_NON_INTEGER_PTYPE
#undef DEFINE_FOR_NUMERIC_PTYPE

template <typename T>
T LinalgBackendEigen::cross_entropy_impl(
    const SGMatrix<T>& p, const SGMatrix<T>& q) const
{
	typename SGMatrix<T>::EigenMatrixXtMap p_eig = p;
	typename SGMatrix<T>::EigenMatrixXtMap q_eig = q;

	return -1 * (p_eig.array() * (q_eig.array() + 1e-30).log()).sum();
}

template <typename T>
void LinalgBackendEigen::logistic_impl(
    const SGMatrix<T>& a, SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig = (T)1 / (1 + ((-1 * a_eig).array()).exp());
}

template <typename T>
void LinalgBackendEigen::multiply_by_logistic_derivative_impl(
    const SGMatrix<T>& a, SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	result_eig = result_eig.array() * a_eig.array() * ((T)1 - a_eig.array());
}

template <typename T>
void LinalgBackendEigen::multiply_by_rectified_linear_derivative_impl(
    const SGMatrix<T>& a, SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	for (index_t i = 0; i < a_eig.rows() * a_eig.cols(); ++i)
		if (a_eig(i) == 0)
			result_eig(i) = 0;
}

template <typename T>
void LinalgBackendEigen::rectified_linear_impl(
    const SGMatrix<T>& a, SGMatrix<T>& result) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	typename SGMatrix<T>::EigenMatrixXtMap result_eig = result;

	for (index_t i = 0; i < a_eig.rows() * a_eig.cols(); ++i)
		result_eig(i) = CMath::max((T)0, a_eig(i));
}

/** Eigen3 softmax method */
template <typename T, template <typename> class Container>
void LinalgBackendEigen::softmax_impl(Container<T>& a) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;

	auto max = a_eig.maxCoeff();
	for (index_t j = 0; j < a.num_cols; ++j)
	{
		auto sum = (a_eig.col(j).array() - max).exp().sum();
		T normalizer = (T)std::log(sum); // Has to use T instead of float
		a_eig.col(j) = (a_eig.col(j).array() - normalizer - max).exp();
	}
}

template <typename T>
T LinalgBackendEigen::squared_error_impl(
    const SGMatrix<T>& p, const SGMatrix<T>& q) const
{
	typename SGMatrix<T>::EigenMatrixXtMap p_eig = p;
	typename SGMatrix<T>::EigenMatrixXtMap q_eig = q;

	return 0.5 * (p_eig - q_eig).array().square().sum();
}
