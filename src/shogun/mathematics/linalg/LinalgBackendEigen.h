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

#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/LinalgBackendBase.h>

#ifndef LINALG_BACKEND_EIGEN_H__
#define LINALG_BACKEND_EIGEN_H__

namespace shogun
{

/** @brief linalg methods with Eigen3 backend */
class LinalgBackendEigen : public LinalgBackendBase
{
public:
	/** Implementation of @see LinalgBackendBase::add */
	#define BACKEND_GENERIC_ADD(Type) \
	virtual SGVector<Type> add(const SGVector<Type>& a, const SGVector<Type>& b, Type alpha, Type beta) const \
	{  \
		return add_impl(a, b, alpha, beta); \
	}

	BACKEND_GENERIC_ADD(bool);
	BACKEND_GENERIC_ADD(char);
	BACKEND_GENERIC_ADD(int8_t);
	BACKEND_GENERIC_ADD(uint8_t);
	BACKEND_GENERIC_ADD(int16_t);
	BACKEND_GENERIC_ADD(uint16_t);
	BACKEND_GENERIC_ADD(int32_t);
	BACKEND_GENERIC_ADD(uint32_t);
	BACKEND_GENERIC_ADD(int64_t);
	BACKEND_GENERIC_ADD(uint64_t);
	BACKEND_GENERIC_ADD(float32_t);
	BACKEND_GENERIC_ADD(float64_t);
	BACKEND_GENERIC_ADD(floatmax_t);
	BACKEND_GENERIC_ADD(complex128_t);
	#undef BACKEND_GENERIC_ADD

	/** Implementation of @see LinalgBackendBase::dot */
	#define BACKEND_GENERIC_DOT(Type) \
	virtual Type dot(const SGVector<Type>& a, const SGVector<Type>& b) const \
	{  \
		return dot_impl(a, b);  \
	}

	BACKEND_GENERIC_DOT(bool);
	BACKEND_GENERIC_DOT(char);
	BACKEND_GENERIC_DOT(int8_t);
	BACKEND_GENERIC_DOT(uint8_t);
	BACKEND_GENERIC_DOT(int16_t);
	BACKEND_GENERIC_DOT(uint16_t);
	BACKEND_GENERIC_DOT(int32_t);
	BACKEND_GENERIC_DOT(uint32_t);
	BACKEND_GENERIC_DOT(int64_t);
	BACKEND_GENERIC_DOT(uint64_t);
	BACKEND_GENERIC_DOT(float32_t);
	BACKEND_GENERIC_DOT(float64_t);
	BACKEND_GENERIC_DOT(floatmax_t);
	BACKEND_GENERIC_DOT(complex128_t);
	#undef BACKEND_GENERIC_DOT

	/** Implementation of @see LinalgBackendBase::sum */
	#define BACKEND_GENERIC_SUM(Type) \
	virtual Type sum(const SGVector<Type>& vec) const \
	{  \
		return sum_impl(vec);  \
	}
	BACKEND_GENERIC_SUM(bool);
	BACKEND_GENERIC_SUM(char);
	BACKEND_GENERIC_SUM(int8_t);
	BACKEND_GENERIC_SUM(uint8_t);
	BACKEND_GENERIC_SUM(int16_t);
	BACKEND_GENERIC_SUM(uint16_t);
	BACKEND_GENERIC_SUM(int32_t);
	BACKEND_GENERIC_SUM(uint32_t);
	BACKEND_GENERIC_SUM(int64_t);
	BACKEND_GENERIC_SUM(uint64_t);
	BACKEND_GENERIC_SUM(float32_t);
	BACKEND_GENERIC_SUM(float64_t);
	BACKEND_GENERIC_SUM(floatmax_t);
	BACKEND_GENERIC_SUM(complex128_t);
	#undef BACKEND_GENERIC_SUM

private:
	/** Eigen3 vector C = alpha*A + beta*B method */
	template <typename T>
	SGVector<T> add_impl(const SGVector<T>& a, const SGVector<T>& b, T alpha, T beta) const
	{
		SGVector<T> c(a.vlen);
		typename SGVector<T>::EigenVectorXtMap a_eig = a;
		typename SGVector<T>::EigenVectorXtMap b_eig = b;
		typename SGVector<T>::EigenVectorXtMap c_eig = c;

		c_eig = alpha * a_eig + beta * b_eig;
		return c;
	}

	/** Eigen3 vector dot-product method */
	template <typename T>
	T dot_impl(const SGVector<T>& a, const SGVector<T>& b) const
	{
		return (typename SGVector<T>::EigenVectorXtMap(a)).dot(typename SGVector<T>::EigenVectorXtMap(b));
	}

	/** Eigen3 vector sum method */
	template <typename T>
	T sum_impl(const SGVector<T>& vec) const
	{
		return (typename SGVector<T>::EigenVectorXtMap(vec)).sum();
	}

};

}

#endif //LINALG_BACKEND_EIGEN_H__
