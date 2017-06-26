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

#define BACKEND_GENERIC_IDENTITY(Type, Container)                              \
	void LinalgBackendEigen::identity(Container<Type>& I) const                \
	{                                                                          \
		identity_impl(I);                                                      \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_IDENTITY, SGMatrix)
#undef BACKEND_GENERIC_IDENTITY

#define BACKEND_GENERIC_SET_CONST(Type, Container)                             \
	void LinalgBackendEigen::set_const(Container<Type>& a, const Type value)   \
	    const                                                                  \
	{                                                                          \
		set_const_impl(a, value);                                              \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SET_CONST, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_SET_CONST, SGMatrix)
#undef BACKEND_GENERIC_SET_CONST

#define BACKEND_GENERIC_RANGE_FILL(Type, Container)                            \
	void LinalgBackendEigen::range_fill(Container<Type>& a, const Type start)  \
	    const                                                                  \
	{                                                                          \
		range_fill_impl(a, start);                                             \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_RANGE_FILL, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_RANGE_FILL, SGMatrix)
#undef BACKEND_GENERIC_RANGE_FILL

#define BACKEND_GENERIC_ZERO(Type, Container)                                  \
	void LinalgBackendEigen::zero(Container<Type>& a) const                    \
	{                                                                          \
		zero_impl(a);                                                          \
	}
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ZERO, SGVector)
DEFINE_FOR_ALL_PTYPE(BACKEND_GENERIC_ZERO, SGMatrix)
#undef BACKEND_GENERIC_ZERO

#undef DEFINE_FOR_ALL_PTYPE
#undef DEFINE_FOR_REAL_PTYPE
#undef DEFINE_FOR_NON_INTEGER_PTYPE
#undef DEFINE_FOR_NUMERIC_PTYPE

template <typename T>
void LinalgBackendEigen::identity_impl(SGMatrix<T>& I) const
{
	typename SGMatrix<T>::EigenMatrixXtMap I_eig = I;
	I_eig.setIdentity();
}

template <typename T, template <typename> class Container>
void LinalgBackendEigen::range_fill_impl(Container<T>& a, const T start) const
{
	for (decltype(a.size()) i = 0; i < a.size(); ++i)
		a[i] = start + T(i);
}

template <typename T, template <typename> class Container>
void LinalgBackendEigen::set_const_impl(Container<T>& a, T value) const
{
	for (decltype(a.size()) i = 0; i < a.size(); ++i)
		a[i] = value;
}

template <typename T>
void LinalgBackendEigen::zero_impl(SGVector<T>& a) const
{
	typename SGVector<T>::EigenVectorXtMap a_eig = a;
	a_eig.setZero();
}

template <typename T>
void LinalgBackendEigen::zero_impl(SGMatrix<T>& a) const
{
	typename SGMatrix<T>::EigenMatrixXtMap a_eig = a;
	a_eig.setZero();
}
