/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef REDUX_IMPL_H_
#define REDUX_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

namespace linalg
{

/**
 * All backend specific implementations are defined within this namespace
 */
namespace impl
{

/**
 * @brief Generic class dot which provides a static compute method. This class
 * is specialized for different types of vectors and backend, providing a mean
 * to deal with various vectors directly without having to convert
 */
template <class Info,template<class,Info...>class Vector,class T,enum Backend,Info... I>
struct dot
{
	typedef Vector<T,I...> vector_type;

	/**
	 * Method that computes the dot product
	 *
	 * @param \f$\mathbf{a}\f$ first vector
	 * @param \f$\mathbf{b}\f$ second vector
	 * @return the dot product of \f$\mathbf{a}\f$ and \$\mathbf{b}\f$, computed
	 * as \f$\sum_i a_i b_i\f$
	 */
	static T compute(vector_type a, vector_type b);
};

#ifdef HAVE_EIGEN3
/**
 * @brief Speciazation of generic dot which works with Eigen3 vectors using
 * Eigen3 backend for computing dot. Available whenever Eigen3 is found, no
 * matter what global backend was set
 */
template <> template <class T,int... Info>
struct dot<int,Eigen::Matrix,T,Backend::EIGEN3,Info...>
{
	typedef Eigen::Matrix<T,Info...> vector_type;

	/**
	 * Method that computes the dot product of Eigen3 vectors
	 *
	 * @param \f$\mathbf{a}\f$ first vector
	 * @param \f$\mathbf{b}\f$ second vector
	 * @return the dot product of \f$\mathbf{a}\f$ and \$\mathbf{b}\f$, computed
	 * as \f$\sum_i a_i b_i\f$
	 */
	static T compute(vector_type a, vector_type b)
	{
		return a.dot(b);
	}
};

/**
 * @brief Speciazation of generic dot which works with SGVectors and uses Eigen3
 * as backend for computing dot.
 */
template <> template <class T>
struct dot<int,shogun::SGVector,T,Backend::EIGEN3>
{
	typedef shogun::SGVector<T> vector_type;

	/**
	 * Method that computes the dot product of SGVectors using Eigen3
	 *
	 * @param \f$\mathbf{a}\f$ first vector
	 * @param \f$\mathbf{b}\f$ second vector
	 * @return the dot product of \f$\mathbf{a}\f$ and \$\mathbf{b}\f$, computed
	 * as \f$\sum_i a_i b_i\f$
	 */
	static T compute(vector_type a, vector_type b)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
		Eigen::Map<VectorXt> vec_a(a.vector, a.vlen);
		Eigen::Map<VectorXt> vec_b(b.vector, b.vlen);
		return vec_a.dot(vec_b);
	}
};
#endif // HAVE_EIGEN3

}

}
#endif // REDUX_IMPL_H_
