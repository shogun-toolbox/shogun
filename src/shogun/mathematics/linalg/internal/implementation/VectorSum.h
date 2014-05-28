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

#ifndef VECTOR_SUM_IMPL_H_
#define VECTOR_SUM_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

namespace shogun
{

namespace linalg
{

/**
 * All backend specific implementations are defined within this namespace
 */
namespace implementation
{

/**
 * @brief Generic class vector_sum which provides a static compute method. This class
 * is specialized for different types of vectors and backend, providing a mean
 * to deal with various vectors directly without having to convert
 */
template <class Info,enum Backend,template<class,Info...>class Vector,class T,Info... I>
struct vector_sum
{
	typedef Vector<T,I...> vector_type;

	/**
	 * Method that computes the vector sum
	 *
	 * @param a vector whose sum has to be computed
	 * @return the vector sum \f$\sum_i a_i\f$
	 */
	static T compute(vector_type a);
};

#ifdef HAVE_EIGEN3
/**
 * @brief Specialization of generic vector_sum which works with SGVectors and uses Eigen3
 * as backend for computing vector_sum.
 */
template <> template <class T>
struct vector_sum<int,Backend::EIGEN3,shogun::SGVector,T>
{
	typedef shogun::SGVector<T> vector_type;

	/**
	 * Method that computes the sum of SGVectors using Eigen3
	 *
	 * @param a vector whose sum has to be computed
	 * @return the vector sum \f$\sum_i a_i\f$
	 */
	static T compute(vector_type a)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
		Eigen::Map<VectorXt> vec_a(a.vector, a.vlen);
		return vec_a.sum();
	}
};

/**
 * @brief Specialization of generic vector_sum which works with Eigen3 vectors using
 * Eigen3 backend for computing vector_sum.
 */
template <> template <class T,int... Info>
struct vector_sum<int,Backend::EIGEN3,Eigen::Matrix,T,Info...>
{
	typedef Eigen::Matrix<T,Info...> vector_type;

	/**
	 * Method that computes the sum of Eigen3 vectors using Eigen3
	 *
	 * @param a vector whose sum has to be computed
	 * @return the vector sum \f$\sum_i a_i\f$
	 */
	static T compute(vector_type a)
	{
		return a.sum();
	}
};

#endif // HAVE_EIGEN3

}

}

}
#endif // VECTOR_SUM_IMPL_H_
