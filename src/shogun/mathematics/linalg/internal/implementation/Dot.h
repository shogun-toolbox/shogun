/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * Written (w) 2014 Khaled Nasr
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

#ifndef DOT_IMPL_H_
#define DOT_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>

#include <shogun/mathematics/eigen3.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUVector.h>
#include <viennacl/linalg/inner_prod.hpp>
#endif // HAVE_VIENNACL

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
 * @brief Generic class dot which provides a static compute method. This class
 * is specialized for different types of vectors and backend, providing a mean
 * to deal with various vectors directly without having to convert
 */
template <enum Backend, class Vector>
struct dot
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/**
	 * Method that computes the dot product
	 *
	 * @param a first vector
	 * @param b second vector
	 * @return the dot product of \f$\mathbf{a}\f$ and \f$\mathbf{b}\f$, computed
	 * as \f$\sum_i a_i b_i\f$
	 */
	static T compute(Vector a, Vector b);
};

/**
 * @brief Specialization of generic dot for the Eigen3 backend
 */
template <class Vector>
struct dot<Backend::EIGEN3, Vector>
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/**
	 * Method that computes the dot product of SGVectors/GPUVectors using Eigen3
	 *
	 * @param a first vector
	 * @param b second vector
	 * @return the dot product of \f$\mathbf{a}\f$ and \f$\mathbf{b}\f$, computed
	 * as \f$\sum_i a_i b_i\f$
	 */
	static T compute(shogun::SGVector<T> a, shogun::SGVector<T> b)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
		Eigen::Map<VectorXt> vec_a = a;
		Eigen::Map<VectorXt> vec_b = b;
		return vec_a.dot(vec_b);
	}
};

#ifdef HAVE_VIENNACL
/**
 * @brief Specialization of generic dot for the ViennaCL backend
 */
template <class Vector>
struct dot<Backend::VIENNACL, Vector>
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/**
	 * Method that computes the dot product using ViennaCL
	 *
	 * @param a first vector
	 * @param b second vector
	 * @return the dot product of \f$\mathbf{a}\f$ and \f$\mathbf{b}\f$, computed
	 * as \f$\sum_i a_i b_i\f$
	 */
	static T compute(shogun::CGPUVector<T> a, shogun::CGPUVector<T> b)
	{
		return viennacl::linalg::inner_prod(a.vcl_vector(), b.vcl_vector());
	}
};
#endif // HAVE_VIENNACL

}

}

}
#endif // DOT_IMPL_H_
