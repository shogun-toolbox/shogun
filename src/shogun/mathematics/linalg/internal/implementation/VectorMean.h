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

#ifndef VECTOR_MEAN_IMPL_H_
#define VECTOR_MEAN_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/internal/implementation/VectorSum.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUVector.h>
#include <shogun/lib/GPUMatrix.h>
#endif

#include <shogun/mathematics/eigen3.h>

#include <numeric>

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
 * @brief Generic class vector_mean which provides a static compute method. This class
 * is specialized for different types of vectors and backend, providing a mean
 * to deal with various vectors directly without having to convert
 */
template <enum Backend, class Vector>
struct vector_mean
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/**
	 * Method that computes the vector mean
	 *
	 * @param a vector whose mean has to be computed
	 * @return the vector mean \f$\mean_i a_i\f$
	 */
	static T compute(Vector a);
};

/**
 * @brief Specialization of generic vector_mean for the Native backend
 */
template <class Vector>
struct vector_mean<Backend::NATIVE, Vector>
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/**
	 * Method that computes the mean of SGVectors
	 *
	 * @param vec a vector whose mean has to be computed
	 * @return the vector mean \f$\mean_i a_i\f$
	 */
	static T compute(SGVector<T> vec)
	{
		REQUIRE(vec.vlen > 0, "Vector can not be empty!\n");
		return (vector_sum<Backend::NATIVE, SGVector<T> >::compute(vec)
                 / vec.vlen);
	}
};

/**
 * @brief Specialization of generic vector_mean for the Eigen3 backend
 */
template <class Vector>
struct vector_mean<Backend::EIGEN3, Vector>
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/**
	 * Method that computes the mean of SGVectors using Eigen3
	 *
	 * @param vec a vector whose mean has to be computed
	 * @return the vector mean \f$\mean_i a_i\f$
	 */
	static T compute(SGVector<T> vec)
	{
		REQUIRE(vec.vlen > 0, "Vector can not be empty!\n");
                return (vector_sum<Backend::EIGEN3, SGVector<T> >::compute(vec)
                 / vec.vlen);
	}
};


#ifdef HAVE_VIENNACL
/**
 * @brief Specialization of generic vector_mean for the ViennaCL backend
 */
template <class Vector>
struct vector_mean<Backend::VIENNACL, Vector>
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/**
	 * Method that computes the mean of SGVectors using Eigen3
	 *
	 * @param a vector whose mean has to be computed
	 * @return the vector mean \f$\mean_i a_i\f$
	 */
	static T compute(CGPUVector<T> vec)
	{
		REQUIRE(vec.vlen > 0, "Vector can not be empty!\n");
                return (vector_sum<Backend::VIENNACL, CGPUVector<T> >::compute(vec)
                 / vec.vlen);
	}
};

#endif // HAVE_VIENNACL

}

}

}
#endif // VECTOR_MEAN_IMPL_H_
