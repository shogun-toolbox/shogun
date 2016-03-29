/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Christopher Goldsworthy, modeled after code written by Pen Deng,
 * who wrote MeanEign3.h
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

#ifndef COVARIANCE_EIGEN_H
#define COVARIANCE_EIGEN_H

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>


namespace shogun{

namespace linalg{

namespace implementation{

/**
 * @brief Generic class int2float which converts different types of integer
 * into float64 type.
 */
template <typename inputType>
struct int2float
{
	using floatType = inputType;
};

/**
 * @brief Specialization of generic class int2float which converts int32 into float64.
 */
template <>
struct int2float<int32_t>
{
	using floatType = float64_t;
};

/**
 * @brief Specialization of generic class int2float which converts int64 into float64.
 */
template <>
struct int2float<int64_t>
{
	using floatType = float64_t;
};

/**
 * @brief A generic implementation of the covariance function that be
 * used for various matrix and vector types.
 */
template <enum Backend, class Matrix>
struct covariance
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** int2float type */
	typedef typename int2float<T>::floatType ReturnType;

	/**
	 * This computes the covariance of a given vector or matrix
	 *
	 * @param a the vector or matrix whose covariance we want to compute
	 * @return the covariance of the matrix or vector \f$\covariance_i a_i\f$
	 */
	static ReturnType compute(Matrix a, bool no_diag=false);
};

/**
 * @brief A specific implementation of covariance that implements Eigen3
 * for it's backend linear algebra library.  It offers a static compute
 * method that computes the covariance for a given matrix or vector
 */
template <class Matrix>
struct covariance<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** int2float type */
	typedef typename int2float<T>::floatType ReturnType;

	/**
	 * This computes the covariance of a given vector or matrix
	 *
	 * @param a the vector or matrix whose covariance we want to compute
	 * @return the covariance of the matrix or vector \f$\covariance_i a_i\f$
	 */
	static ReturnType compute(Matrix a, bool no_diag=false){

		ReturnType covariance;



		return covariance;
	}
};

}

}

}

#endif
