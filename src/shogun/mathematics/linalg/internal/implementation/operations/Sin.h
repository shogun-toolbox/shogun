/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Soumyajit De
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

#ifndef SIN_IMPL_H_
#define SIN_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <string>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

#include <shogun/mathematics/linalg/internal/implementation/operations/opencl_operation.h>

namespace shogun
{

namespace linalg
{

namespace operations
{

/**
 * Template struct sin for computing element-wise sin for matrices and vectors.
 * The operator() is for NATIVE backend implementation. Methods compute_using_eigen3
 * are for computing element-wise sin using EIGEN3 backend.
 */
template <typename T>
struct sin : public ocl_operation
{
	/** The return type */
	using return_type = float64_t;

	/*
	 * Default constructor. Initializes the OpenCL operation
	 */
	sin() : ocl_operation("return sin(element);")
	{
	}

	/**
	 * @param val The scalar value
	 * @return sin(val)
	 */
	return_type operator () (T& val) const
	{
		return CMath::sin(val);
	}

#ifdef HAVE_EIGEN3
	/** Eigen3 matrix type */
	using MatrixXt = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

	/** Eigen3 vector type */
	using VectorXt = Eigen::Matrix<T,Eigen::Dynamic,1>;

	/** Eigen3 matrix map type */
	using MapMatrixXt = Eigen::Map<MatrixXt>;

	/** Eigen3 vector map type */
	using MapVectorXt = Eigen::Map<VectorXt>;

	/**
	 * @param m The Eigen3 matrix map of the operand
	 * @return Element-wise sin in a newly allocated matrix
	 */
	Eigen::MatrixXd compute_using_eigen3(MapMatrixXt m) const
	{
		return m.array().template cast<double>().sin();
	}

	/**
	 * @param v The Eigen3 vector map of the operand
	 * @return Element-wise sin in a newly allocated vector
	 */
	Eigen::VectorXd compute_using_eigen3(MapVectorXt v) const
	{
		return v.array().template cast<double>().sin();
	}
#endif // HAVE_EIGEN3
};

/**
 * Specialization of template struct sin when scalar type is complex<double>.
 * The operator() is for NATIVE backend implementation. Not available for
 * ViennaCL or Eigen3 backend.
 */
template <>
struct sin<complex128_t>
{
	/** The return type */
	using return_type = complex128_t;

	/**
	 * @param val The scalar value
	 * @return sin(val)
	 */
	return_type operator () (complex128_t& val) const
	{
		return CMath::sin(val);
	}
};

}

}

}
#endif // SIN_IMPL_H_
