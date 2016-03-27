/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Chris Goldsworthy
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

#ifndef SRC_SHOGUN_MATHEMATICS_LINALG_INTERNAL_IMPLEMENTATION_VARIANCE_H_
#define SRC_SHOGUN_MATHEMATICS_LINALG_INTERNAL_IMPLEMENTATION_VARIANCE_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/linalg/internal/implementation/MeanEigen3.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun::linalg;

namespace shogun
{

namespace linalg
{

namespace implementation
{


/**
 * @brief Generic class variance which provides a static compute method. This class
 * is specialized for different types of vectors.
 */
template <enum Backend, class Vector>
struct variance
{
	/** Scalar type */
	typedef typename Vector::Scalar T;


	/**
	 * Method that computes the variance of the elements of a vector
	 *
	 * @param a vector whose variance is to be computed
	 * @return the vector's variance
	 */
	static T compute(Vector a);
};

/**
 * @brief Specialization of generic variance which works with SGVector
 * and uses Eigen3 as backend for computing variance.
 */
template <class Vector>
struct variance<Backend::EIGEN3, Vector>
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/**
	 * Method that computes the variance of SGVectors using Eigen3
	 *
	 * @param a vector whose variance has to be computed
	 * @return the vector's variance
	 */
	static T compute(SGVector<T> vec)
	{
		ASSERT(vec.vlen>1)
		ASSERT(vec.vector)

		float64_t mean = implementation::mean<Backend::EIGEN3, Vector>::compute(vec);

		float64_t sum_squared_diff=0;
		for (index_t i=0; i<vec.vlen; ++i)
			sum_squared_diff+=CMath::pow(vec.vector[i]-mean, 2);

		return sum_squared_diff / vec.vlen;
	}
};

/**
 * @Brief A generic class that contains two methods for computing the variance of
 * a matrix.  There's one method for computing column-wise and row-wise variance and one
 * method for computing element-wise variance
 */
template<enum Backend, typename Matrix>
struct matrix_variance{

	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Vector return type */
	typedef SGVector<T> ReturnTypeVec;

	/**
	 * Method that can compute an column-wise or row-wise variance
	 * for a matrix
	 *
	 * @param m the matrix whose variance we want to compute
	 * @param col_wise if true, we compute the column wise variance -
	 * otherwise, we compute the row-wise variance
	 * @return a vector containing the row-wise / col-wise variance
	 */
	static ReturnTypeVec compute(Matrix m, bool col_wise);

	/**
	 * Method that computes the element-wise variance of a matrix
	 *
	 * @param m the matrix whose element-wise variance we want to
	 * compute
	 * @return the element-wise variance of m
	 */
	static T compute(Matrix m);
};

/**
 * @Brief A specialization of matrix_variance that uses SGMatrix and SGVector as it's types
 * and uses Eigen3 as it's backend component
 */
template<typename Matrix>
struct matrix_variance<Backend::EIGEN3, Matrix>{

	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Vector return type */
	typedef SGVector<T> ReturnTypeVec;

	/**
	 * Method that can compute an column-wise or row-wise variance
	 * for a matrix
	 *
	 * @param m the matrix whose variance we want to compute
	 * @param col_wise if true, we compute the column wise variance -
	 * otherwise, we compute the row-wise variance
	 * @return a vector containing the row-wise / col-wise variance
	 */
	static ReturnTypeVec compute(SGMatrix<T> m, bool col_wise){

		ASSERT(m.num_rows>0)
		ASSERT(m.num_cols>0)
		ASSERT(m.matrix)

		ReturnTypeVec variance;
		ReturnTypeVec mean = matrix_mean<Backend::EIGEN3, Matrix>::compute(m, col_wise);;

		if(col_wise){

			variance = ReturnTypeVec(m.num_cols);

			for (index_t i=0; i< m.num_cols; ++i)
				variance[i] = implementation::variance<Backend::EIGEN3, Matrix>::compute(*m.get_column_vector(i))
				/mean[i];
		}
		else{

			variance = ReturnTypeVec(m.num_rows);

			for (index_t i=0; i< m.num_rows; ++i)
				variance[i] = implementation::variance<Backend::EIGEN3, Matrix>::compute(m.get_row_vector(i))
				/mean[i];
		}

		return variance;
	}

	/**
	 * Method that computes the element-wise variance of a matrix
	 *
	 * @param m the matrix whose element-wise variance we want to
	 * compute
	 * @return the element-wise variance of m
	 */
	static T compute(SGMatrix<T> m){

		ASSERT(m.num_rows>0)
		ASSERT(m.num_cols>0)
		ASSERT(m.matrix)

		T mean = matrix_mean<Backend::EIGEN3, Matrix>::compute(m);

		float64_t sum_squared_diff=0;
		for (index_t i=0; i< m.num_rows * m.num_cols; ++i)
			sum_squared_diff+=CMath::pow(m[i]- mean, 2);

		return sum_squared_diff / (m.num_cols * m.num_rows);
	}
};

}

}

}


#endif /* SRC_SHOGUN_MATHEMATICS_LINALG_INTERNAL_IMPLEMENTATION_VARIANCE_H_ */
