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

#ifndef SRC_SHOGUN_MATHEMATICS_LINALG_INTERNAL_IMPLEMENTATION_STDDEVIATION_H_
#define SRC_SHOGUN_MATHEMATICS_LINALG_INTERNAL_IMPLEMENTATION_STDDEVIATION_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/linalg/internal/implementation/Variance.h>
#include <shogun/mathematics/eigen3.h>

namespace shogun
{

namespace linalg
{

namespace implementation
{

/**
 * @brief Generic class std_deviation which provides a static compute method. This class
 * is specialized for different types of vectors.
 */
template <enum Backend, class Vector>
struct std_deviation
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/** Calculates unbiased empirical standard deviation estimator of given
	 * values. Given \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\sqrt{\frac{1}{m-1}\sum_{i=1}^m (x-\bar{x})^2}\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * @param a vector of values
	 * @return standard deviation of given values
	 */
	static T compute(Vector a);
};

/**
 * @brief Specialization of generic std_deviation which works with SGVector
 * and uses Eigen3 as backend for computing std_deviation.
 */
template <class Vector>
struct std_deviation<Backend::EIGEN3, Vector>
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/** Calculates unbiased empirical standard deviation estimator of given
	 * values. Given \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\sqrt{\frac{1}{m-1}\sum_{i=1}^m (x-\bar{x})^2}\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * @param vec vector of values
	 * @return standard deviation of given values
	 */
	static T compute(SGVector<T> vec)
	{
		ASSERT(vec.vlen>1)
		ASSERT(vec.vector)

		return CMath::sqrt(variance<Backend::EIGEN3, Vector>::compute(vec));
	}
};

/**
 * @Brief A generic class that contains two methods for computing the std_deviation of
 * a matrix.  There's one method for computing column-wise and row-wise std_deviation and one
 * method for computing element-wise std_deviation
 */
template<enum Backend, typename Matrix>
struct matrix_std_deviation{

	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Vector return type */
	typedef SGVector<T> ReturnTypeVec;


	/** Calculates unbiased empirical standard deviation estimator of given
	 * values. Given \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\sqrt{\frac{1}{m-1}\sum_{i=1}^m (x-\bar{x})^2}\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * Computes the standard deviation for each row/col of matrix
	 *
	 * @param m matrix of values
	 * @param col_wise if true, every column vector will be used, row vectors
	 * otherwise
	 * @return standard deviation of given values
	 */
	static ReturnTypeVec compute(Matrix m, bool col_wise);

	/** Calculates unbiased empirical standard deviation estimator of given
	 * values. Given \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\sqrt{\frac{1}{m-1}\sum_{i=1}^m (x-\bar{x})^2}\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * Computes the standard deviation for each row/col of matrix
	 *
	 * @param m matrix of values
	 * @return standard deviation of given values
	 */
	static T compute(Matrix m);
};

/**
 * @Brief A specialization of matrix_std_deviation that uses SGMatrix and SGVector as it's types
 * and uses Eigen3 as it's backend component
 */
template<typename Matrix>
struct matrix_std_deviation<Backend::EIGEN3, Matrix>{

	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Vector return type */
	typedef SGVector<T> ReturnTypeVec;

	/** Calculates unbiased empirical standard deviation estimator of given
	 * values. Given \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\sqrt{\frac{1}{m-1}\sum_{i=1}^m (x-\bar{x})^2}\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * Computes the standard deviation for each row/col of matrix
	 *
	 * @param m matrix of values
	 * @param col_wise if true, every column vector will be used, row vectors
	 * otherwise
	 * @return standard deviation of given values
	 */
	static ReturnTypeVec compute(SGMatrix<T> m, bool col_wise){

		ASSERT(m.num_rows>0)
		ASSERT(m.num_cols>0)
		ASSERT(m.matrix)

		ReturnTypeVec std_deviation;
		ReturnTypeVec variance = matrix_variance<Backend::EIGEN3, Matrix>::compute(m, col_wise);;

		if(col_wise){

			std_deviation = ReturnTypeVec(m.num_cols);

			for (index_t i=0; i< m.num_cols; ++i)
				std_deviation[i] = CMath::sqrt(variance[i]);
		}
		else{

			std_deviation = ReturnTypeVec(m.num_rows);

			for (index_t i=0; i< m.num_rows; ++i)
				std_deviation[i] = CMath::sqrt(variance[i]);
		}

		return std_deviation;
	}

	/** Calculates unbiased empirical standard deviation estimator of given
	 * values. Given \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\sqrt{\frac{1}{m-1}\sum_{i=1}^m (x-\bar{x})^2}\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * Computes the standard deviation for each row/col of matrix
	 *
	 * @param m matrix of values
	 * @return standard deviation of given values
	 */
	static T compute(SGMatrix<T> m){

		ASSERT(m.num_rows>0)
		ASSERT(m.num_cols>0)
		ASSERT(m.matrix)

		return CMath::sqrt(matrix_variance<Backend::EIGEN3, Matrix>::compute(m));
	}
};

}

}

}


#endif /* SRC_SHOGUN_MATHEMATICS_LINALG_INTERNAL_IMPLEMENTATION_STDDEVIATION_H_ */
