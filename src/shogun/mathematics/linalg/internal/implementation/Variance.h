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

#include <iostream>
using namespace std;

namespace shogun
{

namespace linalg
{

namespace implementation
{

/**
 * @brief Generic class variance which provides a static compute method. This class
 * can work with generic matricies.
 */
template <enum Backend, class Matrix>
struct variance
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Calculates unbiased empirical variance estimate given the entries from a matrix. Given a matrix
	 * \f$x\f$ with entries \f$\{x_{11}, ..., x_{mn}\}\f$, this is
	 * \f$\frac{1}{m*n-1}\sum_{i=1}^m\sum_{j=1}^n (x_{ij}-\bar{x})^2\f$ where
	 * \f$\bar x=\frac{1}{mn}\sum_{i=1}^m\sum_{j=1}^n x_{ij}\f$
	 *
	 * @param x matrix of values
	 * @return variance of given values
	 */
	static T compute(Matrix x);
};

/**
 * @brief Specialization of element-wise variance which works with SGMatrix
 * and uses Eigen3 as backend for computing variance.
 */
template <class Matrix>
struct variance<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Eigen matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic> MatrixXt;

	/** Calculates unbiased empirical variance estimate given the entries from a matrix. Given a matrix
	 * \f$x\f$ with entries \f$\{x_{11}, ..., x_{mn}\}\f$, this is
	 * \f$\frac{1}{m*n-1}\sum_{i=1}^m\sum_{j=1}^n (x_{ij}-\bar{x})^2\f$ where
	 * \f$\bar x=\frac{1}{mn}\sum_{i=1}^m\sum_{j=1}^n x_{ij}\f$
	 *
	 * @param x matrix of values
	 * @return variance of given values
	 */
	static T compute(SGMatrix<T> x)
	{
		REQUIRE(x.num_rows > 0, "Please ensure that m has more than 0 rows.\n")
		REQUIRE(x.num_cols > 0, "Please ensure that m has more than %d columns.\n")

		Eigen::Map<MatrixXt> eigX = x;
		MatrixXt eigSquaredResult(x.num_rows, x.num_cols);

		T meanVal = mean<Backend::EIGEN3, SGMatrix<T>>::compute(x);		
		eigSquaredResult.fill(meanVal);
		eigSquaredResult = (eigX - eigSquaredResult).array().square();

		return ((T) 1 / (x.num_rows*x.num_cols - 1)) * eigSquaredResult.sum();
	}
};

/**
 * @brief Generic class variance which provides a static compute method. This class
 * can work with generic vectors.
 */
template <enum Backend, class Vector>
struct vector_variance
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/** Calculates unbiased empirical variance estimate of given values. Given
	 * \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\frac{1}{m-1}\sum_{i=1}^m (x_i-\bar{x})^2\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * @param x vector of values
	 * @return variance of given values
	 */
	static T compute(Vector x);
};

/**
 * @brief Specialization of generic variance which works with SGVector
 * and uses Eigen3 as backend for computing variance.
 */
template <class Vector>
struct vector_variance<Backend::EIGEN3, Vector>
{
	/** Scalar type */
	typedef typename Vector::Scalar T;

	/** Eigen vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/** Calculates unbiased empirical variance estimate of given values. Given
	 * \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\frac{1}{m-1}\sum_{i=1}^m (x_i-\bar{x})^2\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * @param x vector of values
	 * @return variance of given values
	 */
	static T compute(SGVector<T> x)
	{
		REQUIRE(x.vlen>1, "Please ensure that vector length is greater than 1.\n")

		Eigen::Map<VectorXt> eigX = x;
		VectorXt eigSquaredResult(x.vlen);

		T meanVal = mean<Backend::EIGEN3, SGVector<T>>::compute(x);
		eigSquaredResult.fill(meanVal);
		eigSquaredResult = (eigX - eigSquaredResult).array().square();
		return ((T) 1 / (x.vlen - 1)) * eigSquaredResult.sum();
	}
};

/**
 * @Brief A generic class that computes the variance of a matrix column-wise.
 */
template<enum Backend, typename Matrix>
struct colwise_variance
{
	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Vector return type */
	typedef SGVector<T> ReturnType;

	/** Calculates unbiased empirical variance estimate of given values for each column of
	 * an input matrix. Given a single column \f$k\f$ with entries \f$\{x_{1k}, ..., x_{mk}\}\f$ 
	 * from matrix \f$x\f$, this is \f$\frac{1}{m-1}\sum_{i=1}^m (k_i-\bar{k})^2\f$ where
	 * \f$\bar k=\frac{1}{m}\sum_{i=1}^m k_i\f$.  This is computed for each column \f$k\f$.
	 *
	 * Computes the variance for each column of a matrix
	 *
	 * @param x matrix of values
	 * @return the column-wise variance of a matrix
	 */
	static ReturnType compute(Matrix x);
};

/**
 * @Brief A specialization of colwise_variance that uses SGMatrix and SGVector as its types
 * and uses Eigen3 as its backend component
 */
template<typename Matrix>
struct colwise_variance<Backend::EIGEN3, Matrix>
{

	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Vector return type */
	typedef SGVector<T> ReturnType;

	/** Eigen vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/** Calculates unbiased empirical variance estimate of given values for each column of
	 * an input matrix. Given a single column \f$k\f$ with entries \f$\{x_{1k}, ..., x_{mk}\}\f$ 
	 * from matrix \f$x\f$, this is \f$\frac{1}{m-1}\sum_{i=1}^m (k_i-\bar{k})^2\f$ where
	 * \f$\bar k=\frac{1}{m}\sum_{i=1}^m k_i\f$.  This is computed for each column \f$k\f$.
	 *
	 * Computes the variance for each column of a matrix
	 *
	 * @param m matrix of values
	 * @return the column-wise variance of a matrix
	 */
	static ReturnType compute(SGMatrix<T> m)
	{
		REQUIRE(m.num_rows > 0, "Please ensure that m has more than 0 rows.\n")
		REQUIRE(m.num_cols > 0, "Please ensure that m has more than %d columns.\n")

		
	}
};

/**
 * @Brief A generic class that computes the variance of a matrix row-wise.
 */
template<enum Backend, typename Matrix>
struct rowwise_variance
{
	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Vector return type */
	typedef SGVector<T> ReturnType;

	/** Calculates unbiased empirical variance estimate of the given values from each row of
	 * an input matrix. Given a single row \f$k\f$ with entries \f$\{x_{k1}, ..., x_{kn}\}\f$ 
	 * from matrix \f$x\f$, this is \f$\frac{1}{n-1}\sum_{i=1}^n (k_i-\bar{k})^2\f$ where
	 * \f$\bar k=\frac{1}{n}\sum_{i=1}^n k_i\f$.  This is computed for each row \f$n\f$.
	 *
	 * Computes the variance for each row of a matrix
	 *
	 * @param x matrix of values
	 * @return the row-wise variance of a matrix
	 */
	static ReturnType compute(Matrix x);
};

/**
 * @Brief A specialization of colwise_variance that uses SGMatrix and SGVector as its types
 * and uses Eigen3 as its backend component
 */
template<typename Matrix>
struct rowwise_variance<Backend::EIGEN3, Matrix>
{

	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Vector return type */
	typedef SGVector<T> ReturnType;

	/** Eigen vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/** Eigen matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic> MatrixXt;

	/** Calculates unbiased empirical variance estimate of the given values from each row of
	 * an input matrix. Given a single row \f$k\f$ with entries \f$\{x_{k1}, ..., x_{kn}\}\f$ 
	 * from matrix \f$x\f$, this is \f$\frac{1}{n-1}\sum_{i=1}^n (k_i-\bar{k})^2\f$ where
	 * \f$\bar k=\frac{1}{n}\sum_{i=1}^n k_i\f$.  This is computed for each row \f$n\f$.
	 *
	 * Computes the variance for each row of a matrix
	 *
	 * @param m matrix of values
	 * @return the column-wise variance of a matrix
	 */
	static ReturnType compute(SGMatrix<T> x)
	{
		REQUIRE(x.num_rows > 0, "Please ensure that x has more than 0 rows.\n")
		REQUIRE(x.num_cols > 0, "Please ensure that x has more than 0 columns.\n")

		SGVector<T> tempVec; //used to store multiple results
		Eigen::Map<MatrixXt> eigX = x;
		MatrixXt eigSquaredResult(x.num_rows, x.num_cols);

		tempVec = rowwise_mean<Backend::EIGEN3, SGVector<T>>::compute(x, false);
		Eigen::Map<VectorXt> eigTempVec = tempVec;
		for(int i = 0; i < x.num_cols; ++i)
		{
			eigSquaredResult.col(i) = eigTempVec;
		}
		eigSquaredResult = (eigX - eigSquaredResult).array().square();

		eigTempVec = eigSquaredResult.rowwise().sum();
		eigTempVec *= ((T) 1 / (x.num_cols -1));
		return  tempVec;
	}
};

}

}

}


#endif /* SRC_SHOGUN_MATHEMATICS_LINALG_INTERNAL_IMPLEMENTATION_VARIANCE_H_ */
