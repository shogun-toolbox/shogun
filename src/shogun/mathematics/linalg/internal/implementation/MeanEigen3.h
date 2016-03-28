/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Pan Deng
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

#ifndef MEAN_EIGEN_H_
#define MEAN_EIGEN_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/linalg/internal/implementation/Sum.h>

#include <shogun/mathematics/eigen3.h>
#include <numeric>

#include <iostream>

namespace shogun
{

namespace linalg
{
    
namespace implementation
{
    
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
 * @brief Generic class mean which provides a static compute method. This class
 * is specialized for different types of vectors and matrices, providing a mean
 * to deal with various vectors and matrices directly.
 */
template <enum Backend, class Matrix>
struct mean
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** int2float type */
	typedef typename int2float<T>::floatType ReturnType;

	/**
	 * Method that computes the vector/matrix mean
	 *
	 * @param a vector/matrix whose mean has to be computed
	 * @return the vector/matrix mean \f$\mean_i a_i\f$
	 */
	static ReturnType compute(Matrix a, bool no_diag=false);
};

/**
 * @brief Specialization of generic mean which works with SGVector and 
 * SGMatrix and uses Eigen3 as backend for computing mean.
 */
template <class Matrix>
struct mean<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** int2float type */
	typedef typename int2float<T>::floatType ReturnType;

	/**
	 * Method that computes the mean of SGVectors using Eigen3
	 *
	 * @param a vector whose mean has to be computed
	 * @return the vector mean \f$\mean_i a_i\f$
	 */
	static ReturnType compute(SGVector<T> vec)
	{
		REQUIRE(vec.vlen > 0, "Vector cannot be empty!\n");

		ReturnType result = 0;
		for(int i = 0; i< vec.vlen; ++i)
			result += (((ReturnType) vec[i]) / vec.vlen);

		return result;
	}
}; 

/**
 * @Brief A generic struct that contains two methods for computing the mean of
 * a matrix.  There's one method for computing column-wise and row-wise means and one
 * method for computing element-wise means
 */
template<enum Backend, typename Matrix>
struct matrix_mean{

	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Vector return type */
	typedef SGVector<T> ReturnTypeVec;

	/**
	 * Method that can compute an column-wise or row-wise mean
	 * for a matrix
	 *
	 * @param m the matrix whose mean we want to compute
	 * @param col_wise if true, we compute the column wise mean -
	 * otherwise, we compute the row-wise mean
	 * @return a vector containing the row-wise / col-wise mean
	 */
	static ReturnTypeVec compute(Matrix m, bool col_wise);

	/**
	 * Method that computes the element-wise mean of a matrix
	 *
	 * @param m the matrix whose element-wise mean we want to
	 * compute
	 * @return the element-wise mean of m
	 */
	static T compute(Matrix m);
};

/**
 * @Brief A specialization of matrix_mean that uses SGMatrix and SGVector as it's types
 * and uses Eigen3 as it's backend component
 */
template<typename Matrix>
struct matrix_mean<Backend::EIGEN3, Matrix>{

	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Vector return type */
	typedef SGVector<T> ReturnTypeVec;

	/**
	 * Method that can compute an column-wise or row-wise mean
	 * for a matrix
	 *
	 * @param m the matrix whose mean we want to compute
	 * @param col_wise if true, we compute the column wise mean -
	 * otherwise, we compute the row-wise mean
	 * @return a vector containing the row-wise / col-wise mean
	 */
	static ReturnTypeVec compute(SGMatrix<T> m, bool col_wise){

		ASSERT(m.num_rows>0)
		ASSERT(m.num_cols>0)
		ASSERT(m.matrix)

		ReturnTypeVec result;

		if(col_wise){

			result = colwise_sum<Backend::EIGEN3, Matrix>::compute(m, false);
			for(int i = 0; i < m.num_cols; ++i)
				result[i] /= m.num_rows;
		}
		else{

			result = rowwise_sum<Backend::EIGEN3, Matrix>::compute(m, false);
			for(int i = 0; i < m.num_rows; ++i)
				result[i] /= m.num_cols;
		}

		return result;
	}

	/**
	 * Method that computes the element-wise mean of a matrix
	 *
	 * @param m the matrix whose element-wise mean we want to
	 * compute
	 * @return the element-wise mean of m
	 */
	static T compute(SGMatrix<T> m){

		ASSERT(m.num_rows>0)
		ASSERT(m.num_cols>0)
		ASSERT(m.matrix)

		return sum<Backend::EIGEN3, Matrix>::compute(m, false) / (m.num_rows * m.num_cols);
	}
};

}
             
}

}

#endif
