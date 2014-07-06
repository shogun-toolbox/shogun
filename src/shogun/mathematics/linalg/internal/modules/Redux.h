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

#ifndef REDUX_H_
#define REDUX_H_

#include <shogun/mathematics/linalg/internal/implementation/Dot.h>
#include <shogun/mathematics/linalg/internal/implementation/Sum.h>
#include <shogun/mathematics/linalg/internal/implementation/VectorSum.h>
#include <shogun/mathematics/linalg/internal/implementation/Square.h>

namespace shogun
{

namespace linalg
{

/**
 * Wrapper method for internal implementation of vector dot-product that works
 * with generic vectors.
 *
 * Uses globally set backend
 *
 * @param a first vector
 * @param b second vector
 * @return the dot product of \f$\mathbf{a}\f$ and \f$\mathbf{b}\f$, represented
 * as \f$\sum_i a_i b_i\f$
 */
template <class Vector>
typename Vector::Scalar dot(const Vector& a, const Vector& b)
{
	return implementation::dot<linalg_traits<Redux>::backend,Vector>::compute(a, b);
}

/**
 * Wrapper method for internal implementation of vector dot-product that works
 * with generic vectors.
 *
 * Uses templated specified backend
 *
 * @param a first vector
 * @param b second vector
 * @return the dot product of \f$\mathbf{a}\f$ and \f$\mathbf{b}\f$, represented
 * as \f$\sum_i a_i b_i\f$
 */
template <Backend backend, class Vector>
typename Vector::Scalar dot(const Vector& a, const Vector& b)
{
	return implementation::dot<backend,Vector>::compute(a, b);
}

/**
 * Wrapper method for internal implementation of matrix sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param m the matrix whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
T sum(Matrix<T,Info...> m, bool no_diag=false)
{
	return implementation::sum<int,linalg_traits<Redux>::backend,Matrix,T,Info...>
		::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of matrix-block sum of values that works
 * with generic dense matrix blocks with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param b the matrix-block whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
T sum(Block<int,Matrix,T,Info...> b, bool no_diag=false)
{
	return implementation::sum<int,linalg_traits<Redux>::backend,Matrix,T,Info...>
		::compute(b, no_diag);
}

/**
 * Wrapper method for internal implementation of symmetric matrix sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param m the matrix whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
T sum_symmetric(Matrix<T,Info...> m, bool no_diag=false)
{
	return implementation::sum_symmetric<int,linalg_traits<Redux>::backend,Matrix,T,Info...>
		::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of symmetric matrix-block sum of values that works
 * with generic dense matrix blocks with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param b the matrix-block whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
T sum_symmetric(Block<int,Matrix,T,Info...> b, bool no_diag=false)
{
	return implementation::sum_symmetric<int,linalg_traits<Redux>::backend,Matrix,T,Info...>
		::compute(b, no_diag);
}

/**
 * Wrapper method for internal implementation of matrix colwise sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param m the matrix whose colwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}m_{i,j}\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
SGVector<T> colwise_sum(Matrix<T,Info...> m, bool no_diag=false)
{
	return implementation::colwise_sum<int,linalg_traits<Redux>::backend,Matrix,T,Info...>
		::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of block colwise sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param b the matrix-block whose colwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
SGVector<T> colwise_sum(Block<int,Matrix,T,Info...> b, bool no_diag=false)
{
	return implementation::colwise_sum<int,linalg_traits<Redux>::backend,Matrix,T,Info...>
		::compute(b, no_diag);
}

/**
 * Wrapper method for internal implementation of matrix rowwise sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param m the matrix whose rowwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}m_{i,j}\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
SGVector<T> rowwise_sum(Matrix<T,Info...> m, bool no_diag=false)
{
	return implementation::rowwise_sum<int,linalg_traits<Redux>::backend,Matrix,T,Info...>
		::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of block rowwise sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param b the matrix-block whose rowwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}b_{i,j}\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
SGVector<T> rowwise_sum(Block<int,Matrix,T,Info...> b, bool no_diag=false)
{
	return implementation::rowwise_sum<int,linalg_traits<Redux>::backend,Matrix,T,Info...>
		::compute(b, no_diag);
}

/**
 * Wrapper method for internal implementation of matrix sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param m the matrix whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
 */
template <Backend backend, template <class,int...> class Matrix, class T, int... Info>
T sum(Matrix<T,Info...> m, bool no_diag=false)
{
	return implementation::sum<int,backend,Matrix,T,Info...>::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of symmetric matrix sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param m the matrix whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
 */
template <Backend backend,template <class,int...> class Matrix, class T, int... Info>
T sum_symmetric(Matrix<T,Info...> m, bool no_diag=false)
{
	return implementation::sum_symmetric<int,backend,Matrix,T,Info...>::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of matrix-block sum of values that works
 * with generic dense matrix blocks with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param b the matrix-block whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
 */
template <Backend backend,template <class,int...> class Matrix, class T, int... Info>
T sum(Block<int,Matrix,T,Info...> b, bool no_diag=false)
{
	return implementation::sum<int,backend,Matrix,T,Info...>::compute(b, no_diag);
}

/**
 * Wrapper method for internal implementation of symmetric matrix-block sum of values that works
 * with generic dense matrix blocks with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param b the matrix-block whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
 */
template <Backend backend,template <class,int...> class Matrix, class T, int... Info>
T sum_symmetric(Block<int,Matrix,T,Info...> b, bool no_diag=false)
{
	return implementation::sum_symmetric<int,backend,Matrix,T,Info...>
		::compute(b, no_diag);
}

/**
 * Wrapper method for internal implementation of matrix colwise sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param m the matrix whose colwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}m_{i,j}\f$
 */
template <Backend backend,template <class,int...> class Matrix, class T, int... Info>
SGVector<T> colwise_sum(Matrix<T,Info...> m, bool no_diag=false)
{
	return implementation::colwise_sum<int,backend,Matrix,T,Info...>::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of block colwise sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param b the matrix-block whose colwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
 */
template <Backend backend,template <class,int...> class Matrix, class T, int... Info>
SGVector<T> colwise_sum(Block<int,Matrix,T,Info...> b, bool no_diag=false)
{
	return implementation::colwise_sum<int,backend,Matrix,T,Info...>::compute(b, no_diag);
}

/**
 * Wrapper method for internal implementation of matrix rowwise sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param m the matrix whose rowwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}m_{i,j}\f$
 */
template <Backend backend,template <class,int...> class Matrix, class T, int... Info>
SGVector<T> rowwise_sum(Matrix<T,Info...> m, bool no_diag=false)
{
	return implementation::rowwise_sum<int,backend,Matrix,T,Info...>::compute(m, no_diag);
}

/**
 * Wrapper method for internal implementation of block rowwise sum of values that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix etc
 *
 * @param b the matrix-block whose rowwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}b_{i,j}\f$
 */
template <Backend backend,template <class,int...> class Matrix, class T, int... Info>
SGVector<T> rowwise_sum(Block<int,Matrix,T,Info...> b, bool no_diag=false)
{
	return implementation::rowwise_sum<int,backend,Matrix,T,Info...>::compute(b, no_diag);
}

/**
 * Wrapper method for internal implementation of square of co-efficients that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param m the matrix whose squared co-efficients matrix has to be computed
 * @return another matrix whose co-efficients are \f$m'_{i,j}=m_(i,j}^2\f$
 * for all \f$i,j\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
Matrix<T,Info...> square(Matrix<T,Info...> m)
{
	return implementation::square<int,linalg_traits<Redux>::backend,Matrix,T,Info...>::compute(m);
}

/**
 * Wrapper method for internal implementation of square of co-efficients that works
 * with generic dense matrix blocks with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param b the matrix-block whose squared co-efficients matrix has to be computed
 * @return another matrix whose co-efficients are \f$m'_{i,j}=b_(i,j}^2\f$
 * for all \f$i,j\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
Matrix<T,Info...> square(Block<int,Matrix,T,Info...> b)
{
	return implementation::square<int,linalg_traits<Redux>::backend,Matrix,T,Info...>::compute(b);
}

/**
 * Wrapper method for internal implementation of square of co-efficients that works
 * with generic dense matrices with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param m the matrix whose squared co-efficients matrix has to be computed
 * @return another matrix whose co-efficients are \f$m'_{i,j}=m_(i,j}^2\f$
 * for all \f$i,j\f$
 */
template <Backend backend,template <class,int...> class Matrix, class T, int... Info>
Matrix<T,Info...> square(Matrix<T,Info...> m)
{
	return implementation::square<int,backend,Matrix,T,Info...>::compute(m);
}

/**
 * Wrapper method for internal implementation of square of co-efficients that works
 * with generic dense matrix blocks with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param b the matrix-block whose squared co-efficients matrix has to be computed
 * @return another matrix whose co-efficients are \f$m'_{i,j}=b_(i,j}^2\f$
 * for all \f$i,j\f$
 */
template <Backend backend,template <class,int...> class Matrix, class T, int... Info>
Matrix<T,Info...> square(Block<int,Matrix,T,Info...> b)
{
	return implementation::square<int,backend,Matrix,T,Info...>::compute(b);
}


/**
 * Wrapper method for internal implementation of vector sum of values that works
 * with generic dense vectors with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGVector, Eigen3's Vector etc
 *
 * @param a vector whose sum has to be computed
 * @return the vector sum \f$\sum_i a_i\f$
 */
template <template <class,int...> class Vector, class T, int... Info>
T vector_sum(Vector<T,Info...> a)
{
	return implementation::vector_sum<int,linalg_traits<Redux>::backend,Vector,T,Info...>
		::compute(a);
}

/**
 * Wrapper method for internal implementation of vector sum of values that works
 * with generic dense vectors with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGVector, Eigen3's Vector etc
 *
 * @param a vector whose sum has to be computed
 * @return the vector sum \f$\sum_i a_i\f$
 */
template <Backend backend,template <class,int...> class Vector, class T, int... Info>
T vector_sum(Vector<T,Info...> a)
{
	return implementation::vector_sum<int,backend,Vector,T,Info...>::compute(a);
}

}

}
#endif // REDUX_H_
