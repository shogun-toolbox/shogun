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

namespace shogun
{

namespace linalg
{

/**
 * Wrapper method for internal implementation of vector dot-product that works
 * with generic vectors with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGVector, Eigen3's Vector etc
 *
 * @param \f$\mathbf{a}\f$ first vector
 * @param \f$\mathbf{b}\f$ second vector
 * @return the dot product of \f$\mathbf{a}\f$ and \$\mathbf{b}\f$, represented
 * as \f$\sum_i a_i b_i\f$
 */
template <template <class,int...> class Vector, class T, int... Info>
T dot(Vector<T,Info...> a, Vector<T,Info...> b)
{
	return implementation::dot<int,linalg_traits<Redux>::backend,Vector,T,Info...>::compute(a, b);
}

/**
 * Wrapper method for internal implementation of vector dot-product that works
 * with generic vectors with first templated-argument as its value-type and
 * other (optional) templated-arguments of unsigned int type for compile time
 * information
 *
 * Uses globally set backend
 *
 * Suited for ViennaCL vectors
 *
 * @param \f$\mathbf{a}\f$ first vector
 * @param \f$\mathbf{b}\f$ second vector
 * @return the dot product of \f$\mathbf{a}\f$ and \$\mathbf{b}\f$, represented
 * as \f$\sum_i a_i b_i\f$
 */
template <template <class,unsigned int> class Vector, class T, unsigned int Info>
T dot(Vector<T,Info> a, Vector<T,Info> b)
{
	return implementation::dot<unsigned int,linalg_traits<Redux>::backend,Vector,T,Info>::compute(a, b);
}

/**
 * Wrapper method for internal implementation of vector dot-product that works
 * with generic vectors with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGVector, Eigen3's Vector etc
 *
 * @param \f$\mathbf{a}\f$ first vector
 * @param \f$\mathbf{b}\f$ second vector
 * @return the dot product of \f$\mathbf{a}\f$ and \$\mathbf{b}\f$, represented
 * as \f$\sum_i a_i b_i\f$
 */
template <Backend backend,template <class,int...> class Vector, class T, int... Info>
T dot(Vector<T,Info...> a, Vector<T,Info...> b)
{
	return implementation::dot<int,backend,Vector,T,Info...>::compute(a, b);
}

/**
 * Wrapper method for internal implementation of vector dot-product that works
 * with generic vectors with first templated-argument as its value-type and
 * other (optional) templated-arguments of unsigned int type for compile time
 * information
 *
 * Uses templated specified backend
 *
 * Suited for ViennaCL vectors
 *
 * @param \f$\mathbf{a}\f$ first vector
 * @param \f$\mathbf{b}\f$ second vector
 * @return the dot product of \f$\mathbf{a}\f$ and \$\mathbf{b}\f$, represented
 * as \f$\sum_i a_i b_i\f$
 */
template <Backend backend,template <class,unsigned int> class Vector, class T, unsigned int Info>
T dot(Vector<T,Info> a, Vector<T,Info> b)
{
	return implementation::dot<unsigned int,backend,Vector,T,Info>::compute(a, b);
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
 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
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
 * with generic dense matricx blocks with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
T sum(Block<int,Matrix,T,Info...> block, bool no_diag=false)
{
	return implementation::sum<int,linalg_traits<Redux>::backend,Matrix,T,Info...>
		::compute(block, no_diag);
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
 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
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
 * with generic dense matricx blocks with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses globally set backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param \f$\mathbf{m}\f$ the matrix-block whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
 */
template <template <class,int...> class Matrix, class T, int... Info>
T sum_symmetric(Block<int,Matrix,T,Info...> block, bool no_diag=false)
{
	return implementation::sum_symmetric<int,linalg_traits<Redux>::backend,Matrix,T,Info...>
		::compute(block, no_diag);
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
 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
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
 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
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
 * with generic dense matricx blocks with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
 */
template <Backend backend,template <class,int...> class Matrix, class T, int... Info>
T sum(Block<int,Matrix,T,Info...> block, bool no_diag=false)
{
	return implementation::sum<int,backend,Matrix,T,Info...>::compute(block, no_diag);
}

/**
 * Wrapper method for internal implementation of symmetric matrix-block sum of values that works
 * with generic dense matricx blocks with first templated-argument as its value-type and
 * other (optional) templated-arguments of int type for compile time information
 *
 * Uses templated specified backend
 *
 * Suited for Shogun's SGMatrix, Eigen3's Matrix blocks etc
 *
 * @param \f$\mathbf{m}\f$ the matrix-block whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum (default - false)
 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
 */
template <Backend backend,template <class,int...> class Matrix, class T, int... Info>
T sum_symmetric(Block<int,Matrix,T,Info...> block, bool no_diag=false)
{
	return implementation::sum_symmetric<int,backend,Matrix,T,Info...>
		::compute(block, no_diag);
}

}

}
#endif // REDUX_H_
