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

#ifndef SQUARE_IMPL_H_
#define SQUARE_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/internal/Block.h>
#include <algorithm>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

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
 * @brief Generic class square which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a mean
 * to deal with various matrices directly without having to convert
 */
template <class Info,enum Backend,template<class,Info...>class Matrix,class T,Info... I>
struct square
{
	typedef Matrix<T,I...> matrix_type;

	/**
	 * Method that computes the square of co-efficients of a dense matrix
	 *
	 * @param m the matrix whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=m_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static matrix_type compute(matrix_type m);

	/**
	 * Method that computes the square of co-efficients of a dense matrix-block
	 *
	 * @param b the matrix-block whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=b_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static matrix_type compute(Block<Matrix<T,I...> > b);
};

#ifdef HAVE_EIGEN3
/**
 * @brief Specialization of generic square which works with SGMatrix and uses Eigen3
 * as backend for computing square.
 */
template <> template <class T>
struct square<int,Backend::EIGEN3,shogun::SGMatrix,T>
{
	typedef shogun::SGMatrix<T> matrix_type;

	/**
	 * Method that computes the square of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=m_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static matrix_type compute(matrix_type m)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(m.matrix, m.num_rows, m.num_cols);

		MatrixXt sq=square<int,Backend::EIGEN3,Eigen::Matrix,T,Eigen::Dynamic,Eigen::Dynamic>
			::compute(eig_m);

		matrix_type square(m.num_rows, m.num_cols);
		std::template copy(sq.data(), sq.data()+sq.size(), square.matrix);
		return square;
	}

	/**
	 * Method that computes the square of co-efficients of SGMatrix blocks using Eigen3
	 *
	 * @param b the matrix-block whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=b_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static matrix_type compute(Block<shogun::SGMatrix<T> > b)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(b.m_matrix.matrix, b.m_matrix.num_rows,
				b.m_matrix.num_cols);

		const MatrixXt& block=eig_m.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);

		MatrixXt sq=square<int,Backend::EIGEN3,Eigen::Matrix,T,Eigen::Dynamic,Eigen::Dynamic>
			::compute(block);

		matrix_type square(block.rows(), block.cols());
		std::template copy(sq.data(), sq.data()+sq.size(), square.matrix);
		return square;
	}
};

/**
 * @brief Specialization of generic square which works with Eigen3 Matrix and uses Eigen3
 * as backend for computing square.
 */
template <> template <class T,int...Info>
struct square<int,Backend::EIGEN3,Eigen::Matrix,T,Info...>
{
	typedef Eigen::Matrix<T,Info...> matrix_type;

	/**
	 * Method that computes the square of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=m_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static matrix_type compute(matrix_type m)
	{
		return m.array().template square();
	}

	/**
	 * Method that computes the square of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=m_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static matrix_type compute(Block<Eigen::Matrix<T,Info...> > b)
	{
		const matrix_type& block=b.m_matrix.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);

		return compute(block);
	}
};

#endif // HAVE_EIGEN3

}

}

}
#endif // SQUARE_IMPL_H_
