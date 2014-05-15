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

#ifndef SUM_IMPL_H_
#define SUM_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/io/SGIO.h>

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
 * @brief Generic class sum which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a mean
 * to deal with various matrices directly without having to convert
 */
template <class Info,enum Backend,template<class,Info...>class Matrix,class T,Info... I>
struct sum
{
	typedef Matrix<T,I...> matrix_type;

	/**
	 * Method that computes the sum of co-efficients of a dense matrix
	 *
	 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(matrix_type m, bool no_diag);
};

/**
 * @brief Generic class sum symmetric which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a mean
 * to deal with various matrices directly without having to convert
 */
template <class Info,enum Backend,template<class,Info...>class Matrix,class T,Info... I>
struct sum_symmetric
{
	typedef Matrix<T,I...> matrix_type;

	/**
	 * Method that computes the sum of co-efficients of a symmetric dense matrix
	 *
	 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(matrix_type m, bool no_diag);
};

#ifdef HAVE_EIGEN3
/**
 * @brief Specialization of generic sum which works with SGMatrix and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class T>
struct sum<int,Backend::EIGEN3,shogun::SGMatrix,T>
{
	typedef shogun::SGMatrix<T> matrix_type;

	/**
	 * Method that computes the sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(matrix_type m, bool no_diag)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(m.matrix, m.num_rows, m.num_cols);

		T sum=eig_m.sum();

		// remove the main diagonal elements if required
		if (no_diag)
			sum-=eig_m.diagonal().sum();

		return sum;
	}
};

/**
 * @brief Specialization of generic sum which works with Eigen3 and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class T,int...Info>
struct sum<int,Backend::EIGEN3,Eigen::Matrix,T,Info...>
{
	typedef Eigen::Matrix<T,Info...> matrix_type;

	/**
	 * Method that computes the sum of co-efficients of Eigen3 Matrix using Eigen3
	 *
	 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(matrix_type m, bool no_diag)
	{
		T sum=m.sum();

		// remove the main diagonal elements if required
		if (no_diag)
			sum-=m.diagonal().sum();

		return sum;
	}
};

/**
 * @brief Specialization of generic sum symmetric which works with SGMatrix and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class T>
struct sum_symmetric<int,Backend::EIGEN3,shogun::SGMatrix,T>
{
	typedef shogun::SGMatrix<T> matrix_type;

	/**
	 * Method that computes the sum of co-efficients of symmetric SGMatrix using Eigen3
	 *
	 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(matrix_type m, bool no_diag)
	{
		REQUIRE(m.num_rows==m.num_cols, "Matrix is not symmetric!\n");

		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(m.matrix, m.num_rows, m.num_cols);

		// since the matrix is symmetric with main diagonal inside, we can save half
		// the computation with using only the upper triangular part.
		const MatrixXt& m_upper=eig_m.template triangularView<Eigen::StrictlyUpper>();
		T sum=m_upper.sum();

		// the actual sum would be twice of what we computed
		sum*=2;

		// add the diagonal elements if required
		if (!no_diag)
			sum+=eig_m.diagonal().sum();

		return sum;
	}
};

/**
 * @brief Specialization of generic sum symmetric which works with Eigen3 and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class T,int...Info>
struct sum_symmetric<int,Backend::EIGEN3,Eigen::Matrix,T,Info...>
{
	typedef Eigen::Matrix<T,Info...> matrix_type;

	/**
	 * Method that computes the sum of co-efficients of symmetric Eigen3 Matrix using Eigen3
	 *
	 * @param \f$\mathbf{m}\f$ the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(matrix_type m, bool no_diag)
	{
		REQUIRE(m.rows()==m.cols(), "Matrix is not symmetric!\n");

		// since the matrix is symmetric with main diagonal inside, we can save half
		// the computation with using only the upper triangular part.
		const matrix_type& m_upper=m.template triangularView<Eigen::StrictlyUpper>();
		T sum=m_upper.sum();

		// the actual sum would be twice of what we computed
		sum*=2;

		// add the diagonal elements if required
		if (!no_diag)
			sum+=m.diagonal().sum();

		return sum;
	}
};
#endif // HAVE_EIGEN3

}

}

}
#endif // SUM_IMPL_H_
