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
#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/internal/Block.h>

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
	 * @param m the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(matrix_type m, bool no_diag);

	/**
	 * Method that computes the sum of co-efficients of a symmetric dense matrix blocks
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<Info,Matrix,T,I...> b, bool no_diag);
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
	 * @param m the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(matrix_type m, bool no_diag);

	/**
	 * Method that computes the sum of co-efficients of a symmetric dense matrix blocks
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<Info,Matrix,T,I...> b, bool no_diag);
};

/**
 * @brief Generic class colwise_sum which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a mean
 * to deal with various matrices directly without having to convert
 */
template <class Info,enum Backend,template<class,Info...>class Matrix,class T,Info... I>
struct colwise_sum
{
	typedef Matrix<T,I...> matrix_type;

	/**
	 * Method that computes column wise sum of co-efficients of a dense matrix
	 *
	 * @param m the matrix whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}m_{i,j}\f$
	 */
	static SGVector<T> compute(matrix_type m, bool no_diag);

	/**
	 * Method that computes column wise sum of co-efficients of a symmetric dense
	 * matrix blocks
	 *
	 * @param b the matrix-block whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<Info,Matrix,T,I...> b, bool no_diag);
};

/**
 * @brief Generic class rowwise_sum which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a mean
 * to deal with various matrices directly without having to convert
 */
template <class Info,enum Backend,template<class,Info...>class Matrix,class T,Info... I>
struct rowwise_sum
{
	typedef Matrix<T,I...> matrix_type;

	/**
	 * Method that computes row wise sum of co-efficients of a dense matrix
	 *
	 * @param m the matrix whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}m_{i,j}\f$
	 */
	static SGVector<T> compute(matrix_type m, bool no_diag);

	/**
	 * Method that computes row wise sum of co-efficients of a symmetric dense
	 * matrix blocks
	 *
	 * @param b the matrix-block whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<Info,Matrix,T,I...> b, bool no_diag);
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
	 * @param m the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(matrix_type m, bool no_diag)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(m.matrix, m.num_rows, m.num_cols);

		return sum<int,Backend::EIGEN3,Eigen::Matrix,T,Eigen::Dynamic,Eigen::Dynamic>
			::compute(eig_m, no_diag);
	}

	/**
	 * Method that computes the sum of co-efficients of SGMatrix blocks using Eigen3
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<int,shogun::SGMatrix,T> b, bool no_diag)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(b.m_matrix.matrix, b.m_matrix.num_rows,
				b.m_matrix.num_cols);

		const MatrixXt& block=eig_m.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);

		return sum<int,Backend::EIGEN3,Eigen::Matrix,T,Eigen::Dynamic,Eigen::Dynamic>
			::compute(block, no_diag);
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
	 * @param m the matrix whose sum of co-efficients has to be computed
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

	/**
	 * Method that computes the sum of co-efficients of Eigen3 Matrix blocks using Eigen3
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<int,Eigen::Matrix,T,Info...> b, bool no_diag)
	{
		const matrix_type& block=b.m_matrix.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);

		return compute(block, no_diag);
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
	 * @param m the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(matrix_type m, bool no_diag)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(m.matrix, m.num_rows, m.num_cols);

		return sum_symmetric<int,Backend::EIGEN3,Eigen::Matrix,T,Eigen::Dynamic,Eigen::Dynamic>
			::compute(eig_m, no_diag);
	}

	/**
	 * Method that computes the sum of co-efficients of symmetric SGMatrix blocks using Eigen3
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<int,shogun::SGMatrix,T> b, bool no_diag)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(b.m_matrix.matrix, b.m_matrix.num_rows,
				b.m_matrix.num_cols);

		const MatrixXt& block=eig_m.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);

		return sum_symmetric<int,Backend::EIGEN3,Eigen::Matrix,T,Eigen::Dynamic,Eigen::Dynamic>
			::compute(block, no_diag);
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
	 * @param m the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(matrix_type m, bool no_diag)
	{
		REQUIRE(m.rows()==m.cols(), "Matrix is not square!\n");

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

	/**
	 * Method that computes the sum of co-efficients of symmetric Eigen3 Matrix blocks using Eigen3
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<int,Eigen::Matrix,T,Info...> b, bool no_diag)
	{
		const matrix_type& block=b.m_matrix.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);

		return compute(block, no_diag);
	}
};

/**
 * @brief Specialization of generic colwise_sum which works with SGMatrix and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class T>
struct colwise_sum<int,Backend::EIGEN3,shogun::SGMatrix,T>
{
	typedef shogun::SGMatrix<T> matrix_type;

	/**
	 * Method that computes the column wise sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}m_{i,j}\f$
	 */
	static SGVector<T> compute(matrix_type m, bool no_diag)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(m.matrix, m.num_rows, m.num_cols);

		return colwise_sum<int,Backend::EIGEN3,Eigen::Matrix,T,Eigen::Dynamic,Eigen::Dynamic>
			::compute(eig_m, no_diag);
	}

	/**
	 * Method that computes the column wise sum of co-efficients of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<int,shogun::SGMatrix,T> b, bool no_diag)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(b.m_matrix.matrix, b.m_matrix.num_rows,
				b.m_matrix.num_cols);

		const MatrixXt& block=eig_m.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);

		return colwise_sum<int,Backend::EIGEN3,Eigen::Matrix,T,Eigen::Dynamic,Eigen::Dynamic>
			::compute(block, no_diag);
	}
};

/**
 * @brief Specialization of generic colwise_sum which works with Eigen3 and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class T,int...Info>
struct colwise_sum<int,Backend::EIGEN3,Eigen::Matrix,T,Info...>
{
	typedef Eigen::Matrix<T,Info...> matrix_type;

	/**
	 * Method that computes the column wise sum of co-efficients of Eigen3 Matrix
	 * using Eigen3
	 *
	 * @param m the matrix whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}m_{i,j}\f$
	 */
	static SGVector<T> compute(matrix_type m, bool no_diag)
	{
		Eigen::Matrix<T,Eigen::Dynamic,1> sum=m.colwise().sum();

		// remove the main diagonal elements if required
		if (no_diag)
		{
			index_t len_major_diag=m.rows() < m.cols() ? m.rows() : m.cols();
			for (index_t i=0; i<len_major_diag; ++i)
				sum[i]-=m(i,i);
		}

		SGVector<T> s(sum.rows());
		std::template copy(sum.data(), sum.data()+sum.size(), s.vector);

		return s;
	}

	/**
	 * Method that computes the sum of co-efficients of Eigen3 Matrix blocks using Eigen3
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<int,Eigen::Matrix,T,Info...> b, bool no_diag)
	{
		const matrix_type& block=b.m_matrix.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);

		return compute(block, no_diag);
	}
};

/**
 * @brief Specialization of generic rowwise_sum which works with SGMatrix and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class T>
struct rowwise_sum<int,Backend::EIGEN3,shogun::SGMatrix,T>
{
	typedef shogun::SGMatrix<T> matrix_type;

	/**
	 * Method that computes the row wise sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}m_{i,j}\f$
	 */
	static SGVector<T> compute(matrix_type m, bool no_diag)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(m.matrix, m.num_rows, m.num_cols);

		return rowwise_sum<int,Backend::EIGEN3,Eigen::Matrix,T,Eigen::Dynamic,Eigen::Dynamic>
			::compute(eig_m, no_diag);
	}

	/**
	 * Method that computes the row wise sum of co-efficients of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<int,shogun::SGMatrix,T> b, bool no_diag)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		Eigen::Map<MatrixXt> eig_m(b.m_matrix.matrix, b.m_matrix.num_rows,
				b.m_matrix.num_cols);

		const MatrixXt& block=eig_m.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);

		return rowwise_sum<int,Backend::EIGEN3,Eigen::Matrix,T,Eigen::Dynamic,Eigen::Dynamic>
			::compute(block, no_diag);
	}
};

/**
 * @brief Specialization of generic rowwise_sum which works with Eigen3 and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class T,int...Info>
struct rowwise_sum<int,Backend::EIGEN3,Eigen::Matrix,T,Info...>
{
	typedef Eigen::Matrix<T,Info...> matrix_type;

	/**
	 * Method that computes the row wise sum of co-efficients of Eigen3 Matrix
	 * using Eigen3
	 *
	 * @param m the matrix whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_j=\sum_{i}m_{i,j}\f$
	 */
	static SGVector<T> compute(matrix_type m, bool no_diag)
	{
		Eigen::Matrix<T,Eigen::Dynamic,1> sum=m.rowwise().sum();

		// remove the main diagonal elements if required
		if (no_diag)
		{
			index_t len_major_diag=m.rows() < m.cols() ? m.rows() : m.cols();
			for (index_t i=0; i<len_major_diag; ++i)
				sum[i]-=m(i,i);
		}

		SGVector<T> s(sum.rows());
		std::template copy(sum.data(), sum.data()+sum.size(), s.vector);

		return s;
	}

	/**
	 * Method that computes the sum of co-efficients of Eigen3 Matrix blocks using Eigen3
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<int,Eigen::Matrix,T,Info...> b, bool no_diag)
	{
		const matrix_type& block=b.m_matrix.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);

		return compute(block, no_diag);
	}
};

#endif // HAVE_EIGEN3

}

}

}
#endif // SUM_IMPL_H_
