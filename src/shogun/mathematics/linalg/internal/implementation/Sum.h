/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * Written (w) 2014 Khaled Nasr
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
template <enum Backend,class Matrix>
struct sum
{
	typedef typename Matrix::Scalar T;
	
	/**
	 * Method that computes the sum of co-efficients of a dense matrix
	 *
	 * @param m the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(Matrix m, bool no_diag);

	/**
	 * Method that computes the sum of co-efficients of dense matrix blocks
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<Matrix> b, bool no_diag);
};

/**
 * @brief Generic class sum symmetric which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a mean
 * to deal with various matrices directly without having to convert
 */
template <enum Backend,class Matrix>
struct sum_symmetric
{
	typedef typename Matrix::Scalar T;
	
	/**
	 * Method that computes the sum of co-efficients of a symmetric dense matrix
	 *
	 * @param m the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(Matrix, bool no_diag);

	/**
	 * Method that computes the sum of co-efficients of symmetric dense matrix blocks
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<Matrix> b, bool no_diag);
};

/**
 * @brief Generic class colwise_sum which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a mean
 * to deal with various matrices directly without having to convert
 */
template <enum Backend,class Matrix>
struct colwise_sum
{
	typedef typename Matrix::Scalar T;
	
	/**
	 * Method that computes column wise sum of co-efficients of a dense matrix
	 *
	 * @param m the matrix whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}m_{i,j}\f$
	 */
	static SGVector<T> compute(Matrix, bool no_diag);

	/**
	 * Method that computes column wise sum of co-efficients of dense matrix blocks
	 *
	 * @param b the matrix-block whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<Matrix> b, bool no_diag);
	
	/**
	 * Method that computes the column wise sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(SGMatrix<T> mat, SGVector<T> result, bool no_diag);
	
	/**
	 * Method that computes the column wise sum of co-efficients of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(Block<SGMatrix<T> > b, SGVector<T> result, bool no_diag);
};

/**
 * @brief Generic class rowwise_sum which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a mean
 * to deal with various matrices directly without having to convert
 */
template <enum Backend,class Matrix>
struct rowwise_sum
{
	typedef typename Matrix::Scalar T;
	
	/**
	 * Method that computes row wise sum of co-efficients of a dense matrix
	 *
	 * @param m the matrix whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}m_{i,j}\f$
	 */
	static SGVector<T> compute(Matrix m, bool no_diag);

	/**
	 * Method that computes row wise sum of co-efficients of a dense matrix blocks
	 *
	 * @param b the matrix-block whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<Matrix> b, bool no_diag);
	
	/**
	 * Method that computes the row wise sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(SGMatrix<T> mat, SGVector<T> result, bool no_diag);
	
	/**
	 * Method that computes the column wise sum of co-efficients of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(Block<SGMatrix<T> > b, SGVector<T> result, bool no_diag);
};

#ifdef HAVE_EIGEN3
/**
 * @brief Specialization of generic sum which works with SGMatrix and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class Matrix>
struct sum<Backend::EIGEN3,Matrix>
{
	typedef typename Matrix::Scalar T;
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;
	
	/**
	 * Method that computes the sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(SGMatrix<T> mat, bool no_diag)
	{
		Eigen::Map<MatrixXt> m = mat;
		
		T sum=m.sum();

		// remove the main diagonal elements if required
		if (no_diag)
			sum-=m.diagonal().sum();
		
		return sum;
	}
	
	/**
	 * Method that computes the sum of co-efficients of SGMatrix blocks using Eigen3
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<SGMatrix<T> > b, bool no_diag)
	{
		Eigen::Map<MatrixXt> map = b.m_matrix;
		
		Eigen::Block< Eigen::Map<MatrixXt> > b_eigen = map.block(
			b.m_row_begin, b.m_col_begin,
			b.m_row_size, b.m_col_size);

		T sum=b_eigen.sum();

		// remove the main diagonal elements if required
		if (no_diag)
			sum-=b_eigen.diagonal().sum();
		
		return sum;
	}
};

/**
 * @brief Specialization of generic sum symmetric which works with SGMatrix and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class Matrix>
struct sum_symmetric<Backend::EIGEN3,Matrix>
{
	typedef typename Matrix::Scalar T;
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;
	
	/**
	 * Method that computes the sum of co-efficients of symmetric SGMatrix using Eigen3
	 *
	 * @param m the matrix whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}m_{i,j}\f$
	 */
	static T compute(SGMatrix<T> mat, bool no_diag)
	{
		Eigen::Map<MatrixXt> m = mat;
		
		REQUIRE(m.rows()==m.cols(), "Matrix is not square!\n");

		// since the matrix is symmetric with main diagonal inside, we can save half
		// the computation with using only the upper triangular part.
		const MatrixXt& m_upper=m.template triangularView<Eigen::StrictlyUpper>();
		T sum=m_upper.sum();

		// the actual sum would be twice of what we computed
		sum*=2;

		// add the diagonal elements if required
		if (!no_diag)
			sum+=m.diagonal().sum();

		return sum;
	}
	
	/**
	 * Method that computes the sum of co-efficients of symmetric SGMatrix blocks using Eigen3
	 *
	 * @param b the matrix-block whose sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the sum of co-efficients computed as \f$\sum_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<SGMatrix<T> > b, bool no_diag)
	{
		Eigen::Map<MatrixXt> map = b.m_matrix;
		
		const MatrixXt& m=map.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);
		
		REQUIRE(m.rows()==m.cols(), "Matrix is not square!\n");

		// since the matrix is symmetric with main diagonal inside, we can save half
		// the computation with using only the upper triangular part.
		const MatrixXt& m_upper=m.template triangularView<Eigen::StrictlyUpper>();
		T sum=m_upper.sum();

		// the actual sum would be twice of what we computed
		sum*=2;

		// add the diagonal elements if required
		if (!no_diag)
			sum+=m.diagonal().sum();

		return sum;
	}
};

/**
 * @brief Specialization of generic colwise_sum which works with SGMatrix and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class Matrix>
struct colwise_sum<Backend::EIGEN3,Matrix>
{
	typedef typename Matrix::Scalar T;
	typedef SGVector<T> ReturnType;
	
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/**
	 * Method that computes the column wise sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}m_{i,j}\f$
	 */
	static SGVector<T> compute(SGMatrix<T> m, bool no_diag)
	{
		SGVector<T> result(m.num_cols);
		compute(m, result, no_diag);
		return result;
	}
	
	/**
	 * Method that computes the column wise sum of co-efficients of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<SGMatrix<T> > b, bool no_diag)
	{
		SGVector<T> result(b.m_col_size);
		compute(b, result, no_diag);
		return result;
	}
	
	/**
	 * Method that computes the column wise sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(SGMatrix<T> mat, SGVector<T> result, bool no_diag)
	{
		Eigen::Map<MatrixXt> m = mat;
		Eigen::Map<VectorXt> r = result;
		
		r = m.colwise().sum();
		
		// remove the main diagonal elements if required
		if (no_diag)
		{
			index_t len_major_diag=m.rows() < m.cols() ? m.rows() : m.cols();
			for (index_t i=0; i<len_major_diag; ++i)
				r[i]-=m(i,i);
		}
	}
	
	/**
	 * Method that computes the column wise sum of co-efficients of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(Block<SGMatrix<T> > b, SGVector<T> result, bool no_diag)
	{
		Eigen::Map<MatrixXt> map = b.m_matrix;
		Eigen::Map<VectorXt> r = result;
		
		Eigen::Block< Eigen::Map<MatrixXt> > m = map.block(
			b.m_row_begin, b.m_col_begin,
			b.m_row_size, b.m_col_size);

		r = m.colwise().sum();
		
		// remove the main diagonal elements if required
		if (no_diag)
		{
			index_t len_major_diag=m.rows() < m.cols() ? m.rows() : m.cols();
			for (index_t i=0; i<len_major_diag; ++i)
				r[i]-=m(i,i);
		}
	}
};

/**
 * @brief Specialization of generic rowwise_sum which works with SGMatrix and uses Eigen3
 * as backend for computing sum.
 */
template <> template <class Matrix>
struct rowwise_sum<Backend::EIGEN3,Matrix>
{
	typedef typename Matrix::Scalar T;
	typedef SGVector<T> ReturnType;
	
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/**
	 * Method that computes the row wise sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}m_{i,j}\f$
	 */
	static SGVector<T> compute(SGMatrix<T> m, bool no_diag)
	{
		SGVector<T> result(m.num_rows);
		compute(m, result, no_diag);
		return result;
	}
	
	/**
	 * Method that computes the row wise sum of co-efficients of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}m_{i,j}\f$
	 */
	static SGVector<T> compute(Block<SGMatrix<T> > b, bool no_diag)
	{
		SGVector<T> result(b.m_row_size);
		compute(b, result, no_diag);
		return result;
	}
	
	/**
	 * Method that computes the row wise sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(SGMatrix<T> mat, SGVector<T> result, bool no_diag)
	{
		Eigen::Map<MatrixXt> m = mat;
		Eigen::Map<VectorXt> r = result;
		
		r = m.rowwise().sum();
		
		// remove the main diagonal elements if required
		if (no_diag)
		{
			index_t len_major_diag=m.rows() < m.cols() ? m.rows() : m.cols();
			for (index_t i=0; i<len_major_diag; ++i)
				r[i]-=m(i,i);
		}
	}
	
	/**
	 * Method that computes the column wise sum of co-efficients of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(Block<SGMatrix<T> > b, SGVector<T> result, bool no_diag)
	{
		Eigen::Map<MatrixXt> map = b.m_matrix;
		Eigen::Map<VectorXt> r = result;
		
		Eigen::Block< Eigen::Map<MatrixXt> > m = map.block(
			b.m_row_begin, b.m_col_begin,
			b.m_row_size, b.m_col_size);

		r = m.rowwise().sum();
		
		// remove the main diagonal elements if required
		if (no_diag)
		{
			index_t len_major_diag=m.rows() < m.cols() ? m.rows() : m.cols();
			for (index_t i=0; i<len_major_diag; ++i)
				r[i]-=m(i,i);
		}
	}
};

#endif // HAVE_EIGEN3

}

}

}
#endif // SUM_IMPL_H_
