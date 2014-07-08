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
	static T compute(const Matrix& m, bool no_diag);

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
	static T compute(const Matrix&, bool no_diag);

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
	static SGVector<T> compute(const Matrix&, bool no_diag);

	/**
	 * Method that computes column wise sum of co-efficients of dense matrix blocks
	 *
	 * @param b the matrix-block whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<Matrix> b, bool no_diag);
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
	static SGVector<T> compute(const Matrix& m, bool no_diag);

	/**
	 * Method that computes row wise sum of co-efficients of a dense matrix blocks
	 *
	 * @param b the matrix-block whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<Matrix> b, bool no_diag);
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
	static T compute(SGMatrix<T> m, bool no_diag)
	{
		return compute((Eigen::Map<MatrixXt>)m, no_diag);
	}
	
	template <class Derived>
	static T compute(const Eigen::MatrixBase<Derived>& m, bool no_diag)
	{
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

		return compute(b_eigen, no_diag);
	}
	
	template <int... Info>
	static T compute(Block<Eigen::Matrix<T,Info...> > b, bool no_diag)
	{
		Eigen::Block<Eigen::Matrix<T,Info...> > b_eigen = 
			b.m_matrix.block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);
		
		return compute(b_eigen, no_diag);
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
	static T compute(SGMatrix<T> m, bool no_diag)
	{
		Eigen::Map<MatrixXt> map = m;
		return compute((const MatrixXt&)map, no_diag);
	}
	
	template <int... Info>
	static T compute(const Eigen::Matrix<T,Info...>& m, bool no_diag)
	{
		REQUIRE(m.rows()==m.cols(), "Matrix is not square!\n");

		// since the matrix is symmetric with main diagonal inside, we can save half
		// the computation with using only the upper triangular part.
		const Eigen::Matrix<T, Info...>& m_upper=m.template triangularView<Eigen::StrictlyUpper>();
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
		
		const MatrixXt& block=map.template block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);
		
		return compute(block, no_diag);
	}
	
	template <int... Info>
	static T compute(Block<Eigen::Matrix<T,Info...> > b, bool no_diag)
	{
		const Eigen::Matrix<T,Info...>& block=b.m_matrix.template block(
			b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);
		
		return compute(block, no_diag);
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
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/**
	 * Method that computes the column wise sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}m_{i,j}\f$
	 */
	static SGVector<T> compute(SGMatrix<T> m, bool no_diag)
	{
		return compute((Eigen::Map<MatrixXt>)m, no_diag);
	}
	
	template <class Derived>
	static SGVector<T> compute(const Eigen::MatrixBase<Derived>& m, bool no_diag)
	{
		Eigen::Matrix<T,Eigen::Dynamic,1> sum = m.colwise().sum();
		
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
	 * Method that computes the column wise sum of co-efficients of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose colwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<SGMatrix<T> > b, bool no_diag)
	{
		Eigen::Map<MatrixXt> map = b.m_matrix;
		
		Eigen::Block< Eigen::Map<MatrixXt> > b_eigen = map.block(
			b.m_row_begin, b.m_col_begin,
			b.m_row_size, b.m_col_size);

		return compute(b_eigen, no_diag);
	}
	
	template <int... Info>
	static SGVector<T> compute(Block<Eigen::Matrix<T,Info...> > b, bool no_diag)
	{
		Eigen::Block<Eigen::Matrix<T,Info...> > b_eigen = 
			b.m_matrix.block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);
		
		return compute(b_eigen, no_diag);
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
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;
	
	/**
	 * Method that computes the row wise sum of co-efficients of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}m_{i,j}\f$
	 */
	static SGVector<T> compute(SGMatrix<T> m, bool no_diag)
	{
		return compute((Eigen::Map<MatrixXt>)m, no_diag);
	}
	
	template <class Derived>
	static SGVector<T> compute(const Eigen::MatrixBase<Derived>& m, bool no_diag)
	{
		Eigen::Matrix<T,Eigen::Dynamic,1> sum = m.rowwise().sum();
		
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
	 * Method that computes the row wise sum of co-efficients of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose rowwise sum of co-efficients has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the sum
	 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<SGMatrix<T> > b, bool no_diag)
	{
		Eigen::Map<MatrixXt> map = b.m_matrix;
		
		Eigen::Block< Eigen::Map<MatrixXt> > b_eigen = map.block(
			b.m_row_begin, b.m_col_begin,
			b.m_row_size, b.m_col_size);

		return compute(b_eigen, no_diag);
	}
	
	template <int... Info>
	static SGVector<T> compute(Block<Eigen::Matrix<T,Info...> > b, bool no_diag)
	{
		Eigen::Block<Eigen::Matrix<T,Info...> > b_eigen = 
			b.m_matrix.block(b.m_row_begin, b.m_col_begin,
				b.m_row_size, b.m_col_size);
		
		return compute(b_eigen, no_diag);
	}
};

#endif // HAVE_EIGEN3

}

}

}
#endif // SUM_IMPL_H_
