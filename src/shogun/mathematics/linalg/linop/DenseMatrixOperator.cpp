/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>

using namespace Eigen;

namespace shogun
{

template<class T>
DenseMatrixOperator<T>::DenseMatrixOperator()
	: MatrixOperator<T>()
	{
		init();

		SG_TRACE("{} created ({})", this->get_name(), fmt::ptr(this));
	}

template<class T>
DenseMatrixOperator<T>::DenseMatrixOperator(SGMatrix<T> op)
	: MatrixOperator<T>(op.num_cols),
	  m_operator(op)
	{
		init();

		SG_TRACE("{} created ({})", this->get_name(), fmt::ptr(this));
	}

template<class T>
DenseMatrixOperator<T>::DenseMatrixOperator(
	const DenseMatrixOperator<T>& orig)
	: MatrixOperator<T>(orig.get_dimension())
	{
		init();

		m_operator=SGMatrix<T>(orig.m_operator.num_rows, orig.m_operator.num_cols);
		for (index_t i=0; i<m_operator.num_cols; ++i)
		{
			for (index_t j=0; j<m_operator.num_rows; ++j)
				m_operator(j,i)=orig.m_operator(j,i);
		}

		SG_TRACE("{} deep copy created ({})", this->get_name(), fmt::ptr(this));
	}

template<class T>
void DenseMatrixOperator<T>::init()
	{
		SGObject::set_generic<T>();
	}

template<class T>
DenseMatrixOperator<T>::~DenseMatrixOperator()
	{
		SG_TRACE("{} destroyed ({})", this->get_name(), fmt::ptr(this));
	}

template<class T>
SGMatrix<T> DenseMatrixOperator<T>::get_matrix_operator() const
	{
		return m_operator;
	}

template<class T>
SGVector<T> DenseMatrixOperator<T>::get_diagonal() const
	{
		require(m_operator.matrix, "Operator not initialized!");

		typedef Matrix<T, Dynamic, 1> VectorXt;
		typedef Matrix<T, Dynamic, Dynamic> MatrixXt;

		Map<MatrixXt> _op(m_operator.matrix, m_operator.num_rows,
			m_operator.num_cols);

		SGVector<T> diag(static_cast<int32_t>(_op.diagonalSize()));
		Map<VectorXt> _diag(diag.vector, diag.vlen);
		_diag=_op.diagonal();

		return diag;
	}

template<class T>
void DenseMatrixOperator<T>::set_diagonal(SGVector<T> diag)
	{
		require(m_operator.matrix, "Operator not initialized!");
		require(diag.vector, "Diagonal not initialized!");

		typedef Matrix<T, Dynamic, 1> VectorXt;
		typedef Matrix<T, Dynamic, Dynamic> MatrixXt;

		Map<MatrixXt> _op(m_operator.matrix, m_operator.num_rows,
			m_operator.num_cols);

		require(static_cast<int32_t>(_op.diagonalSize())==diag.vlen,
			"Dimension mismatch!");

		Map<VectorXt> _diag(diag.vector, diag.vlen);
		_op.diagonal()=_diag;
	}

template<class T>
SGVector<T> DenseMatrixOperator<T>::apply(SGVector<T> b) const
	{
		require(m_operator.matrix, "Operator not initialized!");
		require(this->get_dimension()==b.vlen,
			"Number of rows of vector must be equal to the "
			"number of cols of the operator!");

		typedef Matrix<T, Dynamic, 1> VectorXt;
		typedef Matrix<T, Dynamic, Dynamic> MatrixXt;

		Map<VectorXt> _b(b.vector, b.vlen);
		Map<MatrixXt> _op(m_operator.matrix, m_operator.num_rows,
			m_operator.num_cols);

		SGVector<T> result(m_operator.num_rows);
		Map<VectorXt> _result(result.vector, result.vlen);
		_result=_op*_b;

		return result;
	}

#define UNDEFINED(type) \
template<> \
SGVector<type> DenseMatrixOperator<type>::apply(SGVector<type> b) const \
	{	\
		error("Not supported for {}", #type);\
		return b; \
	}

UNDEFINED(bool)
UNDEFINED(char)
UNDEFINED(int8_t)
UNDEFINED(uint8_t)
#undef UNDEFINED

template class DenseMatrixOperator<bool>;
template class DenseMatrixOperator<char>;
template class DenseMatrixOperator<int8_t>;
template class DenseMatrixOperator<uint8_t>;
template class DenseMatrixOperator<int16_t>;
template class DenseMatrixOperator<uint16_t>;
template class DenseMatrixOperator<int32_t>;
template class DenseMatrixOperator<uint32_t>;
template class DenseMatrixOperator<int64_t>;
template class DenseMatrixOperator<uint64_t>;
template class DenseMatrixOperator<float32_t>;
template class DenseMatrixOperator<float64_t>;
template class DenseMatrixOperator<floatmax_t>;
template class DenseMatrixOperator<complex128_t>;
}
