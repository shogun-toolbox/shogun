/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>

using namespace Eigen;

namespace shogun
{

template<class T>
CDenseMatrixOperator<T>::CDenseMatrixOperator()
	: CMatrixOperator<T>()
	{
		init();

		SG_SGCDEBUG("%s created (%p)\n", this->get_name(), this);
	}

template<class T>
CDenseMatrixOperator<T>::CDenseMatrixOperator(SGMatrix<T> op)
	: CMatrixOperator<T>(op.num_cols),
	  m_operator(op)
	{
		init();

		SG_SGCDEBUG("%s created (%p)\n", this->get_name(), this);
	}

template<class T>
CDenseMatrixOperator<T>::CDenseMatrixOperator(
	const CDenseMatrixOperator<T>& orig)
	: CMatrixOperator<T>(orig.get_dimension())
	{
		init();

		m_operator=SGMatrix<T>(orig.m_operator.num_rows, orig.m_operator.num_cols);
		for (index_t i=0; i<m_operator.num_cols; ++i)
		{
			for (index_t j=0; j<m_operator.num_rows; ++j)
				m_operator(j,i)=orig.m_operator(j,i);
		}

		SG_SGCDEBUG("%s deep copy created (%p)\n", this->get_name(), this);
	}

template<class T>
void CDenseMatrixOperator<T>::init()
	{
		CSGObject::set_generic<T>();

		this->m_parameters->add(&m_operator, "dense_matrix",
				"The dense matrix of the linear operator");
	}

template<class T>
CDenseMatrixOperator<T>::~CDenseMatrixOperator()
	{
		SG_SGCDEBUG("%s destroyed (%p)\n", this->get_name(), this);
	}

template<class T>
SGMatrix<T> CDenseMatrixOperator<T>::get_matrix_operator() const
	{
		return m_operator;
	}

template<class T>
SGVector<T> CDenseMatrixOperator<T>::get_diagonal() const
	{
		REQUIRE(m_operator.matrix, "Operator not initialized!\n");

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
void CDenseMatrixOperator<T>::set_diagonal(SGVector<T> diag)
	{
		REQUIRE(m_operator.matrix, "Operator not initialized!\n");
		REQUIRE(diag.vector, "Diagonal not initialized!\n");

		typedef Matrix<T, Dynamic, 1> VectorXt;
		typedef Matrix<T, Dynamic, Dynamic> MatrixXt;

		Map<MatrixXt> _op(m_operator.matrix, m_operator.num_rows,
			m_operator.num_cols);

		REQUIRE(static_cast<int32_t>(_op.diagonalSize())==diag.vlen,
			"Dimension mismatch!\n");

		Map<VectorXt> _diag(diag.vector, diag.vlen);
		_op.diagonal()=_diag;
	}

template<class T>
SGVector<T> CDenseMatrixOperator<T>::apply(SGVector<T> b) const
	{
		REQUIRE(m_operator.matrix, "Operator not initialized!\n");
		REQUIRE(this->get_dimension()==b.vlen,
			"Number of rows of vector must be equal to the "
			"number of cols of the operator!\n");

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
SGVector<type> CDenseMatrixOperator<type>::apply(SGVector<type> b) const \
	{	\
		SG_SERROR("Not supported for %s\n", #type);\
		return b; \
	}

UNDEFINED(bool)
UNDEFINED(char)
UNDEFINED(int8_t)
UNDEFINED(uint8_t)
#undef UNDEFINED

template class CDenseMatrixOperator<bool>;
template class CDenseMatrixOperator<char>;
template class CDenseMatrixOperator<int8_t>;
template class CDenseMatrixOperator<uint8_t>;
template class CDenseMatrixOperator<int16_t>;
template class CDenseMatrixOperator<uint16_t>;
template class CDenseMatrixOperator<int32_t>;
template class CDenseMatrixOperator<uint32_t>;
template class CDenseMatrixOperator<int64_t>;
template class CDenseMatrixOperator<uint64_t>;
template class CDenseMatrixOperator<float32_t>;
template class CDenseMatrixOperator<float64_t>;
template class CDenseMatrixOperator<floatmax_t>;
template class CDenseMatrixOperator<complex128_t>;
}
#endif // HAVE_EIGEN3
