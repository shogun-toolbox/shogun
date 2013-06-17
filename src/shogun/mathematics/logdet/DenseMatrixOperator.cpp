/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/common.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;
#endif // HAVE_EIGEN3

namespace shogun
{

template<class T>
CDenseMatrixOperator<T>::CDenseMatrixOperator()
	: CLinearOperator<T>()
	{
		SG_SGCDEBUG("%s created (%p)\n", this->get_name(), this);
	}

template<class T>
CDenseMatrixOperator<T>::CDenseMatrixOperator(SGMatrix<T> op)
	: CLinearOperator<T>(op.num_cols), m_operator(op)
	{
		SG_SGCDEBUG("%s created (%p)\n", this->get_name(), this);
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

#ifdef HAVE_EIGEN3
template<class T>
SGVector<T> CDenseMatrixOperator<T>::apply(SGVector<T> b) const
	{
		REQUIRE(m_operator.matrix, "Operator not initialized!\n");
		REQUIRE(this->get_dim()==b.vlen,
			"Number of rows of vector must be equal to the "
			"number of cols of the operator!\n");

		typedef Matrix<T, Dynamic, 1> VectorXt;
		typedef Matrix<T, Dynamic, Dynamic> MatrixXt;

		Map<VectorXt> _b(b.vector, b.vlen);
		Map<MatrixXt> _op(m_operator.matrix, m_operator.num_rows,
			m_operator.num_cols);
	
		SGVector<T> result(b.vlen);
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

#else
template<class T>
SGVector<T> CDenseMatrixOperator<T>::apply(SGVector<T> b) const
	{
		SG_SWARNING("Eigen3 required!\n");
		return b;
	}
#endif // HAVE_EIGEN3

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
template class CDenseMatrixOperator<complex64_t>;
}
